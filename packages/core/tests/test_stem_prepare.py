"""Tests for StemUpmixPipeline.prepare_stems (separation-only, no routing/master).

Project preparation (apps/api worker, mode="stem_prepare") only needs stems
separated and cached for a later mix/export; it must not run routing or
mastering or write an output file.
"""
from __future__ import annotations

from dataclasses import replace
import os

import numpy as np
import soundfile as sf
from unittest.mock import patch

from upmixer.config import UpmixConfig
from upmixer.separation import render_prepared_stem_bed
from upmixer.separation.stem_pipeline import StemUpmixPipeline
from upmixer.separation.stem_store import PlainStemStore

_EXEC_PLAN = "upmixer.separation.stem_pipeline_exec.execute_plan"

SR = 48_000


def _sine(n: int, freq: float = 440.0, amp: float = 0.3) -> np.ndarray:
    t = np.linspace(0, n / SR, n, endpoint=False)
    ch = (amp * np.sin(2 * np.pi * freq * t)).astype(np.float64)
    return np.column_stack([ch, ch])


def _fake_execute_plan(get_separator, plan, sep_path, sep_sr, stage_callback=None,
                       cfg=None, resume_key=None, **kwargs):
    from upmixer.separation.stem_pipeline_exec import cacheable_plan_stems

    audio, _ = sf.read(sep_path, dtype="float32", always_2d=True)
    n = len(audio)
    names = (
        cacheable_plan_stems(plan, retain_private=True)
        if kwargs.get("retain_private")
        else plan.requested_stems
    )
    return {name: np.full((n, 2), 0.2, dtype=np.float32) for name in names}


def test_prepare_stems_skips_routing_and_mastering(tmp_path):
    cfg = UpmixConfig(stems=["Vocals"], output_format="5.1")
    pipeline = StemUpmixPipeline(cfg)
    source = str(tmp_path / "in.wav")
    sf.write(source, _sine(SR), SR, subtype="FLOAT")
    messages: list[str] = []

    with patch(_EXEC_PLAN, side_effect=_fake_execute_plan):
        result = pipeline.prepare_stems(
            source, progress_callback=lambda m, f: messages.append(m)
        )
    pipeline.close()

    assert result.mode == "stem"
    assert result.stems == ["Vocals"]
    assert result.n_channels_out == 0
    assert result.output_path == ""
    assert result.measured_lkfs is None
    joined = " ".join(messages)
    assert "Routing" not in joined
    assert "Mastering" not in joined


def test_prepare_stems_writes_cache(tmp_path):
    cache_dir = tmp_path / "cache"
    cfg = UpmixConfig(stems=["Vocals"], output_format="5.1", stem_cache_dir=str(cache_dir))
    pipeline = StemUpmixPipeline(cfg)
    source = str(tmp_path / "in.wav")
    sf.write(source, _sine(SR), SR, subtype="FLOAT")

    with patch(_EXEC_PLAN, side_effect=_fake_execute_plan):
        pipeline.prepare_stems(source)
    pipeline.close()

    assert cache_dir.exists()
    assert any(os.scandir(cache_dir))


def test_prepare_opt_in_writes_private_terminals_without_public_summary_change(tmp_path):
    output_dir = tmp_path / "prepared"
    cfg = UpmixConfig(stems=["Vocals"], stem_output_dir=str(output_dir))
    pipeline = StemUpmixPipeline(cfg)
    source = str(tmp_path / "in.wav")
    sf.write(source, _sine(SR), SR, subtype="FLOAT")

    with patch(_EXEC_PLAN, side_effect=_fake_execute_plan):
        result = pipeline.prepare_stems(source, retain_private=True)
    pipeline.close()

    loaded, stored_sr = PlainStemStore(str(output_dir)).load()
    assert stored_sr == 44_100
    assert set(loaded) == {"Vocals", "_deux_inst"}
    assert result.stems == ["Vocals"]


def test_prepare_opt_in_bypasses_supplied_and_public_cached_stems(tmp_path):
    from upmixer.separation.stem_cache import StemCache
    from upmixer.separation.stem_identity import stem_cache_identity
    from upmixer.separation.stem_plan import resolve_separation_plan

    source = str(tmp_path / "in.wav")
    sf.write(source, _sine(SR), SR, subtype="FLOAT")
    input_dir = tmp_path / "input"
    cache_dir = tmp_path / "cache"
    cached = np.full((SR, 2), 0.7, dtype=np.float32)
    PlainStemStore(str(input_dir)).write({"Vocals": cached}, SR)
    cfg = UpmixConfig(
        stems=["Vocals"],
        stem_input_dir=str(input_dir),
        stem_cache_dir=str(cache_dir),
    )
    plan = resolve_separation_plan(["Vocals"], False)
    StemCache(str(cache_dir)).save(
        source,
        stem_cache_identity(plan, cfg, 44_100),
        SR,
        {"Vocals": cached},
        SR,
    )
    before_cache = {
        path.relative_to(cache_dir): path.read_bytes()
        for path in cache_dir.rglob("*")
        if path.is_file()
    }

    def prepare(output_dir, **config_overrides):
        run_cfg = replace(cfg, stem_output_dir=str(output_dir), **config_overrides)
        pipeline = StemUpmixPipeline(run_cfg)
        try:
            with patch(_EXEC_PLAN, side_effect=_fake_execute_plan):
                result = pipeline.prepare_stems(source, retain_private=True)
        finally:
            pipeline.close()
        loaded, _ = PlainStemStore(str(output_dir)).load()
        return result, loaded

    result, loaded = prepare(tmp_path / "from_input")
    assert result.stems == ["Vocals"]
    assert set(loaded) == {"Vocals", "_deux_inst"}

    result, loaded = prepare(tmp_path / "from_cache", stem_input_dir=None)
    assert result.stems == ["Vocals"]
    assert set(loaded) == {"Vocals", "_deux_inst"}
    assert {
        path.relative_to(cache_dir): path.read_bytes()
        for path in cache_dir.rglob("*")
        if path.is_file()
    } == before_cache


def test_prepare_prefers_supplied_stems_over_cache_or_inference(tmp_path):
    source = str(tmp_path / "in.wav")
    sf.write(source, _sine(SR), SR, subtype="FLOAT")
    input_dir = tmp_path / "prepared"
    PlainStemStore(str(input_dir)).write(
        {"Vocals": np.full((SR, 2), 0.2, dtype=np.float32)}, SR,
    )
    cfg = UpmixConfig(
        stems=["Vocals"],
        stem_input_dir=str(input_dir),
        stem_cache_dir=str(tmp_path / "cache"),
    )
    pipeline = StemUpmixPipeline(cfg)
    try:
        with patch(_EXEC_PLAN, side_effect=AssertionError("inference was not skipped")):
            result = pipeline.prepare_stems(source)
    finally:
        pipeline.close()

    assert result.stems == ["Vocals"]
    assert result.output_sample_rate == SR
    assert not (tmp_path / "cache").exists()


def test_prepare_and_render_keep_the_accepted_stem_sample_rate(tmp_path):
    source_sr = SR
    target_sr = 44_100
    source_frames = 480
    source = str(tmp_path / "in.wav")
    sf.write(source, _sine(source_frames), source_sr, subtype="FLOAT")
    output_dir = tmp_path / "prepared"
    cfg = UpmixConfig(
        stems=["Vocals"],
        output_format="5.1",
        output_sample_rate=target_sr,
        stem_output_dir=str(output_dir),
    )

    def fake_at_rate(get_separator, plan, sep_path, sep_sr, stage_callback=None,
                     cfg=None, resume_key=None):
        audio, input_sr = sf.read(sep_path, dtype="float32", always_2d=True)
        frames = round(len(audio) * sep_sr / input_sr)
        return {
            name: np.full((frames, 2), 0.2, dtype=np.float32)
            for name in plan.requested_stems
        }

    pipeline = StemUpmixPipeline(cfg)
    try:
        with patch(_EXEC_PLAN, side_effect=fake_at_rate):
            result = pipeline.prepare_stems(source)
    finally:
        pipeline.close()

    loaded, stored_sr = PlainStemStore(str(output_dir)).load()
    assert result.output_sample_rate == target_sr
    assert stored_sr == target_sr
    assert len(loaded["Vocals"]) == round(source_frames * target_sr / source_sr)

    render_cfg = UpmixConfig(
        stems=["Vocals"],
        output_format="5.1",
        output_sample_rate=target_sr,
        stem_input_dir=str(output_dir),
    )
    rendered, render_sr = render_prepared_stem_bed(render_cfg, source)
    assert render_sr == target_sr
    assert len(next(iter(rendered.values()))) == len(loaded["Vocals"])

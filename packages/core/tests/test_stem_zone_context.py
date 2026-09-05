"""Pipeline characterization for source zones, silence, and preview windows."""
from __future__ import annotations

from unittest.mock import patch

import numpy as np
import pytest
import soundfile as sf

from upmixer.config import UpmixConfig
from upmixer.separation.stem_pipeline import PreMasterAbort, StemUpmixPipeline
from upmixer.utils import itu_downmix_stereo

_EXEC_PLAN = "upmixer.separation.stem_pipeline_separate.execute_plan"
_EXEC_PLAN_ENGINE = "upmixer.separation.stem_pipeline_exec.execute_plan"
SR = 1_000


def _write_source(path, audio: np.ndarray, sample_rate: int = SR) -> None:
    sf.write(path, audio, sample_rate, subtype="FLOAT")


def _fake_execute_plan(seen: list, *, value: float = 0.0):
    def execute(_get_separator, plan, sep_path, sep_sr, stage_callback=None,
                cfg=None, resume_key=None):
        audio, rate = sf.read(sep_path, dtype="float32", always_2d=True)
        seen.append((str(sep_path), audio.copy(), rate, sep_sr))
        return {
            name: np.full((len(audio), 2), value, dtype=np.float32)
            for name in plan.requested_stems
        }

    return execute


def _pipeline(**kwargs) -> StemUpmixPipeline:
    return StemUpmixPipeline(
        UpmixConfig(stems=["Vocals"], normalize_output=False,
                    loudness_normalize=False, stem_source_anchor_strength=0.0,
                    **kwargs)
    )


def test_multichannel_pipeline_preserves_zone_order_and_passthrough(tmp_path):
    n = SR * 2
    source_audio = np.zeros((n, 6), dtype=np.float32)
    for channel, level in enumerate((0.01, 0.02, 0.30, 0.40, 0.05, 0.06)):
        source_audio[:, channel] = level
    source = tmp_path / "input.wav"
    _write_source(source, source_audio)
    seen: list = []
    captured: dict[str, np.ndarray] = {}

    def hook(channels, _sample_rate, _output_format):
        captured.update(channels)
        raise PreMasterAbort()

    pipeline = _pipeline(output_format="5.1", stem_silence_skip=False)
    try:
        with patch(_EXEC_PLAN, side_effect=_fake_execute_plan(seen)):
            with pytest.raises(PreMasterAbort):
                pipeline.process_file(str(source), str(tmp_path / "out.wav"),
                                      pre_master_hook=hook)
    finally:
        pipeline.close()

    np.testing.assert_allclose(
        [entry[1][:, 0].mean() for entry in seen], [0.01, 0.05], atol=1e-6
    )
    np.testing.assert_allclose(
        [entry[1][:, 1].mean() for entry in seen], [0.02, 0.06], atol=1e-6
    )
    np.testing.assert_array_equal(captured["C"], source_audio[:, 2])
    np.testing.assert_array_equal(captured["LFE"], source_audio[:, 3])


def test_mono_source_stays_mono_for_separator_and_duplicates_source_zone(tmp_path):
    mono = np.linspace(-0.2, 0.2, SR * 2, dtype=np.float32)[:, None]
    source = tmp_path / "mono.wav"
    _write_source(source, mono)
    seen: list = []
    pipeline = _pipeline(output_format="5.1", stem_silence_skip=False)
    try:
        with patch(_EXEC_PLAN, side_effect=_fake_execute_plan(seen)):
            result = pipeline._separate(str(source), None, lambda *_: None)
    finally:
        pipeline.close()

    assert result.input_fmt.name == "mono"
    assert result.audio_full.shape == mono.shape
    assert result.source_zones["front"].shape == (len(mono), 2)
    np.testing.assert_array_equal(result.source_zones["front"], np.repeat(mono, 2, axis=1))
    assert seen[0][0] == str(source)
    assert seen[0][1].shape == mono.shape


def test_multichannel_source_is_folded_before_stereo_separation(tmp_path):
    source_audio = np.column_stack([
        np.full(SR * 2, level, dtype=np.float32)
        for level in (0.01, 0.02, 0.03, 0.04, 0.05, 0.06)
    ])
    source = tmp_path / "surround.wav"
    _write_source(source, source_audio)
    seen: list = []
    pipeline = _pipeline(output_format="stereo", stem_silence_skip=False)
    try:
        with patch(_EXEC_PLAN, side_effect=_fake_execute_plan(seen)):
            result = pipeline._separate(str(source), None, lambda *_: None)
    finally:
        pipeline.close()

    labels = [label.value for label in result.input_fmt.channels]
    expected = np.column_stack(itu_downmix_stereo(
        {label: source_audio[:, index] for index, label in enumerate(labels)},
        surround_coeff=pipeline.config.surround_downmix_coeff,
        height_coeff=pipeline.config.height_downmix_coeff,
    )).astype(np.float32)
    assert result.input_fmt.name == "5.1"
    assert result.output_fmt.name == "stereo"
    assert result.passthrough == {}
    np.testing.assert_allclose(result.audio_full, expected)
    np.testing.assert_allclose(result.source_zones["front"], expected)
    assert len(seen) == 1
    assert seen[0][0] != str(source)
    np.testing.assert_allclose(seen[0][1], expected)


@pytest.mark.parametrize("kind, expected_calls", [
    ("silent", 0),
    ("long_gap", 2),
    ("quiet_tail", 1),
])
def test_pipeline_silence_skip_keeps_full_length(kind, expected_calls, tmp_path):
    n = SR * 14
    audio = np.zeros((n, 2), dtype=np.float32)
    if kind == "long_gap":
        audio[: SR * 5] = 0.2
        audio[SR * 9 :] = 0.2
    elif kind == "quiet_tail":
        audio[: SR * 8] = 0.2
        audio[SR * 8 :] = 1e-6
    source = tmp_path / f"{kind}.wav"
    _write_source(source, audio)
    seen: list = []
    pipeline = _pipeline(
        output_format="5.1", stem_silence_threshold_db=-90.0,
        stem_silence_min_duration_s=2.0, stem_silence_pad_ms=0.0,
        stem_silence_crossfade_ms=0.0, stem_silence_skip=True,
    )
    try:
        with patch(_EXEC_PLAN_ENGINE,
                   side_effect=_fake_execute_plan(seen, value=0.25)):
            result = pipeline._separate(str(source), None, lambda *_: None)
    finally:
        pipeline.close()

    assert result.n_samples == n
    assert result.all_stems["Vocals"].shape == (n, 2)
    assert len(seen) == expected_calls
    if kind == "silent":
        assert np.all(result.all_stems["Vocals"] == 0.0)
    elif kind == "long_gap":
        assert np.all(result.all_stems["Vocals"][SR * 5 + 50 : SR * 9 - 50] == 0.0)
    else:
        assert np.all(result.all_stems["Vocals"][SR * 8 + 50 :] == 0.0)


def test_preview_separation_uses_explicit_trim_coordinates(tmp_path):
    n = SR * 5
    values = np.arange(n, dtype=np.float32)[:, None] / n
    source_audio = np.repeat(values, 2, axis=1)
    source = tmp_path / "preview.wav"
    _write_source(source, source_audio)
    seen: list = []
    pipeline = _pipeline(
        output_format="5.1", preview=True, preview_duration_s=1.5,
        preview_start_s=2.0, stem_silence_skip=False,
    )
    try:
        with patch(_EXEC_PLAN, side_effect=_fake_execute_plan(seen)):
            result = pipeline._separate(str(source), None, lambda *_: None)
    finally:
        pipeline.close()

    expected = source_audio[SR * 2 : SR * 3 + SR // 2]
    assert result.n_samples == len(expected)
    np.testing.assert_allclose(result.audio_full, expected)
    np.testing.assert_allclose(result.source_zones["front"], expected)
    assert seen[0][1].shape == expected.shape
    np.testing.assert_allclose(seen[0][1], expected)

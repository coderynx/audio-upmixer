"""Integration characterization for stem-plan execution order and closure."""
from __future__ import annotations

from dataclasses import dataclass
from pathlib import Path

import numpy as np
import pytest
import soundfile as sf

from upmixer.config import UpmixConfig
from upmixer.separation.stem_pipeline_exec import cacheable_plan_stems, execute_plan
from upmixer.separation.stem_plan import (
    DEFAULT_STEMS,
    DRUM_SUB_STEMS,
    MODEL_CROWD,
    MODEL_DEUX,
    MODEL_DRUMS,
    MODEL_KARAOKE,
    MODEL_PRIMARY,
    PRIMARY_INSTRUMENTAL_STEMS,
    resolve_separation_plan,
)

SR = 48_000
ORIGINAL_MARKER = -1.0
CROWD_OTHER_MARKER = 10.0
VOCALS_MARKER = 20.0
DEUX_INSTRUMENTAL_MARKER = 21.0
PRIMARY_DRUMS_MARKER = 31.0

_SOURCE_BY_MARKER = {
    ORIGINAL_MARKER: "original",
    CROWD_OTHER_MARKER: "_crowd_other",
    VOCALS_MARKER: "Vocals",
    DEUX_INSTRUMENTAL_MARKER: "_deux_inst",
    PRIMARY_DRUMS_MARKER: "Drums",
}


@dataclass
class _StageCall:
    model: str
    input_source: str | None
    input_audio: np.ndarray


class _RecordingSeparator:
    def __init__(self, model: str, root: Path, calls: list[_StageCall]) -> None:
        self.model = model
        self.root = root / model.replace("/", "_")
        self.root.mkdir(parents=True, exist_ok=True)
        self.calls = calls
        self.outputs: list[dict[str, np.ndarray]] = []
        self._last_parent: np.ndarray | None = None
        self._write_index = 0

    def separate_to_file(
        self,
        audio_path: str,
        keep_on_disk: frozenset[str],
        _stem_overrides=None,
        wanted: frozenset[str] | None = None,
        retain_parent: bool = False,
    ) -> tuple[dict[str, np.ndarray], dict[str, str]]:
        input_audio, _ = sf.read(audio_path, dtype="float32", always_2d=True)
        marker = float(input_audio[0, 0])
        self.calls.append(
            _StageCall(self.model, _SOURCE_BY_MARKER.get(marker), input_audio.copy())
        )
        if retain_parent:
            self._last_parent = input_audio.copy()

        output_markers = {
            MODEL_CROWD: {"Crowd": 11.0, "_crowd_other": CROWD_OTHER_MARKER},
            MODEL_DEUX: {
                "Vocals": VOCALS_MARKER,
                "_deux_inst": DEUX_INSTRUMENTAL_MARKER,
            },
            MODEL_PRIMARY: {
                name: PRIMARY_DRUMS_MARKER if name == "Drums" else 32.0 + index
                for index, name in enumerate(sorted(PRIMARY_INSTRUMENTAL_STEMS))
            },
            MODEL_DRUMS: {
                name: 40.0 + index
                for index, name in enumerate(sorted(DRUM_SUB_STEMS))
            },
            MODEL_KARAOKE: {"Lead Vocals": 50.0, "Backing Vocals": 51.0},
        }[self.model]
        self.outputs.append(
            {
                name: np.full((32, 2), value, dtype=np.float32)
                for name, value in output_markers.items()
            }
        )

        loaded: dict[str, np.ndarray] = {}
        on_disk: dict[str, str] = {}
        selected = set(output_markers) if wanted is None else set(wanted)
        for name, audio in self.outputs[-1].items():
            if name not in selected:
                continue
            if name in keep_on_disk:
                path = self.root / f"{self._write_index}-{name}.wav"
                self._write_index += 1
                sf.write(path, audio, SR, subtype="FLOAT")
                on_disk[name] = str(path)
            else:
                loaded[name] = audio
        return loaded, on_disk

    def take_last_parent(self) -> np.ndarray:
        assert self._last_parent is not None
        parent = self._last_parent
        self._last_parent = None
        return parent


class _RecordingFactory:
    def __init__(self, root: Path) -> None:
        self.root = root
        self.calls: list[_StageCall] = []
        self.separators: dict[str, _RecordingSeparator] = {}

    def __call__(self, model: str, _sample_rate: int) -> _RecordingSeparator:
        return self.separators.setdefault(
            model, _RecordingSeparator(model, self.root, self.calls)
        )


def _source(tmp_path: Path) -> str:
    path = tmp_path / "source.wav"
    audio = np.full((32, 2), ORIGINAL_MARKER, dtype=np.float32)
    sf.write(path, audio, SR, subtype="FLOAT")
    return str(path)


@pytest.mark.parametrize(
    ("requested", "expected_models", "expected_sources", "expected_final"),
    [
        (
            DEFAULT_STEMS,
            (MODEL_DEUX, MODEL_PRIMARY),
            ("original", "_deux_inst"),
            frozenset(DEFAULT_STEMS),
        ),
        (
            ["Vocals"],
            (MODEL_DEUX,),
            ("original",),
            frozenset({"Vocals"}),
        ),
        (
            ["Bass"],
            (MODEL_DEUX, MODEL_PRIMARY),
            ("original", "_deux_inst"),
            frozenset({"Bass"}),
        ),
        (
            ["Kick"],
            (MODEL_DEUX, MODEL_PRIMARY, MODEL_DRUMS),
            ("original", "_deux_inst", "Drums"),
            frozenset({"Kick"}),
        ),
        (
            ["Drums", "Kick"],
            (MODEL_DEUX, MODEL_PRIMARY, MODEL_DRUMS),
            ("original", "_deux_inst", "Drums"),
            frozenset({"Kick"}),
        ),
        (
            ["Lead Vocals"],
            (MODEL_DEUX, MODEL_KARAOKE),
            ("original", "Vocals"),
            frozenset({"Lead Vocals"}),
        ),
        (
            ["Backing Vocals"],
            (MODEL_DEUX, MODEL_KARAOKE),
            ("original", "Vocals"),
            frozenset({"Backing Vocals"}),
        ),
        (
            ["Vocals", "Backing Vocals"],
            (MODEL_DEUX, MODEL_KARAOKE),
            ("original", "Vocals"),
            frozenset({"Backing Vocals"}),
        ),
        (
            ["Lead Vocals", "Backing Vocals"],
            (MODEL_DEUX, MODEL_KARAOKE),
            ("original", "Vocals"),
            frozenset({"Lead Vocals", "Backing Vocals"}),
        ),
        (
            ["Crowd"],
            (MODEL_CROWD,),
            ("original",),
            frozenset({"Crowd"}),
        ),
        (
            ["Crowd", "Kick", "Backing Vocals"],
            (MODEL_CROWD, MODEL_DEUX, MODEL_PRIMARY, MODEL_DRUMS, MODEL_KARAOKE),
            ("original", "_crowd_other", "_deux_inst", "Drums", "Vocals"),
            frozenset({"Crowd", "Kick", "Backing Vocals"}),
        ),
    ],
)
def test_execution_records_order_and_replaces_ancestor_outputs(
    tmp_path: Path,
    requested: list[str],
    expected_models: tuple[str, ...],
    expected_sources: tuple[str, ...],
    expected_final: frozenset[str],
):
    factory = _RecordingFactory(tmp_path / "separators")
    plan = resolve_separation_plan(requested)

    stems = execute_plan(
        factory,
        plan,
        _source(tmp_path),
        SR,
        cfg=UpmixConfig(stem_primary_remask=False, stem_drum_remask=False),
    )

    assert [(call.model, call.input_source) for call in factory.calls] == list(
        zip(expected_models, expected_sources)
    )
    assert plan.requested_stems == expected_final
    assert set(stems) == cacheable_plan_stems(plan)
    assert {name for name in stems if name in plan.requested_stems} == expected_final

    if MODEL_DEUX in expected_models:
        deux_heads = factory.separators[MODEL_DEUX].outputs[0]
        assert deux_heads["Vocals"][0, 0] == VOCALS_MARKER
        assert deux_heads["_deux_inst"][0, 0] == DEUX_INSTRUMENTAL_MARKER
        assert deux_heads["Vocals"][0, 0] != deux_heads["_deux_inst"][0, 0]

    assert [
        float(call.input_audio[0, 0])
        for call in factory.calls
        if call.input_source == "_deux_inst"
    ] == [DEUX_INSTRUMENTAL_MARKER] * expected_sources.count("_deux_inst")
    assert [
        float(call.input_audio[0, 0])
        for call in factory.calls
        if call.input_source == "Vocals"
    ] == [VOCALS_MARKER] * expected_sources.count("Vocals")


@pytest.mark.parametrize(
    ("requested", "expected_private"),
    [
        (["Vocals"], {"_deux_inst"}),
        (["Crowd"], {"_crowd_other"}),
        (["Crowd", "Bass"], set()),
        (["Crowd", "Vocals"], {"_deux_inst"}),
    ],
)
def test_opt_in_returns_only_unconsumed_private_terminals(
    tmp_path: Path, requested: list[str], expected_private: set[str]
):
    factory = _RecordingFactory(tmp_path / "separators")
    plan = resolve_separation_plan(requested)

    stems = execute_plan(
        factory,
        plan,
        _source(tmp_path),
        SR,
        cfg=UpmixConfig(stem_primary_remask=False, stem_drum_remask=False),
        retain_private=True,
    )

    assert {name for name in stems if name.startswith("_")} == expected_private
    assert set(stems) == cacheable_plan_stems(plan, retain_private=True)


def test_remask_cleanup_passes_corrected_drums_to_drumsep(tmp_path, monkeypatch):
    factory = _RecordingFactory(tmp_path / "separators")
    cleanup_calls = []

    def fake_cleanup(parent, vocals, instrumental, sample_rate):
        cleanup_calls.append((parent.copy(), sample_rate))
        return vocals, instrumental

    def fake_remask(parent, children, _sample_rate):
        corrected = {name: audio.copy() for name, audio in children.items()}
        if "Drums" in corrected:
            corrected["Drums"] += 100.0
        return corrected

    monkeypatch.setattr(
        "upmixer.separation.stem_cleanup.apply_stem_cleanup", fake_cleanup
    )
    monkeypatch.setattr(
        "upmixer.separation.stem_workspace.share_parent_residual", fake_remask
    )

    plan = resolve_separation_plan(["Kick"])
    stems = execute_plan(
        factory,
        plan,
        _source(tmp_path),
        SR,
        cfg=UpmixConfig(stem_bleed_reduction=True),
    )

    drumsep_call = next(call for call in factory.calls if call.model == MODEL_DRUMS)
    assert np.all(drumsep_call.input_audio == PRIMARY_DRUMS_MARKER + 100.0)
    assert len(cleanup_calls) == 1
    assert {"Kick"} <= set(stems)

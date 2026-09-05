"""Focused regressions for runtime provenance history."""

from __future__ import annotations

from types import SimpleNamespace
from unittest.mock import patch

from upmixer.config import UpmixConfig
from upmixer.separation.separator import SeparationSettings, StemSeparator
from upmixer.separation.stem_pipeline import StemUpmixPipeline


def test_oom_history_records_effective_engine_settings():
    separator = StemSeparator(
        model="model.ckpt",
        batch_size=4,
        segment_size=None,
        chunk_duration_s=120.0,
    )
    separator._backend = "mps"

    class FakeDevice:
        type = "mps"

        def __str__(self):
            return "mps"

    class FakeEngine:
        _arch = "bs_roformer"
        _config = SimpleNamespace(default_segment_size=3072)
        calls = 0

        def _model_device(self):
            return FakeDevice()

        def _resolved_segment_size(self):
            return 3072

        def separate(self, _audio_path):
            self.calls += 1
            if self.calls == 1:
                raise MemoryError("out of memory")
            return []

    engine = FakeEngine()

    def get_engine():
        separator._engine = engine
        return engine

    try:
        with (
            patch.object(separator, "_get_separator", side_effect=get_engine),
            patch.object(
                separator,
                "_registry_details",
                return_value=("bs_roformer", None, None, None),
            ),
        ):
            assert separator._separate_paths("input.wav") == []

        settings = separator.run_settings
        assert settings is not None
        assert settings.batch_size == 1
        assert settings.segment_size == 3072
        assert settings.oom_fallback_attempts == (
            {
                "batch_size": 1,
                "segment_size": 3072,
                "chunk_duration_s": 120.0,
            },
        )
    finally:
        separator.close()


def test_pipeline_keeps_ordered_snapshots_for_reused_separator():
    class FakeSeparator:
        def __init__(self, model, **_):
            self.model = model
            self.backend = "cpu"
            self.run_settings = None
            self._settings_observer = None
            self.calls = 0

        def close(self):
            pass

        def separate_to_file(self, *_args, **_kwargs):
            self.calls += 1
            fallback = (
                ({"batch_size": 2, "segment_size": 128, "chunk_duration_s": 120.0},)
                if self.calls == 1
                else ()
            )
            self.run_settings = SeparationSettings(
                model=self.model,
                sample_rate=48000,
                batch_size=1,
                segment_size=64,
                chunk_duration_s=120.0,
                overlap=2,
                tta=False,
                pitch_shift=None,
                backend="cpu",
                oom_fallback_attempts=fallback,
                oom_fallback_count=len(fallback),
            )
            if self._settings_observer is not None:
                self._settings_observer(self.run_settings)
            return {}, {}

    def run_separate(get_separator, *_args):
        separator = get_separator("model.ckpt", 48000)
        separator.separate_to_file("first.wav", frozenset())
        separator = get_separator("model.ckpt", 48000)
        separator.separate_to_file("second.wav", frozenset())
        return object()

    with (
        patch("upmixer.separation.stem_pipeline.StemSeparator", FakeSeparator),
        patch("upmixer.separation.stem_pipeline.separate", side_effect=run_separate),
    ):
        pipeline = StemUpmixPipeline(UpmixConfig())
        try:
            pipeline._separate("source.wav", None, lambda *_: None)
            snapshots = pipeline.last_separation_settings
        finally:
            pipeline.close()

    assert len(snapshots) == 2
    assert [snapshot.oom_fallback_count for snapshot in snapshots] == [1, 0]
    assert snapshots[0].oom_fallback_attempts[0]["batch_size"] == 2

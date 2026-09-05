"""Runtime provenance captured by completed stem-separator runs."""

from __future__ import annotations

import hashlib
from pathlib import Path
from unittest.mock import patch

import pytest

from upmixer.separation import separator as separator_module
from upmixer.separation.separator import StemSeparator


def _sha256(path: Path) -> str:
    return hashlib.sha256(path.read_bytes()).hexdigest()


def _bundled_config() -> Path:
    return (
        Path(separator_module.__file__).parent
        / "inference"
        / "configs"
        / "BS-Roformer-SW.yaml"
    )


def test_engine_snapshot_records_hashes_and_runtime_policy(tmp_path):
    model_dir = tmp_path / "models"
    checkpoint = model_dir / "BS-Roformer-SW.ckpt"
    checkpoint.parent.mkdir()
    checkpoint.write_bytes(b"test checkpoint")

    class FakeDevice:
        type = "cpu"

        def __str__(self):
            return "cpu"

    class FakeEngine:
        _arch = "bs_roformer"

        def separate(self, _audio_path):
            return []

        def _resolved_segment_size(self):
            return 123

        def _model_device(self):
            return FakeDevice()

    separator = StemSeparator(
        model=checkpoint.name,
        model_dir=str(model_dir),
        batch_size=2,
        chunk_duration_s=30.0,
    )
    separator._engine = FakeEngine()
    try:
        with patch.object(separator, "_get_separator", return_value=separator._engine):
            assert separator._separate_paths("input.wav") == []

        settings = separator.run_settings
        assert settings is not None
        assert settings.checkpoint_sha256 == _sha256(checkpoint)
        assert settings.model_config_sha256 == _sha256(_bundled_config())
        assert settings.runtime_precision == "float32"
        assert settings.normalization_policy == (
            "peak-downscale-to-0.9; inverse-output-scale"
        )
        assert settings.oom_fallback_attempts == ()
        assert settings.oom_fallback_count == 0
    finally:
        separator.close()


def test_hashes_wait_for_success_and_are_cached_per_separator(tmp_path):
    model_dir = tmp_path / "models"
    checkpoint = model_dir / "BS-Roformer-SW.ckpt"
    checkpoint.parent.mkdir()
    checkpoint.write_bytes(b"test checkpoint")

    class FakeEngine:
        calls = 0

        def separate(self, _audio_path):
            self.calls += 1
            if self.calls == 1:
                raise MemoryError("out of memory")
            return []

        _arch = "bs_roformer"

        def _resolved_segment_size(self):
            return 64

    separator = StemSeparator(
        model=checkpoint.name,
        model_dir=str(model_dir),
        batch_size=1,
        segment_size=64,
        chunk_duration_s=60.0,
    )
    separator._backend = "cpu"
    engine = FakeEngine()

    def get_engine():
        separator._engine = engine
        return engine

    hashed: list[str] = []

    def record_hash(path: Path) -> str:
        hashed.append(path.name)
        return f"digest-{len(hashed)}"

    try:
        with patch.object(separator, "_get_separator", side_effect=get_engine):
            with patch.object(
                separator_module, "_sha256_file", side_effect=record_hash
            ):
                with pytest.raises(MemoryError):
                    separator._separate_paths("input.wav")
                assert hashed == []

                assert separator._separate_paths("input.wav") == []
                assert hashed == [checkpoint.name, "BS-Roformer-SW.yaml"]
                assert separator.run_settings is not None
                assert separator.run_settings.checkpoint_sha256 == "digest-1"
                assert separator.run_settings.model_config_sha256 == "digest-2"

                assert separator._separate_paths("input.wav") == []
                assert hashed == [checkpoint.name, "BS-Roformer-SW.yaml"]
    finally:
        separator.close()


def test_retry_snapshot_keeps_failed_memory_settings_and_resets_next_run():
    separator = StemSeparator(
        model="model.ckpt",
        batch_size=1,
        segment_size=128,
        chunk_duration_s=240.0,
    )
    separator._backend = "cpu"

    class FakeEngine:
        calls = 0

        def separate(self, _audio_path):
            self.calls += 1
            if self.calls <= 2:
                raise MemoryError("out of memory")
            return []

        _arch = "unknown"

        def _resolved_segment_size(self):
            return separator._segment_size

    engine = FakeEngine()

    def get_engine():
        separator._engine = engine
        return engine

    try:
        with patch.object(separator, "_get_separator", side_effect=get_engine):
            assert separator._separate_paths("input.wav") == []
            settings = separator.run_settings
            assert settings is not None
            assert settings.batch_size == 1
            assert settings.segment_size == 64
            assert settings.chunk_duration_s == 120.0
            assert settings.oom_fallback_attempts == (
                {
                    "batch_size": 1,
                    "segment_size": 128,
                    "chunk_duration_s": 240.0,
                },
                {
                    "batch_size": 1,
                    "segment_size": 64,
                    "chunk_duration_s": 240.0,
                },
            )
            assert settings.oom_fallback_count == 2

            assert separator._separate_paths("input.wav") == []
            settings = separator.run_settings
            assert settings is not None
            assert settings.oom_fallback_attempts == ()
            assert settings.oom_fallback_count == 0
    finally:
        separator.close()


def test_scnet_worker_snapshot_records_observable_provenance(monkeypatch, tmp_path):
    model = "model_scnet_ep_36_sdr_10.0891.ckpt"
    model_dir = tmp_path / "models"
    checkpoint = model_dir / model
    checkpoint.parent.mkdir()
    checkpoint.write_bytes(b"test checkpoint")

    class FakeWorker:
        def __init__(self, _model, _model_dir):
            pass

        def separate(self, _audio_path, _output_dir, **_kwargs):
            return []

        def close(self):
            pass

    monkeypatch.setattr(separator_module, "_detect_backend", lambda: "mps")
    monkeypatch.setattr(
        separator_module, "_mlx_scnet_available", lambda: True
    )
    monkeypatch.setattr(separator_module, "SCNetWorker", FakeWorker)
    separator = StemSeparator(model=model, model_dir=str(model_dir))
    try:
        assert separator._separate_paths("input.wav") == []
        settings = separator.run_settings
        assert settings is not None
        assert settings.backend == "mlx"
        assert settings.model_arch == "scnet"
        assert settings.checkpoint_sha256 == _sha256(checkpoint)
        assert settings.model_config_sha256 is not None
        assert settings.runtime_precision == "float32"
        assert settings.normalization_policy == (
            "peak-downscale-to-0.9; inverse-output-scale"
        )
    finally:
        separator.close()

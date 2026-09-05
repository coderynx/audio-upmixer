"""Tests for upmixer.separation.stem_store — PlainStemStore."""
from __future__ import annotations

import numpy as np
import pytest
import soundfile as sf

from upmixer.separation.stem_store import PlainStemStore


def _make_stems() -> dict[str, np.ndarray]:
    t = np.linspace(0, 1, 4096, endpoint=False)
    sig = 0.3 * np.sin(2 * np.pi * 440 * t).astype(np.float32)
    return {
        "Vocals": np.column_stack([sig, sig]),
        "Drums@front": np.column_stack([sig, -sig]),
    }


def test_load_returns_none_when_directory_is_empty(tmp_path):
    store = PlainStemStore(str(tmp_path / "missing"))
    assert store.load() is None


def test_write_then_load_round_trips_stems(tmp_path):
    stems = _make_stems()
    store = PlainStemStore(str(tmp_path))
    store.write(stems, 44100, source_size=12345)

    loaded, sample_rate = store.load()
    assert sample_rate == 44100
    assert set(loaded.keys()) == set(stems.keys())
    for key, audio in stems.items():
        np.testing.assert_array_equal(loaded[key], audio)


def test_failed_write_leaves_no_temporary_file_or_readable_partial_store(tmp_path, monkeypatch):
    store = PlainStemStore(str(tmp_path))
    real_write = sf.write
    writes = 0

    def fail_second_write(*args, **kwargs):
        nonlocal writes
        writes += 1
        if writes == 2:
            raise OSError("synthetic partial store write")
        return real_write(*args, **kwargs)

    monkeypatch.setattr(sf, "write", fail_second_write)
    with pytest.raises(OSError, match="synthetic partial"):
        store.write(_make_stems(), 44100)

    assert not list(tmp_path.glob(".*.tmp.wav"))
    assert store.load() is None


def test_load_writes_no_hash_subdirectory(tmp_path):
    store = PlainStemStore(str(tmp_path))
    store.write(_make_stems(), 44100)
    entries = {p.name for p in tmp_path.iterdir()}
    assert "stems.json" in entries
    assert "Vocals.wav" in entries
    assert "Drums__front.wav" in entries
    assert len(entries) == 3


def test_above_unity_stem_is_not_clipped(tmp_path):
    # Stems carry the source's true level, so one separated from a clipped
    # master exceeds 1.0 — a fixed-point subtype would hard-clip it here.
    peaks = np.array([1.4, -1.25], dtype=np.float32)
    stems = {"Vocals": np.column_stack([peaks, peaks])}
    store = PlainStemStore(str(tmp_path))
    store.write(stems, 44100)

    loaded, _ = store.load()
    np.testing.assert_allclose(loaded["Vocals"], stems["Vocals"], atol=1e-6)


def test_write_replaces_previous_contents(tmp_path):
    store = PlainStemStore(str(tmp_path))
    store.write({"Vocals": np.zeros((100, 2), dtype=np.float32)}, 44100)
    store.write(_make_stems(), 44100)

    loaded, _ = store.load()
    assert set(loaded.keys()) == {"Vocals", "Drums@front"}


def test_write_removes_old_manifest_wavs_but_keeps_unmanaged_files(tmp_path):
    store = PlainStemStore(str(tmp_path))
    store.write(
        {
            "Vocals": np.zeros((100, 2), dtype=np.float32),
            "Bass": np.ones((100, 2), dtype=np.float32),
        },
        44100,
    )
    unmanaged = tmp_path / "unmanaged.wav"
    unmanaged.write_bytes(b"leave me alone")

    store.write(_make_stems(), 44100)

    assert not (tmp_path / "Bass.wav").exists()
    assert (tmp_path / "Vocals.wav").exists()
    assert (tmp_path / "Drums__front.wav").exists()
    assert unmanaged.read_bytes() == b"leave me alone"
    loaded, _ = store.load()
    assert set(loaded) == {"Vocals", "Drums@front"}

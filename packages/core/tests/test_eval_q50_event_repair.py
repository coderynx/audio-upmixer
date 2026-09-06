"""Deterministic Q50 aggregate-Drums ownership probes."""

import numpy as np
import pytest

from upmixer.eval import repair_aggregate_drums

_PRIMARY = ("Bass", "Drums", "Guitar", "Piano", "Other")


def _pulse(
    frames: int,
    sample_rate: int,
    start: int,
    frequency: float = 440.0,
    gain: float = 1.0,
    decay: float = 24.0,
) -> np.ndarray:
    length = min(1800, frames - start)
    time = np.arange(length, dtype=np.float64) / sample_rate
    signal = np.sin(2 * np.pi * frequency * time) * np.exp(-decay * time) * gain
    audio = np.zeros((frames, 2), dtype=np.float32)
    audio[start : start + length, 0] = signal
    audio[start : start + length, 1] = signal * 0.7
    return audio


def _siblings(drums: np.ndarray, guitar: np.ndarray) -> dict[str, np.ndarray]:
    result = {name: np.zeros_like(drums) for name in _PRIMARY}
    result["Drums"] = drums
    result["Guitar"] = guitar
    return result


def test_q50_transfers_current_donor_and_preserves_parent_sum():
    sample_rate = 8_000
    frames = 30_000
    drums = _pulse(frames, sample_rate, 4_000) + _pulse(
        frames, sample_rate, 20_000, frequency=880.0
    )
    guitar = _pulse(frames, sample_rate, 12_000, gain=0.6)
    stems = _siblings(drums, guitar)
    parent = sum(stems.values(), np.zeros_like(drums))

    result = repair_aggregate_drums(parent, stems, sample_rate)

    assert len(result.transfers) == 1
    assert result.transfers[0].donor == "Guitar"
    assert result.transfers[0].gain == pytest.approx(0.25)
    assert result.stems["Drums"].dtype == np.float32
    assert result.stems["Guitar"].dtype == np.float32
    assert np.isfinite(result.stems["Drums"]).all()
    np.testing.assert_allclose(
        sum(result.stems.values(), np.zeros_like(parent)), parent, atol=1e-6, rtol=0
    )
    assert np.max(np.abs(result.stems["Drums"] - drums)) > 0
    assert np.max(np.abs(result.stems["Guitar"] - guitar)) > 0


def test_q50_is_exact_noop_without_repetition_or_on_stereo_mismatch():
    sample_rate = 8_000
    frames = 20_000
    drums = _pulse(frames, sample_rate, 4_000)
    guitar = _pulse(frames, sample_rate, 12_000, gain=0.6)
    stems = _siblings(drums, guitar)
    parent = sum(stems.values(), np.zeros_like(drums))

    result = repair_aggregate_drums(parent, stems, sample_rate)
    assert result.transfers == ()
    for name in _PRIMARY:
        np.testing.assert_array_equal(result.stems[name], stems[name])

    mono = {name: audio[:, :1].copy() for name, audio in stems.items()}
    mono_result = repair_aggregate_drums(parent[:, :1], mono, sample_rate)
    assert mono_result.transfers == ()
    for name in _PRIMARY:
        np.testing.assert_array_equal(mono_result.stems[name], mono[name])


@pytest.mark.parametrize("frequency", [330.0, 450.0, 880.0])
def test_q50_abstains_on_wrong_neighbor_or_changed_articulation(frequency: float):
    sample_rate = 8_000
    frames = 30_000
    drums = _pulse(frames, sample_rate, 4_000) + _pulse(frames, sample_rate, 20_000)
    guitar = _pulse(frames, sample_rate, 12_000, frequency=frequency, gain=0.6)
    stems = _siblings(drums, guitar)
    parent = sum(stems.values(), np.zeros_like(drums))

    result = repair_aggregate_drums(parent, stems, sample_rate)

    assert result.transfers == ()
    for name in _PRIMARY:
        np.testing.assert_array_equal(result.stems[name], stems[name])


def test_q50_abstains_on_ambiguous_repeated_references():
    sample_rate = 8_000
    frames = 40_000
    drums = _pulse(frames, sample_rate, 4_096) + _pulse(frames, sample_rate, 12_288)
    guitar = _pulse(frames, sample_rate, 20_480, gain=0.6)
    stems = _siblings(drums, guitar)
    parent = sum(stems.values(), np.zeros_like(drums))

    result = repair_aggregate_drums(parent, stems, sample_rate)

    assert result.transfers == ()
    for name in _PRIMARY:
        np.testing.assert_array_equal(result.stems[name], stems[name])


def test_q50_abstains_on_stereo_mismatch():
    sample_rate = 8_000
    frames = 30_000
    drums = _pulse(frames, sample_rate, 4_000) + _pulse(
        frames, sample_rate, 20_000, frequency=880.0
    )
    guitar = _pulse(frames, sample_rate, 12_000, gain=0.6)[:, ::-1].copy()
    stems = _siblings(drums, guitar)
    parent = sum(stems.values(), np.zeros_like(drums))

    result = repair_aggregate_drums(parent, stems, sample_rate)

    assert result.transfers == ()
    for name in _PRIMARY:
        np.testing.assert_array_equal(result.stems[name], stems[name])


def test_q50_abstains_on_changed_decay():
    sample_rate = 8_000
    frames = 30_000
    drums = _pulse(frames, sample_rate, 4_000) + _pulse(
        frames, sample_rate, 20_000, frequency=880.0
    )
    guitar = _pulse(frames, sample_rate, 12_000, gain=0.6, decay=10.0)
    stems = _siblings(drums, guitar)
    parent = sum(stems.values(), np.zeros_like(drums))

    result = repair_aggregate_drums(parent, stems, sample_rate)

    assert result.transfers == ()
    for name in _PRIMARY:
        np.testing.assert_array_equal(result.stems[name], stems[name])


def test_q50_validates_shapes_and_finite_values():
    parent = np.zeros((100, 2), dtype=np.float32)
    stems = _siblings(parent.copy(), parent.copy())
    with pytest.raises(ValueError, match="shape"):
        repair_aggregate_drums(parent, {**stems, "Guitar": parent[:-1]}, 8_000)
    bad = {**stems, "Guitar": parent.copy()}
    bad["Guitar"][0, 0] = np.nan
    with pytest.raises(ValueError, match="finite"):
        repair_aggregate_drums(parent, bad, 8_000)

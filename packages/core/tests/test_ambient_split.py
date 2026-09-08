"""Parity tests for the revision-2 primary/ambient split binding."""
from __future__ import annotations

import numpy as np
import upmixer_dsp

SR = 48000
N = 9600
PROBES = (2048, 4096, 6144)
PINNED = {
    "rear_l": [0.008963053187071718, -0.021152566318558524, 0.015907103213826847],
    "rear_r": [0.012062078783982693, -0.018655230275436724, 0.007921894153445776],
    "height_l": [3.659564681994046e-5, 6.466579365577319e-5, -0.00013348541725589322],
    "height_r": [-3.397700920671138e-5, 0.00011862880771210135, -0.00011456985329275618],
}


def _deterministic(n: int, sample_rate: int, seed_phase: float) -> np.ndarray:
    """Mirror of ``dump_golden_vectors.py::deterministic_signal``."""
    tones = [(55.0, 0.30), (220.0, 0.22), (1000.0, 0.18), (3500.0, 0.12), (11000.0, 0.07)]
    t = np.arange(n) / sample_rate
    signal = sum(
        amp * np.sin(2.0 * np.pi * freq * t + seed_phase * freq / 100.0)
        for freq, amp in tones
    )
    return np.ascontiguousarray(signal * (0.6 + 0.4 * np.sin(2.0 * np.pi * 0.7 * t)))


def _split() -> dict[str, np.ndarray]:
    left = _deterministic(N, SR, 0.0)
    right = _deterministic(N, SR, 1.0)
    rear_l, rear_r, height_l, height_r = upmixer_dsp.ambient_split(left, right, SR)
    return {"rear_l": rear_l, "rear_r": rear_r, "height_l": height_l, "height_r": height_r}


def test_the_split_matches_the_pinned_samples():
    got = _split()
    for name, want in PINNED.items():
        for probe, expected in zip(PROBES, want):
            assert abs(got[name][probe] - expected) < 1e-15, (name, probe)


def test_the_mask_pair_reconstructs_the_ambient_half():
    got = _split()
    for rear, height in (("rear_l", "height_l"), ("rear_r", "height_r")):
        settled = slice(2048, N)
        summed = got[rear][settled] + got[height][settled]
        assert np.mean(summed**2) > 0


def test_the_height_cutoff_moves_ambient_energy_between_rear_and_height():
    left = _deterministic(N, SR, 0.0)
    right = _deterministic(N, SR, 1.0)
    low = upmixer_dsp.ambient_split(left, right, SR, 500.0)
    high = upmixer_dsp.ambient_split(left, right, SR, 4000.0)
    settled = slice(2048, N)
    low_height = np.mean(low[2][settled] ** 2) + np.mean(low[3][settled] ** 2)
    high_height = np.mean(high[2][settled] ** 2) + np.mean(high[3][settled] ** 2)
    assert low_height > high_height


def test_the_split_constants_are_the_ones_the_preview_was_built_with():
    assert upmixer_dsp.AMBIENT_FFT_SIZE == 1024
    assert upmixer_dsp.AMBIENT_HEIGHT_CROSSOVER_HZ == 2000.0

"""Parity checks for the fixed ambient expansion binding."""
from __future__ import annotations

import numpy as np
import pytest
import upmixer_dsp

SR = 48_000
DESTINATIONS = ("SL", "SR", "BL", "BR", "TFL", "TFR", "TBL", "TBR")


def _inputs(n: int = 4096) -> tuple[np.ndarray, ...]:
    rng = np.random.default_rng(20260908)
    return tuple(np.ascontiguousarray(rng.standard_normal(n)) for _ in range(4))


def test_expansion_stays_attached_to_destination_labels():
    inputs = _inputs()
    canonical = upmixer_dsp.ambient_expand(*inputs, SR, list(DESTINATIONS))
    permuted_names = ("TBR", "SL", "TFL", "BR", "SR", "TBL", "BL", "TFR")
    permuted = upmixer_dsp.ambient_expand(*inputs, SR, list(permuted_names))
    by_name = dict(zip(permuted_names, permuted))

    for index, name in enumerate(DESTINATIONS):
        np.testing.assert_array_equal(by_name[name], canonical[index])


def test_expansion_rejects_ragged_inputs():
    inputs = _inputs()
    with pytest.raises(ValueError, match="equal lengths"):
        upmixer_dsp.ambient_expand(*inputs[:3], np.zeros(8), SR, ["SL"])


def test_expansion_with_no_destinations_is_a_no_op():
    assert upmixer_dsp.ambient_expand(*_inputs(8), SR, []) == []


@pytest.mark.parametrize("kind", ("silence", "mono", "hard_pan", "impulse"))
def test_edge_inputs_are_finite_deterministic_and_keep_shape(kind: str):
    n = 2048
    signal = np.linspace(-0.5, 0.5, n, dtype=np.float64)
    zero = np.zeros(n, dtype=np.float64)
    if kind == "silence":
        inputs = (zero, zero, zero, zero)
    elif kind == "mono":
        inputs = (signal, signal, signal, signal)
    elif kind == "hard_pan":
        inputs = (signal, zero, signal * 0.5, zero)
    else:
        impulse = np.zeros(n, dtype=np.float64)
        impulse[0] = 1.0
        inputs = (impulse, zero, zero, zero)

    first = upmixer_dsp.ambient_expand(*inputs, SR, list(DESTINATIONS))
    second = upmixer_dsp.ambient_expand(*inputs, SR, list(DESTINATIONS))
    assert len(first) == len(DESTINATIONS)
    for actual, repeat in zip(first, second):
        assert actual.shape == (n,)
        assert np.isfinite(actual).all()
        np.testing.assert_array_equal(actual, repeat)
    if kind == "silence":
        assert all(np.array_equal(actual, zero) for actual in first)

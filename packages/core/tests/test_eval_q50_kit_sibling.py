"""Focused Q50 kit-sibling v2 safety probes."""

import numpy as np
import pytest

from upmixer.eval.q50_kit_sibling import (
    KIT_OUTPUTS,
    evaluate_kit_sibling_arms,
    repair_kit_siblings,
)

SR = 8_000
FRAMES = 24_000


def _hit(
    start: int,
    frequency: float = 440.0,
    gain: float = 1.0,
    decay: float = 24.0,
    polarity: float = 1.0,
) -> np.ndarray:
    length = min(round(SR * 0.16), FRAMES - start)
    time = np.arange(length, dtype=np.float64) / SR
    signal = (
        polarity * gain * np.sin(2.0 * np.pi * frequency * time) * np.exp(-decay * time)
    )
    audio = np.zeros((FRAMES, 2), dtype=np.float32)
    audio[start : start + length, 0] = signal
    audio[start : start + length, 1] = signal * 0.7
    return audio


def _case(
    exemplars: int = 2, candidate: np.ndarray | None = None
) -> tuple[np.ndarray, dict[str, np.ndarray]]:
    stems = {name: np.zeros((FRAMES, 2), dtype=np.float32) for name in KIT_OUTPUTS}
    stems["Snare"] = _hit(4_000)
    if exemplars > 1:
        stems["Snare"] += _hit(8_000)
    stems["Kick"] = _hit(12_000) if candidate is None else candidate
    stems["Kick"] += _hit(18_000, frequency=880.0, gain=0.7)
    parent = sum(stems.values(), np.zeros_like(stems["Kick"]))
    return parent, stems


def _assert_unchanged(
    before: dict[str, np.ndarray], after: dict[str, np.ndarray]
) -> None:
    for name in KIT_OUTPUTS:
        np.testing.assert_array_equal(after[name], before[name])


@pytest.mark.parametrize("arm", ("cue-only", "cue-plus-repetition"))
def test_positive_zero_sum_transfer_for_each_arm(arm: str):
    parent, stems = _case()
    result = repair_kit_siblings(parent, stems, SR, arm)

    assert len(result.proposals) == 1
    proposal = result.proposals[0]
    assert (proposal.donor, proposal.target) == ("Kick", "Snare")
    assert proposal.exemplar_count == (1 if arm == "cue-only" else 2)
    assert np.max(np.abs(result.stems["Snare"] - stems["Snare"])) > 0
    np.testing.assert_allclose(
        result.stems["Snare"] - stems["Snare"],
        -(result.stems["Kick"] - stems["Kick"]),
        atol=1e-7,
        rtol=0,
    )
    total = sum(result.stems.values(), np.zeros_like(parent))
    np.testing.assert_allclose(total, sum(stems.values()), atol=1e-6, rtol=0)
    for name in ("Toms", "Ride", "Crash"):
        np.testing.assert_array_equal(result.stems[name], stems[name])


def test_arms_are_separately_measurable_and_preserve_all_outputs():
    parent, stems = _case()
    results = evaluate_kit_sibling_arms(parent, stems, SR)

    assert set(results) == {"cue-only", "cue-plus-repetition"}
    assert results["cue-only"].repetition_transfer_count == 0
    assert results["cue-plus-repetition"].repetition_transfer_count == 1
    assert set(results["cue-only"].stems) == set(KIT_OUTPUTS)


@pytest.mark.parametrize(
    "candidate",
    (
        pytest.param(np.zeros((FRAMES, 2), dtype=np.float32), id="clean-no-candidate"),
        pytest.param(_hit(12_000, polarity=-1.0), id="phase-inverted"),
        pytest.param(_hit(12_000, frequency=880.0), id="wrong-pitch"),
        pytest.param(_hit(12_000, decay=8.0), id="changed-decay"),
    ),
)
def test_safety_negatives_are_exact_noops(candidate: np.ndarray | None):
    parent, stems = _case(candidate=candidate)
    result = repair_kit_siblings(parent, stems, SR, "cue-plus-repetition")

    assert result.proposals == ()
    _assert_unchanged(stems, result.stems)


def test_occupied_target_and_no_repetition_abstain():
    parent, stems = _case()
    occupied = {name: audio.copy() for name, audio in stems.items()}
    occupied["Snare"] += _hit(12_000, frequency=660.0, gain=0.8)
    occupied_parent = sum(occupied.values(), np.zeros_like(parent))
    occupied_result = repair_kit_siblings(occupied_parent, occupied, SR, "cue-only")
    assert occupied_result.proposals == ()
    _assert_unchanged(occupied, occupied_result.stems)

    single_parent, single_stems = _case(exemplars=1)
    no_repeat_result = repair_kit_siblings(
        single_parent, single_stems, SR, "cue-plus-repetition"
    )
    assert no_repeat_result.proposals == ()
    _assert_unchanged(single_stems, no_repeat_result.stems)


def test_silence_and_validation():
    stems = {name: np.zeros((FRAMES, 2), dtype=np.float32) for name in KIT_OUTPUTS}
    parent = np.zeros_like(stems["Kick"])
    result = repair_kit_siblings(parent, stems, SR)
    assert result.proposals == ()
    _assert_unchanged(stems, result.stems)

    with pytest.raises(ValueError, match="missing"):
        repair_kit_siblings(parent, {"Kick": stems["Kick"]}, SR)
    bad = {name: audio.copy() for name, audio in stems.items()}
    bad["Kick"][0, 0] = np.nan
    with pytest.raises(ValueError, match="finite"):
        repair_kit_siblings(parent, bad, SR)

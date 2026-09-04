"""Paired evaluation report aggregation and bootstrap coverage."""
import json
from types import SimpleNamespace

import pytest

from upmixer.eval.report import EvalReport, StemScore


def _score(
    recording_id: str | None,
    item_id: str | None,
    sdr: float,
    *,
    category: str = "default",
    stem: str = "Vocals",
    fullness: float = 0.5,
    bleedless: float = 0.6,
) -> StemScore:
    return StemScore(
        recording_id=recording_id,
        item_id=item_id,
        category=category,
        stem=stem,
        sdr=sdr,
        fullness=fullness,
        bleedless=bleedless,
    )


def _report(*scores: StemScore) -> EvalReport:
    return EvalReport(settings=SimpleNamespace(), scores=list(scores))


def test_recording_means_names_metrics_and_preserves_existing_groupings():
    report = _report(
        _score("r1", "i1", 1.0, fullness=0.4, bleedless=0.7),
        _score("r1", "i2", 3.0, fullness=0.6, bleedless=0.9),
        _score("r2", "i1", 5.0, fullness=0.8, bleedless=1.0),
    )

    assert report.by_stem() == {"Vocals": (3.0, 0.6, 0.8666666666666667)}
    assert report.recording_means() == [
        {
            "recording_id": "r1",
            "category": "default",
            "stem": "Vocals",
            "item_ids": ["i1", "i2"],
            "n_items": 2,
            "sdr": 2.0,
            "fullness": 0.5,
            "bleedless": 0.8,
        },
        {
            "recording_id": "r2",
            "category": "default",
            "stem": "Vocals",
            "item_ids": ["i1"],
            "n_items": 1,
            "sdr": 5.0,
            "fullness": 0.8,
            "bleedless": 1.0,
        },
    ]


def test_paired_bootstrap_matches_exact_rows_and_exposes_unmatched_coverage():
    baseline = _report(
        _score("r1", "i1", 1.0),
        _score("r1", "i2", 3.0),
        _score("r1", "i3", 5.0),
        _score("r2", "i1", 10.0),
        _score("left-only", "i1", 20.0),
    )
    candidate = _report(
        _score("r1", "i1", 2.0),
        _score("r1", "i2", 4.0),
        _score("r1", "i3", 6.0),
        _score("r2", "i1", 12.0),
        _score("right-only", "i1", 30.0),
        _score("r1", "different-item", 40.0),
    )

    result = baseline.paired_bootstrap(candidate, n_resamples=100, seed=7)
    json.dumps(result)

    assert result["pairs"] == [
        {
            "recording_id": "r1",
            "category": "default",
            "stem": "Vocals",
            "item_ids": ["i1", "i2", "i3"],
            "n_items": 3,
            "left": {"sdr": 3.0, "fullness": 0.5, "bleedless": 0.6},
            "right": {"sdr": 4.0, "fullness": 0.5, "bleedless": 0.6},
            "delta": {"sdr": 1.0, "fullness": 0.0, "bleedless": 0.0},
        },
        {
            "recording_id": "r2",
            "category": "default",
            "stem": "Vocals",
            "item_ids": ["i1"],
            "n_items": 1,
            "left": {"sdr": 10.0, "fullness": 0.5, "bleedless": 0.6},
            "right": {"sdr": 12.0, "fullness": 0.5, "bleedless": 0.6},
            "delta": {"sdr": 2.0, "fullness": 0.0, "bleedless": 0.0},
        },
    ]
    assert result["coverage"] == {
        "matched_items": [
            {
                "recording_id": "r1",
                "item_id": "i1",
                "category": "default",
                "stem": "Vocals",
            },
            {
                "recording_id": "r1",
                "item_id": "i2",
                "category": "default",
                "stem": "Vocals",
            },
            {
                "recording_id": "r1",
                "item_id": "i3",
                "category": "default",
                "stem": "Vocals",
            },
            {
                "recording_id": "r2",
                "item_id": "i1",
                "category": "default",
                "stem": "Vocals",
            },
        ],
        "unmatched_items": [
            {
                "side": "left",
                "recording_id": "left-only",
                "item_id": "i1",
                "category": "default",
                "stem": "Vocals",
            },
            {
                "side": "right",
                "recording_id": "r1",
                "item_id": "different-item",
                "category": "default",
                "stem": "Vocals",
            },
            {
                "side": "right",
                "recording_id": "right-only",
                "item_id": "i1",
                "category": "default",
                "stem": "Vocals",
            },
        ],
        "matched_recordings": ["r1", "r2"],
        "left_only_recordings": ["left-only"],
        "right_only_recordings": ["right-only"],
    }

    intervals = result["confidence_intervals"]
    assert intervals["seed"] == 7
    assert intervals["n_resamples"] == 100
    assert intervals["n_recordings"] == 2
    assert intervals["by_stem"]["Vocals"]["sdr"]["estimate"] == 1.5
    assert set(intervals["by_stem"]["Vocals"]) == {"sdr", "fullness", "bleedless"}


def test_bootstrap_aggregates_excerpts_before_sampling_and_is_seeded():
    baseline = _report(
        _score("r1", "i1", 0.0),
        _score("r1", "i2", 0.0),
        _score("r1", "i3", 0.0),
        _score("r2", "i1", 0.0),
    )
    candidate = _report(
        _score("r1", "i1", 3.0),
        _score("r1", "i2", 3.0),
        _score("r1", "i3", 3.0),
        _score("r2", "i1", 1.0),
    )

    first = baseline.paired_bootstrap(candidate, n_resamples=250, seed=11)
    second = baseline.paired_bootstrap(candidate, n_resamples=250, seed=11)

    assert first == second
    assert first["confidence_intervals"]["n_recordings"] == 2
    assert first["confidence_intervals"]["by_stem"]["Vocals"]["sdr"]["estimate"] == 2.0


@pytest.mark.parametrize(
    ("kwargs", "message"),
    [
        ({"n_resamples": 0}, "n_resamples"),
        ({"confidence": 0.0}, "confidence"),
        ({"confidence": 1.0}, "confidence"),
    ],
)
def test_bootstrap_validates_arguments(kwargs, message):
    report = _report(_score("r1", "i1", 1.0), _score("r2", "i1", 2.0))
    with pytest.raises(ValueError, match=message):
        report.paired_bootstrap(report, **kwargs)


def test_bootstrap_requires_stable_ids_finite_metrics_and_two_recordings():
    with pytest.raises(ValueError, match="recording_id and item_id"):
        _report(_score(None, "i1", 1.0), _score("r2", "i1", 2.0)).paired_bootstrap(
            _report(_score("r1", "i1", 1.0), _score("r2", "i1", 2.0))
        )

    with pytest.raises(ValueError, match="finite"):
        _report(_score("r1", "i1", float("nan"))).recording_means()

    one_recording = _report(_score("r1", "i1", 1.0))
    with pytest.raises(ValueError, match="two matched recordings"):
        one_recording.paired_bootstrap(one_recording)

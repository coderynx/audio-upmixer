"""Paired evaluation report aggregation and bootstrap coverage."""
import json
from types import SimpleNamespace

import pytest

from upmixer.eval.report import CoverageRow, EvalReport, StemScore


def _score(
    recording_id: str | None,
    item_id: str | None,
    sdr: float,
    *,
    category: str = "default",
    stem: str = "Vocals",
    split: str | None = None,
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
        split=split,
    )


def _report(*scores: StemScore, coverage: list[CoverageRow] | None = None) -> EvalReport:
    return EvalReport(settings=SimpleNamespace(), scores=list(scores), coverage=coverage or [])


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
            "split": None,
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
            "split": None,
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
            "split": None,
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
            "split": None,
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
                "split": None,
            },
            {
                "recording_id": "r1",
                "item_id": "i2",
                "category": "default",
                "stem": "Vocals",
                "split": None,
            },
            {
                "recording_id": "r1",
                "item_id": "i3",
                "category": "default",
                "stem": "Vocals",
                "split": None,
            },
            {
                "recording_id": "r2",
                "item_id": "i1",
                "category": "default",
                "stem": "Vocals",
                "split": None,
            },
        ],
        "unmatched_items": [
            {
                "side": "left",
                "recording_id": "left-only",
                "item_id": "i1",
                "category": "default",
                "stem": "Vocals",
                "split": None,
            },
            {
                "side": "right",
                "recording_id": "r1",
                "item_id": "different-item",
                "category": "default",
                "stem": "Vocals",
                "split": None,
            },
            {
                "side": "right",
                "recording_id": "right-only",
                "item_id": "i1",
                "category": "default",
                "stem": "Vocals",
                "split": None,
            },
        ],
        "matched_recordings": ["r1", "r2"],
        "left_only_recordings": ["left-only"],
        "right_only_recordings": ["right-only"],
        "unavailable_items": [],
    }

    intervals = result["confidence_intervals"]
    assert intervals["seed"] == 7
    assert intervals["n_resamples"] == 100
    assert intervals["n_recordings"] == 2
    assert intervals["by_stem"]["Vocals"]["status"] == "ok"
    assert intervals["by_stem"]["Vocals"]["n_recordings"] == 2
    assert intervals["by_stem"]["Vocals"]["sdr"]["estimate"] == 1.5
    assert {"sdr", "fullness", "bleedless"} <= set(intervals["by_stem"]["Vocals"])


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


def test_bootstrap_marks_single_recording_groups_without_blocking_valid_groups():
    baseline = _report(
        _score("r1", "i1", 1.0, stem="Vocals", category="default"),
        _score("r1", "i1", 10.0, stem="Bass", category="rare"),
        _score("r2", "i1", 2.0, stem="Vocals", category="default"),
    )
    candidate = _report(
        _score("r1", "i1", 2.0, stem="Vocals", category="default"),
        _score("r1", "i1", 14.0, stem="Bass", category="rare"),
        _score("r2", "i1", 4.0, stem="Vocals", category="default"),
    )

    result = baseline.paired_bootstrap(candidate, n_resamples=100, seed=3)
    intervals = result["confidence_intervals"]

    assert intervals["n_recordings"] == 2
    assert intervals["by_stem"]["Vocals"]["status"] == "ok"
    assert intervals["by_stem"]["Vocals"]["n_recordings"] == 2
    assert intervals["by_stem"]["Bass"]["status"] == "insufficient_recordings"
    assert intervals["by_stem"]["Bass"]["n_recordings"] == 1
    assert intervals["by_category"]["default"]["status"] == "ok"
    assert intervals["by_category"]["rare"]["status"] == "insufficient_recordings"
    assert intervals["by_category"]["rare"]["n_recordings"] == 1
    for metric in ("sdr", "fullness", "bleedless"):
        assert intervals["by_stem"]["Bass"][metric]["estimate"] == (4.0 if metric == "sdr" else 0.0)
        assert intervals["by_stem"]["Bass"][metric]["low"] is None
        assert intervals["by_stem"]["Bass"][metric]["high"] is None


def test_paired_bootstrap_keeps_split_boundaries_through_pairs_and_sampling():
    baseline = _report(
        _score("r1", "i1", 1.0, split="tuning"),
        _score("r1", "i1", 10.0, split="holdout"),
        _score("r2", "i1", 2.0, split="tuning"),
    )
    candidate = _report(
        _score("r1", "i1", 2.0, split="tuning"),
        _score("r1", "i1", 15.0, split="holdout"),
        _score("r2", "i1", 4.0, split="tuning"),
    )

    result = baseline.paired_bootstrap(candidate, n_resamples=100, seed=5)
    intervals = result["confidence_intervals"]

    assert [(row["split"], row["delta"]["sdr"]) for row in result["pairs"]] == [
        ("holdout", 5.0),
        ("tuning", 1.0),
        ("tuning", 2.0),
    ]
    assert set(intervals["by_stem"]) == {"holdout:Vocals", "tuning:Vocals"}
    assert intervals["by_stem"]["holdout:Vocals"]["split"] == "holdout"
    assert intervals["by_stem"]["holdout:Vocals"]["n_recordings"] == 1
    assert intervals["by_stem"]["holdout:Vocals"]["status"] == "insufficient_recordings"
    assert intervals["by_stem"]["holdout:Vocals"]["sdr"] == {
        "estimate": 5.0,
        "low": None,
        "high": None,
    }
    assert intervals["by_stem"]["tuning:Vocals"]["n_recordings"] == 2
    assert intervals["by_stem"]["tuning:Vocals"]["status"] == "ok"
    assert intervals["by_stem"]["tuning:Vocals"]["sdr"]["estimate"] == 1.5


def test_paired_bootstrap_includes_unavailable_side_coverage():
    baseline = _report(
        _score("r1", "i1", 1.0),
        _score("r2", "i1", 2.0),
        coverage=[
            CoverageRow(
                stem="Crowd",
                category="default",
                status="unavailable",
                recording_id="left-only",
                item_id="i1",
                split="holdout",
            )
        ],
    )
    candidate = _report(
        _score("r1", "i1", 2.0),
        _score("r2", "i1", 4.0),
        coverage=[
            CoverageRow(
                stem="Crowd",
                category="default",
                status="unavailable",
                recording_id="right-only",
                item_id="i1",
                split="holdout",
            )
        ],
    )

    coverage = baseline.paired_bootstrap(candidate)["coverage"]

    assert coverage["left_only_recordings"] == ["left-only"]
    assert coverage["right_only_recordings"] == ["right-only"]
    assert coverage["unavailable_items"] == [
        {
            "side": "left",
            "status": "unavailable",
            "recording_id": "left-only",
            "item_id": "i1",
            "category": "default",
            "stem": "Crowd",
            "split": "holdout",
        },
        {
            "side": "right",
            "status": "unavailable",
            "recording_id": "right-only",
            "item_id": "i1",
            "category": "default",
            "stem": "Crowd",
            "split": "holdout",
        },
    ]


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

"""Aggregation and formatting for evaluation harness results."""
from __future__ import annotations

from collections import defaultdict
from dataclasses import dataclass, field
from typing import TYPE_CHECKING

import numpy as np

if TYPE_CHECKING:
    from upmixer.eval.harness import RunSettings


@dataclass
class StemScore:
    """Scores for one stem on one corpus item."""

    stem: str
    category: str
    sdr: float
    fullness: float
    bleedless: float
    recording_id: str | None = None
    item_id: str | None = None
    split: str | None = None


@dataclass
class CoverageRow:
    """Coverage status for one stem on one corpus item."""

    stem: str
    category: str
    status: str
    recording_id: str | None = None
    item_id: str | None = None
    split: str | None = None


@dataclass
class EvalReport:
    """Full result of an evaluation run: settings plus per-item scores."""

    settings: "RunSettings"
    scores: list[StemScore]
    coverage: list[CoverageRow] = field(default_factory=list)

    def by_stem(self) -> dict[str, tuple[float, float, float]]:
        """Mean (sdr, fullness, bleedless) grouped by canonical stem name."""
        return _grouped_means(self.scores, key=lambda s: s.stem)

    def by_category(self) -> dict[str, tuple[float, float, float]]:
        """Mean (sdr, fullness, bleedless) grouped by regression-probe category."""
        return _grouped_means(self.scores, key=lambda s: s.category)

    def recording_means(self) -> list[dict[str, object]]:
        """Return one named-metric row per recording/category/stem group."""
        score_map = _validated_score_map(self.scores)
        groups: dict[tuple[str, str, str], list[StemScore]] = defaultdict(list)
        for score in score_map.values():
            groups[(score.recording_id, score.category, score.stem)].append(score)

        rows: list[dict[str, object]] = []
        for (recording_id, category, stem), scores in sorted(groups.items()):
            metrics = _mean_metrics(scores)
            rows.append(
                {
                    "recording_id": recording_id,
                    "category": category,
                    "stem": stem,
                    "item_ids": sorted(score.item_id for score in scores),
                    "n_items": len(scores),
                    **metrics,
                }
            )
        return rows

    def paired_bootstrap(
        self,
        other: "EvalReport",
        *,
        n_resamples: int = 2000,
        confidence: float = 0.95,
        seed: int = 0,
    ) -> dict[str, object]:
        """Compare exact rows and bootstrap paired recording-level deltas.

        ``self`` is the left/baseline report and ``other`` is the right/candidate
        report, so each reported delta is ``other - self``. Exact
        ``(recording_id, item_id, category, stem)`` identities are paired first;
        repeated items from one recording are then averaged before resampling.
        """
        _validate_bootstrap_args(n_resamples, confidence, seed)
        left = _validated_score_map(self.scores)
        right = _validated_score_map(other.scores)
        left_keys = set(left)
        right_keys = set(right)
        matched_keys = sorted(left_keys & right_keys)
        if len({key[0] for key in matched_keys}) < 2:
            raise ValueError("paired bootstrap requires at least two matched recordings")

        pairs = _paired_rows(left, right, matched_keys)
        coverage = _paired_coverage(left_keys, right_keys, matched_keys)
        rng = np.random.default_rng(seed)
        confidence_intervals = {
            "confidence": float(confidence),
            "n_resamples": int(n_resamples),
            "seed": int(seed),
            "n_recordings": len({key[0] for key in matched_keys}),
            "by_stem": _bootstrap_groups(left, right, matched_keys, 3, n_resamples, confidence, rng),
            "by_category": _bootstrap_groups(left, right, matched_keys, 2, n_resamples, confidence, rng),
        }
        return {
            "pairs": pairs,
            "coverage": coverage,
            "confidence_intervals": confidence_intervals,
        }


def _grouped_means(scores: list[StemScore], key) -> dict[str, tuple[float, float, float]]:
    groups: dict[str, list[StemScore]] = defaultdict(list)
    for score in scores:
        groups[key(score)].append(score)
    return {
        group: (
            sum(s.sdr for s in items) / len(items),
            sum(s.fullness for s in items) / len(items),
            sum(s.bleedless for s in items) / len(items),
        )
        for group, items in groups.items()
    }


_METRICS = ("sdr", "fullness", "bleedless")


def _validated_score_map(scores: list[StemScore]) -> dict[tuple[str, str, str, str], StemScore]:
    result: dict[tuple[str, str, str, str], StemScore] = {}
    for score in scores:
        if not isinstance(score.recording_id, str) or not score.recording_id.strip() or not isinstance(score.item_id, str) or not score.item_id.strip():
            raise ValueError("paired comparison requires stable recording_id and item_id")
        values = []
        for metric in _METRICS:
            try:
                value = float(getattr(score, metric))
            except (TypeError, ValueError) as exc:
                raise ValueError(f"{metric} scores must be finite numbers") from exc
            if not np.isfinite(value):
                raise ValueError(f"{metric} scores must be finite numbers")
            values.append(value)
        key = (score.recording_id, score.item_id, score.category, score.stem)
        if key in result:
            raise ValueError(f"duplicate score identity: {key!r}")
        result[key] = score
    return result


def _mean_metrics(scores: list[StemScore]) -> dict[str, float]:
    return {
        metric: float(np.mean([float(getattr(score, metric)) for score in scores]))
        for metric in _METRICS
    }


def _metric_values(score: StemScore) -> dict[str, float]:
    return {metric: float(getattr(score, metric)) for metric in _METRICS}


def _paired_rows(
    left: dict[tuple[str, str, str, str], StemScore],
    right: dict[tuple[str, str, str, str], StemScore],
    matched_keys: list[tuple[str, str, str, str]],
) -> list[dict[str, object]]:
    groups: dict[tuple[str, str, str], list[tuple[StemScore, StemScore]]] = defaultdict(list)
    for key in matched_keys:
        groups[(key[0], key[2], key[3])].append((left[key], right[key]))

    rows: list[dict[str, object]] = []
    for (recording_id, category, stem), score_pairs in sorted(groups.items()):
        left_metrics = _mean_metrics([pair[0] for pair in score_pairs])
        right_metrics = _mean_metrics([pair[1] for pair in score_pairs])
        rows.append(
            {
                "recording_id": recording_id,
                "category": category,
                "stem": stem,
                "item_ids": sorted(pair[0].item_id for pair in score_pairs),
                "n_items": len(score_pairs),
                "left": left_metrics,
                "right": right_metrics,
                "delta": {
                    metric: float(right_metrics[metric] - left_metrics[metric])
                    for metric in _METRICS
                },
            }
        )
    return rows


def _identity(key: tuple[str, str, str, str]) -> dict[str, str]:
    recording_id, item_id, category, stem = key
    return {
        "recording_id": recording_id,
        "item_id": item_id,
        "category": category,
        "stem": stem,
    }


def _paired_coverage(
    left_keys: set[tuple[str, str, str, str]],
    right_keys: set[tuple[str, str, str, str]],
    matched_keys: list[tuple[str, str, str, str]],
) -> dict[str, object]:
    left_recordings = {key[0] for key in left_keys}
    right_recordings = {key[0] for key in right_keys}
    unmatched = [
        {"side": "left", **_identity(key)}
        for key in sorted(left_keys - set(matched_keys))
    ]
    unmatched.extend(
        {"side": "right", **_identity(key)}
        for key in sorted(right_keys - set(matched_keys))
    )
    return {
        "matched_items": [_identity(key) for key in matched_keys],
        "unmatched_items": unmatched,
        "matched_recordings": sorted({key[0] for key in matched_keys}),
        "left_only_recordings": sorted(left_recordings - right_recordings),
        "right_only_recordings": sorted(right_recordings - left_recordings),
    }


def _paired_recording_deltas(
    left: dict[tuple[str, str, str, str], StemScore],
    right: dict[tuple[str, str, str, str], StemScore],
    matched_keys: list[tuple[str, str, str, str]],
    group_index: int,
) -> dict[str, dict[str, tuple[float, float, float]]]:
    groups: dict[tuple[str, str], list[tuple[float, float, float]]] = defaultdict(list)
    for key in matched_keys:
        left_values = _metric_values(left[key])
        right_values = _metric_values(right[key])
        groups[(key[0], key[group_index])].append(
            tuple(right_values[metric] - left_values[metric] for metric in _METRICS)
        )

    by_group: dict[str, dict[str, tuple[float, float, float]]] = defaultdict(dict)
    for (recording_id, group), deltas in groups.items():
        means = tuple(float(np.mean([delta[index] for delta in deltas])) for index in range(3))
        by_group[group][recording_id] = means
    return by_group


def _bootstrap_groups(
    left: dict[tuple[str, str, str, str], StemScore],
    right: dict[tuple[str, str, str, str], StemScore],
    matched_keys: list[tuple[str, str, str, str]],
    group_index: int,
    n_resamples: int,
    confidence: float,
    rng: np.random.Generator,
) -> dict[str, dict[str, dict[str, float]]]:
    grouped = _paired_recording_deltas(left, right, matched_keys, group_index)
    result: dict[str, dict[str, dict[str, float]]] = {}
    tail = (1.0 - confidence) / 2.0
    for group, recording_values in sorted(grouped.items()):
        recording_ids = sorted(recording_values)
        values = np.asarray([recording_values[recording_id] for recording_id in recording_ids], dtype=float)
        sample_indices = rng.integers(0, len(recording_ids), size=(n_resamples, len(recording_ids)))
        sample_means = values[sample_indices].mean(axis=1)
        bounds = np.quantile(sample_means, (tail, 1.0 - tail), axis=0, method="linear")
        estimate = values.mean(axis=0)
        result[group] = {
            metric: {
                "estimate": float(estimate[index]),
                "low": float(bounds[0, index]),
                "high": float(bounds[1, index]),
            }
            for index, metric in enumerate(_METRICS)
        }
    return result


def _validate_bootstrap_args(n_resamples: int, confidence: float, seed: int) -> None:
    if isinstance(n_resamples, bool) or not isinstance(n_resamples, (int, np.integer)) or n_resamples < 1:
        raise ValueError("n_resamples must be a positive integer")
    if not isinstance(confidence, (int, float, np.integer, np.floating)) or not np.isfinite(confidence) or not 0 < confidence < 1:
        raise ValueError("confidence must be finite and between 0 and 1")
    if isinstance(seed, bool) or not isinstance(seed, (int, np.integer)) or seed < 0:
        raise ValueError("seed must be a non-negative integer")


def format_report(report: EvalReport) -> str:
    """Render a report as a per-stem, per-category text table.

    Always reports SDR, fullness, and bleedless together (never SDR alone),
    and prefixes the table with the recorded controls and model metadata.
    """
    settings = report.settings
    lines = [
        (
            f"Settings: model={getattr(settings, 'model', None)} "
            f"sample_rate={getattr(settings, 'sample_rate', None)} "
            f"segment_size={getattr(settings, 'segment_size', None)} "
            f"overlap={getattr(settings, 'overlap', None)} "
            f"batch_size={getattr(settings, 'batch_size', None)} "
            f"chunk_duration_s={getattr(settings, 'chunk_duration_s', None)} "
            f"tta={getattr(settings, 'tta', None)} "
            f"pitch_shift={getattr(settings, 'pitch_shift', None)} "
            f"backend={getattr(settings, 'backend', None)} "
            f"model_arch={getattr(settings, 'model_arch', None)} "
            f"model_config_name={getattr(settings, 'model_config_name', None)} "
            "model_native_sample_rate="
            f"{getattr(settings, 'model_native_sample_rate', None)} "
            f"ensemble_algorithm={getattr(settings, 'ensemble_algorithm', None)} "
            f"ensemble_models={getattr(settings, 'ensemble_models', None)}"
        ),
        "",
        "Per-stem (mean SDR dB / fullness / bleedless):",
    ]
    for stem, (mean_sdr, mean_fullness, mean_bleedless) in sorted(report.by_stem().items()):
        lines.append(f"  {stem:<16} SDR={mean_sdr:7.2f}  fullness={mean_fullness:.3f}  bleedless={mean_bleedless:.3f}")

    lines.append("")
    lines.append("Per-category (mean SDR dB / fullness / bleedless):")
    for category, (mean_sdr, mean_fullness, mean_bleedless) in sorted(report.by_category().items()):
        lines.append(f"  {category:<16} SDR={mean_sdr:7.2f}  fullness={mean_fullness:.3f}  bleedless={mean_bleedless:.3f}")

    return "\n".join(lines)

"""Aggregation and formatting for evaluation harness results."""
from __future__ import annotations

import json
from collections import defaultdict
from dataclasses import asdict, dataclass, field, is_dataclass
from pathlib import Path
from typing import TYPE_CHECKING

import numpy as np

from upmixer.execution import write_report

if TYPE_CHECKING:
    from upmixer.eval.types import ItemRunSettings, RunSettings


_ScoreKey = tuple[str, str, str, str, str | None]


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
    """Coverage status for one stem on one corpus item.

    ``detail`` is a concise diagnostic for failed, absent, or skipped rows.
    """

    stem: str
    category: str
    status: str
    recording_id: str | None = None
    item_id: str | None = None
    split: str | None = None
    detail: str | None = None


@dataclass
class EvalReport:
    """Full result of an evaluation run: settings plus per-item scores."""

    settings: "RunSettings | None"
    scores: list[StemScore]
    coverage: list[CoverageRow] = field(default_factory=list)
    item_settings: list["ItemRunSettings"] = field(default_factory=list)
    protocol_id: str | None = None
    corpus_id: str | None = None
    code_revision: str | None = None
    origin_provenance: list[dict[str, object]] = field(default_factory=list)

    def by_stem(self) -> dict[str, tuple[float, float, float]]:
        """Mean (sdr, fullness, bleedless) grouped by canonical stem name."""
        return _grouped_means(self.scores, key=lambda s: s.stem)

    def by_category(self) -> dict[str, tuple[float, float, float]]:
        """Mean (sdr, fullness, bleedless) grouped by regression-probe category."""
        return _grouped_means(self.scores, key=lambda s: s.category)

    def recording_means(self) -> list[dict[str, object]]:
        """Return one named-metric row per recording/category/stem/split group."""
        score_map = _validated_score_map(self.scores)
        groups: dict[tuple[str, str, str, str | None], list[StemScore]] = defaultdict(list)
        for score in score_map.values():
            groups[(score.recording_id, score.category, score.stem, score.split)].append(score)

        rows: list[dict[str, object]] = []
        for (recording_id, category, stem, split), scores in sorted(groups.items(), key=lambda item: _sort_key(item[0])):
            metrics = _mean_metrics(scores)
            rows.append(
                {
                    "recording_id": recording_id,
                    "category": category,
                    "stem": stem,
                    "split": split,
                    "item_ids": sorted(score.item_id for score in scores),
                    "n_items": len(scores),
                    **metrics,
                }
            )
        return rows

    def to_dict(self, *, paired_bootstrap: dict[str, object] | None = None) -> dict[str, object]:
        """Return a versioned, JSON-safe evaluation report."""
        if self.settings is None:
            settings = None
        else:
            settings = asdict(self.settings) if is_dataclass(self.settings) else vars(self.settings)
        payload: dict[str, object] = {
            "schema_version": 1,
            "protocol_id": self.protocol_id,
            "corpus_id": self.corpus_id,
            "code_revision": self.code_revision,
            "settings": settings,
            "item_settings": [asdict(row) for row in self.item_settings],
            "scores": [asdict(score) for score in self.scores],
            "coverage": [asdict(row) for row in self.coverage],
            "origin_provenance": self.origin_provenance,
            "by_stem": _named_means(self.by_stem()),
            "by_category": _named_means(self.by_category()),
            "by_recording": self.recording_means(),
        }
        if paired_bootstrap is not None:
            payload["paired_bootstrap"] = paired_bootstrap
        return _validated_json(_json_safe(payload))

    def to_json(
        self,
        indent: int | None = 2,
        *,
        paired_bootstrap: dict[str, object] | None = None,
    ) -> str:
        """Return the versioned report as JSON."""
        return json.dumps(
            self.to_dict(paired_bootstrap=paired_bootstrap),
            indent=indent,
            allow_nan=False,
        )

    def write_json(
        self,
        path: str | Path,
        *,
        paired_bootstrap: dict[str, object] | None = None,
    ) -> None:
        """Write the versioned report atomically as JSON."""
        write_report(path, self.to_dict(paired_bootstrap=paired_bootstrap))

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
        ``(recording_id, item_id, category, stem, split)`` identities are paired first;
        repeated items from one recording are then averaged before resampling.
        """
        _validate_bootstrap_args(n_resamples, confidence, seed)
        left = _validated_score_map(self.scores)
        right = _validated_score_map(other.scores)
        left_coverage = _validated_coverage_map(self.coverage)
        right_coverage = _validated_coverage_map(other.coverage)
        left_keys = set(left)
        right_keys = set(right)
        matched_keys = sorted(left_keys & right_keys, key=_sort_key)
        if len({key[0] for key in matched_keys}) < 2:
            raise ValueError("paired bootstrap requires at least two matched recordings")

        pairs = _paired_rows(left, right, matched_keys)
        coverage = _paired_coverage(
            left_keys,
            right_keys,
            matched_keys,
            left_coverage,
            right_coverage,
        )
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
_COVERAGE_STATUSES = ("scored", "unavailable", "failed", "absent", "skipped")


def _named_means(grouped: dict[str, tuple[float, float, float]]) -> dict[str, dict[str, float]]:
    return {
        group: {
            metric: float(values[index])
            for index, metric in enumerate(_METRICS)
        }
        for group, values in grouped.items()
    }


def _json_safe(value):
    if isinstance(value, np.generic):
        return _json_safe(value.item())
    if isinstance(value, np.ndarray):
        return _json_safe(value.tolist())
    if is_dataclass(value):
        return _json_safe(asdict(value))
    if isinstance(value, dict):
        return {str(key): _json_safe(item) for key, item in value.items()}
    if isinstance(value, (list, tuple)):
        return [_json_safe(item) for item in value]
    return value


def _validated_json(payload: dict[str, object]) -> dict[str, object]:
    json.dumps(payload, allow_nan=False)
    return payload


def _validated_score_map(scores: list[StemScore]) -> dict[_ScoreKey, StemScore]:
    result: dict[_ScoreKey, StemScore] = {}
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
        key = (score.recording_id, score.item_id, score.category, score.stem, score.split)
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
    left: dict[_ScoreKey, StemScore],
    right: dict[_ScoreKey, StemScore],
    matched_keys: list[_ScoreKey],
) -> list[dict[str, object]]:
    groups: dict[tuple[str, str, str, str | None], list[tuple[StemScore, StemScore]]] = defaultdict(list)
    for key in matched_keys:
        groups[(key[0], key[2], key[3], key[4])].append((left[key], right[key]))

    rows: list[dict[str, object]] = []
    for (recording_id, category, stem, split), score_pairs in sorted(groups.items(), key=lambda item: _sort_key(item[0])):
        left_metrics = _mean_metrics([pair[0] for pair in score_pairs])
        right_metrics = _mean_metrics([pair[1] for pair in score_pairs])
        rows.append(
            {
                "recording_id": recording_id,
                "category": category,
                "stem": stem,
                "split": split,
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


def _identity(key: _ScoreKey) -> dict[str, object]:
    recording_id, item_id, category, stem, split = key
    return {
        "recording_id": recording_id,
        "item_id": item_id,
        "category": category,
        "stem": stem,
        "split": split,
    }


def _sort_key(key: tuple[object, ...]) -> tuple[str, ...]:
    return tuple("" if value is None else str(value) for value in key)


def _validated_coverage_map(rows: list[CoverageRow]) -> dict[_ScoreKey, CoverageRow]:
    result: dict[_ScoreKey, CoverageRow] = {}
    for row in rows:
        if not isinstance(row.recording_id, str) or not row.recording_id.strip() or not isinstance(row.item_id, str) or not row.item_id.strip():
            raise ValueError("paired comparison requires stable recording_id and item_id")
        key = (row.recording_id, row.item_id, row.category, row.stem, row.split)
        if key in result:
            raise ValueError(f"duplicate coverage identity: {key!r}")
        result[key] = row
    return result


def _paired_coverage(
    left_keys: set[_ScoreKey],
    right_keys: set[_ScoreKey],
    matched_keys: list[_ScoreKey],
    left_coverage: dict[_ScoreKey, CoverageRow],
    right_coverage: dict[_ScoreKey, CoverageRow],
) -> dict[str, object]:
    left_all_keys = left_keys | set(left_coverage)
    right_all_keys = right_keys | set(right_coverage)
    unmatched = [
        {"side": "left", **_identity(key)}
        for key in sorted(left_keys - set(matched_keys), key=_sort_key)
    ]
    unmatched.extend(
        {"side": "right", **_identity(key)}
        for key in sorted(right_keys - set(matched_keys), key=_sort_key)
    )
    unavailable = [
        {"side": "left", "status": row.status, **_identity(key)}
        for key, row in sorted(left_coverage.items(), key=lambda item: _sort_key(item[0]))
        if row.status != "scored"
    ]
    unavailable.extend(
        {"side": "right", "status": row.status, **_identity(key)}
        for key, row in sorted(right_coverage.items(), key=lambda item: _sort_key(item[0]))
        if row.status != "scored"
    )
    return {
        "matched_items": [_identity(key) for key in matched_keys],
        "unmatched_items": unmatched,
        "matched_recordings": sorted({key[0] for key in matched_keys}),
        "left_only_recordings": sorted({key[0] for key in left_all_keys} - {key[0] for key in right_all_keys}),
        "right_only_recordings": sorted({key[0] for key in right_all_keys} - {key[0] for key in left_all_keys}),
        "unavailable_items": unavailable,
    }


def _paired_recording_deltas(
    left: dict[_ScoreKey, StemScore],
    right: dict[_ScoreKey, StemScore],
    matched_keys: list[_ScoreKey],
    group_index: int,
) -> dict[tuple[str | None, str], dict[str, tuple[float, float, float]]]:
    groups: dict[tuple[str, str | None, str], list[tuple[float, float, float]]] = defaultdict(list)
    for key in matched_keys:
        left_values = _metric_values(left[key])
        right_values = _metric_values(right[key])
        groups[(key[0], key[4], key[group_index])].append(
            tuple(right_values[metric] - left_values[metric] for metric in _METRICS)
        )

    by_group: dict[tuple[str | None, str], dict[str, tuple[float, float, float]]] = defaultdict(dict)
    for (recording_id, split, group), deltas in groups.items():
        means = tuple(float(np.mean([delta[index] for delta in deltas])) for index in range(3))
        by_group[(split, group)][recording_id] = means
    return by_group


def _bootstrap_groups(
    left: dict[_ScoreKey, StemScore],
    right: dict[_ScoreKey, StemScore],
    matched_keys: list[_ScoreKey],
    group_index: int,
    n_resamples: int,
    confidence: float,
    rng: np.random.Generator,
) -> list[dict[str, object]]:
    grouped = _paired_recording_deltas(left, right, matched_keys, group_index)
    result: list[dict[str, object]] = []
    tail = (1.0 - confidence) / 2.0
    for (split, group), recording_values in sorted(grouped.items(), key=lambda item: _sort_key(item[0])):
        recording_ids = sorted(recording_values)
        values = np.asarray([recording_values[recording_id] for recording_id in recording_ids], dtype=float)
        estimate = values.mean(axis=0)
        group_result: dict[str, object] = {
            "stem" if group_index == 3 else "category": group,
            "split": split,
            "status": "ok" if len(recording_ids) >= 2 else "insufficient_recordings",
            "n_recordings": len(recording_ids),
        }
        if len(recording_ids) >= 2:
            sample_indices = rng.integers(0, len(recording_ids), size=(n_resamples, len(recording_ids)))
            sample_means = values[sample_indices].mean(axis=1)
            bounds = np.quantile(sample_means, (tail, 1.0 - tail), axis=0, method="linear")
        else:
            bounds = None
        group_result.update(
            {
                metric: {
                    "estimate": float(estimate[index]),
                    "low": None if bounds is None else float(bounds[0, index]),
                    "high": None if bounds is None else float(bounds[1, index]),
                }
                for index, metric in enumerate(_METRICS)
            }
        )
        result.append(group_result)
    return result


def _validate_bootstrap_args(n_resamples: int, confidence: float, seed: int) -> None:
    if isinstance(n_resamples, bool) or not isinstance(n_resamples, (int, np.integer)) or n_resamples < 1:
        raise ValueError("n_resamples must be a positive integer")
    if not isinstance(confidence, (int, float, np.integer, np.floating)) or not np.isfinite(confidence) or not 0 < confidence < 1:
        raise ValueError("confidence must be finite and between 0 and 1")
    if isinstance(seed, bool) or not isinstance(seed, (int, np.integer)) or seed < 0:
        raise ValueError("seed must be a non-negative integer")


_TREE_FIELDS = (
    "input_sample_rate", "separation_sample_rate", "output_sample_rate", "scoring_sample_rate",
    "stem_primary_remask", "stem_drum_remask", "stem_bleed_reduction", "stem_ensemble",
    "stem_silence_skip", "stem_silence_threshold_db", "stem_silence_min_duration_s",
    "stem_silence_crossfade_ms", "stem_silence_pad_ms",
)
_RUNTIME_FIELDS = ("checkpoint_sha256", "model_config_sha256", "runtime_precision", "normalization_policy", "oom_fallback_attempts", "oom_fallback_count")


def _short_hash(value: object) -> str:
    if value is None:
        return "-"
    text = str(value)
    return text if len(text) <= 12 else f"{text[:12]}..."


def _format_plan(plan: object) -> str:
    if not isinstance(plan, dict):
        return str(plan)
    compact = dict(plan)
    for key in ("stems_hash", "inference_hash"):
        compact[key] = _short_hash(compact.get(key))
    return json.dumps(compact, sort_keys=True, separators=(",", ":"))


def _format_tree_context(settings: object) -> str:
    plan = getattr(settings, "plan", None)
    if plan is None and not any(
        getattr(settings, field, None) is not None for field in _TREE_FIELDS
    ):
        return ""
    text = " ".join(f"{field}={getattr(settings, field, None)}" for field in _TREE_FIELDS)
    if plan is not None:
        text += f" plan={_format_plan(plan)}"
    return f" {text}"


def _format_runtime_provenance(settings: object) -> str:
    values = {field: getattr(settings, field, None) for field in _RUNTIME_FIELDS}
    if not any(value not in (None, (), 0) for value in values.values()):
        return ""
    attempts = json.dumps(
        values["oom_fallback_attempts"] or (), sort_keys=True, separators=(",", ":")
    )
    return (
        f" checkpoint_sha256={_short_hash(values['checkpoint_sha256'])}"
        f" model_config_sha256={_short_hash(values['model_config_sha256'])}"
        f" runtime_precision={values['runtime_precision']}"
        f" normalization_policy={values['normalization_policy']}"
        f" oom_fallback_attempts={attempts}"
        f" oom_fallback_count={values['oom_fallback_count']}"
    )


def _format_settings(settings: object, *, include_ensemble: bool = False) -> str:
    text = (
        f"model={getattr(settings, 'model', None)} "
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
        f"device={getattr(settings, 'device', None)}"
    )
    if include_ensemble:
        text += (
            f" ensemble_algorithm={getattr(settings, 'ensemble_algorithm', None)}"
            f" ensemble_models={getattr(settings, 'ensemble_models', None)}"
        )
    text += "".join(f" {field}={getattr(settings, field)}" for field in ("rate_arm", "input_frame_count", "separation_frame_count", "output_frame_count", "resampler") if getattr(settings, field, None) is not None)
    return text + _format_tree_context(settings) + _format_runtime_provenance(settings)


def format_report(report: EvalReport) -> str:
    """Render a report as metric tables followed by coverage counts.

    Always reports SDR, fullness, and bleedless together (never SDR alone),
    and prefixes the table with the recorded controls and model metadata.
    """
    settings = report.settings
    lines = [
        f"Protocol: {report.protocol_id}",
        f"Corpus: {report.corpus_id}",
        f"Code revision: {report.code_revision}",
    ]
    if settings is None:
        lines.append(
            "Settings vary by item:"
            if report.item_settings
            else "Settings unavailable: no item completed"
        )
        for row in report.item_settings:
            row_settings = getattr(row, "settings", None)
            lines.append(
                "Item "
                f"recording_id={getattr(row, 'recording_id', None)} "
                f"item_id={getattr(row, 'item_id', None)} "
                f"split={getattr(row, 'split', None)} "
                f"category={getattr(row, 'category', None)} "
                f"{_format_settings(row_settings, include_ensemble=True)}"
            )
            lines.extend(
                f"  Stage {index}: {_format_settings(stage)}"
                for index, stage in enumerate(
                    getattr(row_settings, "stage_settings", ()) or (), 1
                )
            )
    else:
        stage_settings = getattr(settings, "stage_settings", ()) or ()
        lines.extend(
            [
                f"Settings: {_format_settings(settings, include_ensemble=True)}",
                *(
                    f"Stage {index}: {_format_settings(stage)}"
                    for index, stage in enumerate(stage_settings, 1)
                ),
            ]
        )
    lines.extend(
        [
            "",
            "Per-stem (mean SDR dB / fullness / bleedless):",
        ]
    )
    for stem, (mean_sdr, mean_fullness, mean_bleedless) in sorted(report.by_stem().items()):
        lines.append(f"  {stem:<16} SDR={mean_sdr:7.2f}  fullness={mean_fullness:.3f}  bleedless={mean_bleedless:.3f}")

    lines.append("")
    lines.append("Per-category (mean SDR dB / fullness / bleedless):")
    for category, (mean_sdr, mean_fullness, mean_bleedless) in sorted(report.by_category().items()):
        lines.append(f"  {category:<16} SDR={mean_sdr:7.2f}  fullness={mean_fullness:.3f}  bleedless={mean_bleedless:.3f}")

    status_counts: dict[str, int] = defaultdict(int)
    for row in report.coverage:
        status_counts[row.status] += 1
    lines.append("")
    lines.append(f"Coverage: total={len(report.coverage)}")
    lines.extend(
        f"  {status}: {status_counts.get(status, 0)}"
        for status in sorted(_COVERAGE_STATUSES)
    )
    lines.extend(
        f"  {status}: {status_counts[status]}"
        for status in sorted(set(status_counts) - set(_COVERAGE_STATUSES))
    )
    for row in report.coverage:
        detail = getattr(row, "detail", None)
        if detail and row.status in {"failed", "absent", "skipped"}:
            identity = (
                f"recording_id={row.recording_id} item_id={row.item_id} "
                f"split={row.split} category={row.category} stem={row.stem}"
            )
            lines.append(f"  {row.status} {identity}: {detail}")

    return "\n".join(lines)

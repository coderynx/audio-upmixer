"""Text formatting for evaluation reports."""

from __future__ import annotations

import json
from collections import defaultdict
from typing import TYPE_CHECKING

if TYPE_CHECKING:
    from upmixer.eval.report import EvalReport

_COVERAGE_STATUSES = ("scored", "unavailable", "failed", "absent", "skipped")

_TREE_FIELDS = (
    "input_sample_rate",
    "separation_sample_rate",
    "output_sample_rate",
    "scoring_sample_rate",
    "stem_primary_remask",
    "stem_drum_remask",
    "stem_bleed_reduction",
    "stem_ensemble",
    "stem_silence_skip",
    "stem_silence_threshold_db",
    "stem_silence_min_duration_s",
    "stem_silence_crossfade_ms",
    "stem_silence_pad_ms",
)
_RUNTIME_FIELDS = (
    "checkpoint_sha256",
    "model_config_sha256",
    "runtime_precision",
    "normalization_policy",
    "oom_fallback_attempts",
    "oom_fallback_count",
)


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
    text = " ".join(
        f"{field}={getattr(settings, field, None)}" for field in _TREE_FIELDS
    )
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
    text += "".join(
        f" {field}={getattr(settings, field)}"
        for field in (
            "rate_arm",
            "input_frame_count",
            "separation_frame_count",
            "output_frame_count",
            "resampler",
        )
        if getattr(settings, field, None) is not None
    )
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
    for stem, (mean_sdr, mean_fullness, mean_bleedless) in sorted(
        report.by_stem().items()
    ):
        lines.append(
            f"  {stem:<16} SDR={mean_sdr:7.2f}  fullness={mean_fullness:.3f}  bleedless={mean_bleedless:.3f}"
        )

    lines.append("")
    lines.append("Per-category (mean SDR dB / fullness / bleedless):")
    for category, (mean_sdr, mean_fullness, mean_bleedless) in sorted(
        report.by_category().items()
    ):
        lines.append(
            f"  {category:<16} SDR={mean_sdr:7.2f}  fullness={mean_fullness:.3f}  bleedless={mean_bleedless:.3f}"
        )

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

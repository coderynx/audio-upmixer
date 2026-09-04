"""Aggregation and formatting for evaluation harness results."""
from __future__ import annotations

from collections import defaultdict
from dataclasses import dataclass
from typing import TYPE_CHECKING

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
class EvalReport:
    """Full result of an evaluation run: settings plus per-item scores."""

    settings: "RunSettings"
    scores: list[StemScore]

    def by_stem(self) -> dict[str, tuple[float, float, float]]:
        """Mean (sdr, fullness, bleedless) grouped by canonical stem name."""
        return _grouped_means(self.scores, key=lambda s: s.stem)

    def by_category(self) -> dict[str, tuple[float, float, float]]:
        """Mean (sdr, fullness, bleedless) grouped by regression-probe category."""
        return _grouped_means(self.scores, key=lambda s: s.category)


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

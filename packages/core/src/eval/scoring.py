"""Stem score construction shared by ordinary and cascade evaluation."""

from __future__ import annotations

from dataclasses import asdict
from typing import TYPE_CHECKING

import numpy as np
import soundfile as sf

from upmixer.eval.metrics import bleedless, fullness, sdr
from upmixer.eval.reference_targets import (
    estimate_components as _estimate_components,
    sum_estimate_components as _sum_estimate_components,
    validate_audio as _validate_audio,
)
from upmixer.eval.report import StemScore

if TYPE_CHECKING:
    from upmixer.eval.cascade import CascadeEvaluationResult
    from upmixer.eval.corpus import CorpusItem


def _score_stem(
    item: "CorpusItem",
    stem_name: str,
    ref_path: str,
    estimate: np.ndarray,
    sample_rate: int,
) -> StemScore:
    reference, reference_rate = sf.read(ref_path, dtype="float32", always_2d=True)
    if reference_rate != sample_rate:
        raise ValueError(
            f"reference {ref_path} sample rate {reference_rate} does "
            f"not match evaluation sample rate {sample_rate}"
        )
    _validate_audio(reference, f"reference {ref_path}")
    _validate_audio(estimate, f"estimate {stem_name}")
    if reference.shape[1] != estimate.shape[1]:
        raise ValueError(
            f"channel count mismatch for {stem_name}: reference has "
            f"{reference.shape[1]}, estimate has {estimate.shape[1]}"
        )
    if reference.shape[0] != estimate.shape[0]:
        raise ValueError(
            f"frame count mismatch for {stem_name}: reference has "
            f"{reference.shape[0]}, estimate has {estimate.shape[0]}"
        )
    return StemScore(
        stem=stem_name,
        category=item.category,
        sdr=sdr(reference, estimate),
        fullness=fullness(reference, estimate, sample_rate),
        bleedless=bleedless(reference, estimate, sample_rate),
        recording_id=item.recording_id,
        item_id=item.item_id,
        split=item.split,
    )


def score_cascade_arms(
    item: "CorpusItem",
    result: "CascadeEvaluationResult",
    sample_rate: int,
) -> list[dict[str, object]]:
    rows: list[dict[str, object]] = []
    for arm in result.arms:
        for stem_name, ref_path in item.stems.items():
            estimate = _sum_estimate_components(
                stem_name,
                _estimate_components(item, stem_name),
                arm.stems,
            )
            score = _score_stem(item, stem_name, ref_path, estimate, sample_rate)
            rows.append(
                {
                    "arm_id": arm.arm_id,
                    "role": arm.role,
                    "input": arm.input_label,
                    **asdict(score),
                }
            )
    return rows

"""Runs separation over a corpus and scores it against ground truth.

``separate_for_eval`` drives the public ``StemSeparator`` — the same
inference path production code uses — and records every setting that
affects its output, so scores are never reported without the configuration
that produced them (see ``docs/evaluation_harness.md``). ``evaluate_corpus``
takes a pluggable separation callable so tests can substitute a fast, offline
stand-in without downloading model weights.
"""
from __future__ import annotations

from dataclasses import dataclass
from typing import Callable

import numpy as np
import soundfile as sf

from upmixer.eval.corpus import ReferenceCorpus
from upmixer.eval.metrics import bleedless, fullness, sdr
from upmixer.eval.report import EvalReport, StemScore
from upmixer.separation.separator import DEFAULT_MODEL, StemSeparator
from upmixer.separation.stem_plan import ENSEMBLE_ALGORITHM, MODEL_ENSEMBLE

SeparateFn = Callable[[str], tuple[dict[str, np.ndarray], "RunSettings"]]


def _validate_audio(array: np.ndarray, label: str) -> None:
    if not isinstance(array, np.ndarray) or array.ndim != 2:
        raise ValueError(f"{label} must be a 2D array (frames, channels)")
    if not array.size or not array.shape[0] or not array.shape[1]:
        raise ValueError(f"{label} must not be empty")
    if not np.issubdtype(array.dtype, np.number) or not np.all(np.isfinite(array)):
        raise ValueError(f"{label} must contain finite numeric values")


@dataclass
class RunSettings:
    """Inference configuration recorded alongside every score.

    Scores without settings are noise: a model swap, segment-size change, or
    ensemble config change all alter output, so every EvalReport carries the
    exact settings used to produce it.
    """

    model: str
    sample_rate: int
    segment_size: int | None = None
    overlap: int | None = None
    batch_size: int | None = None
    ensemble_algorithm: str | None = None
    ensemble_models: tuple[str, ...] | None = None


def separate_for_eval(
    mixture_path: str,
    sample_rate: int,
    model: str = DEFAULT_MODEL,
    batch_size: int | None = None,
    segment_size: int | None = None,
    chunk_duration_s: float | None = None,
    overlap: int | None = None,
    stem_ensemble: bool = False,
) -> tuple[dict[str, np.ndarray], RunSettings]:
    """Separate a mixture with the real ``StemSeparator`` and record settings.

    Args:
        mixture_path: Path to the mixture audio file.
        sample_rate:  Output sample rate for separated stems.
        model:        Model filename (registry name), defaults to the
                      package's default model.
        batch_size, segment_size, chunk_duration_s, overlap: Forwarded to
            ``StemSeparator``; ``None`` selects its backend-aware defaults.
        stem_ensemble: Run the production fixed BS-Roformer-SW + SCNet
            primary-stem ensemble.

    Returns:
        (stems, settings) — canonical stem name -> (n_samples, 2) float32
        array, and the RunSettings actually used.
    """
    if stem_ensemble:
        if model != DEFAULT_MODEL:
            raise ValueError(
                "stem_ensemble uses the registered BS-Roformer-SW primary model"
            )
        from upmixer.config import UpmixConfig
        from upmixer.separation.stem_pipeline import StemUpmixPipeline

        pipeline = StemUpmixPipeline(UpmixConfig(
            output_sample_rate=sample_rate,
            stem_batch_size=batch_size,
            stem_segment_size=segment_size,
            stem_chunk_duration_s=chunk_duration_s,
            stem_overlap=overlap,
            stem_ensemble=True,
        ))
        try:
            stems = pipeline._separate(
                mixture_path, None, lambda _message, _fraction: None
            ).all_stems
        finally:
            pipeline.close()
    else:
        separator = StemSeparator(
            model=model,
            sample_rate=sample_rate,
            batch_size=batch_size,
            segment_size=segment_size,
            chunk_duration_s=chunk_duration_s,
            overlap=overlap,
        )
        try:
            stems = separator.separate(mixture_path)
        finally:
            separator.close()
    settings = RunSettings(
        model=model,
        sample_rate=sample_rate,
        segment_size=segment_size,
        batch_size=batch_size,
        overlap=overlap,
        ensemble_algorithm=ENSEMBLE_ALGORITHM if stem_ensemble else None,
        ensemble_models=(model, MODEL_ENSEMBLE) if stem_ensemble else None,
    )
    return stems, settings


def evaluate_corpus(
    corpus: ReferenceCorpus,
    separate_fn: SeparateFn,
    sample_rate: int,
) -> EvalReport:
    """Score a separation run over every item in a corpus.

    For each corpus item, calls ``separate_fn(item.mixture)`` and compares
    every stem the estimate and the reference share by name against the
    reference audio, computing SDR + fullness + bleedless (never SDR alone,
    per the harness requirements).

    Args:
        corpus:      Reference corpus to evaluate against.
        separate_fn: Callable producing (stems, RunSettings) for a mixture
            path — either ``separate_for_eval`` (bound to fixed settings via
            ``functools.partial``) for real inference, or a test double.
        sample_rate: Sample rate of the reference audio (used for the
            magnitude-STFT fullness/bleedless computation).

    Returns:
        EvalReport with one StemScore per (item, shared stem) and the
        consistent RunSettings used across the evaluation run.
    """
    scores: list[StemScore] = []
    settings: RunSettings | None = None
    for item in corpus.items:
        if not item.stems:
            raise ValueError(f"item {item.item_id or item.mixture} has no reference stems")
        estimate_stems, item_settings = separate_fn(item.mixture)
        if not isinstance(estimate_stems, dict) or not estimate_stems:
            raise ValueError(f"item {item.item_id or item.mixture} returned empty outputs")
        missing = sorted(set(item.stems) - set(estimate_stems))
        if missing:
            raise ValueError(
                f"item {item.item_id or item.mixture} missing required stem(s): "
                f"{', '.join(missing)}"
            )
        if not isinstance(item_settings, RunSettings):
            raise ValueError("separation returned invalid RunSettings")
        if item_settings.sample_rate != sample_rate:
            raise ValueError(
                f"RunSettings sample rate {item_settings.sample_rate} does not "
                f"match evaluation sample rate {sample_rate}"
            )
        if settings is None:
            settings = item_settings
        elif item_settings != settings:
            raise ValueError("inconsistent RunSettings across corpus items")
        for stem_name, ref_path in item.stems.items():
            reference, reference_rate = sf.read(ref_path, dtype="float32", always_2d=True)
            if reference_rate != sample_rate:
                raise ValueError(
                    f"reference {ref_path} sample rate {reference_rate} does not "
                    f"match evaluation sample rate {sample_rate}"
                )
            _validate_audio(reference, f"reference {ref_path}")
            estimate = estimate_stems[stem_name]
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
            scores.append(
                StemScore(
                    stem=stem_name,
                    category=item.category,
                    sdr=sdr(reference, estimate),
                    fullness=fullness(reference, estimate, sample_rate),
                    bleedless=bleedless(reference, estimate, sample_rate),
                    recording_id=item.recording_id,
                    item_id=item.item_id,
                    split=item.split,
                )
            )
    if settings is None:
        raise ValueError("corpus has no items to evaluate")
    return EvalReport(settings=settings, scores=scores)

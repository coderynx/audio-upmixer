"""Runs separation over a corpus and scores it against ground truth.

``separate_for_eval`` drives the public ``StemSeparator`` — the same
inference path production code uses — and records every setting that
affects its output, so scores are never reported without the configuration
that produced them (see ``docs/evaluation_harness.md``). ``evaluate_corpus``
takes a pluggable separation callable so tests can substitute a fast, offline
stand-in without downloading model weights.
"""
from __future__ import annotations

from dataclasses import dataclass, replace
from tempfile import TemporaryDirectory
from typing import Callable

import numpy as np
import soundfile as sf

from upmixer.config import UpmixConfig
from upmixer.eval.corpus import ReferenceCorpus
from upmixer.eval.metrics import bleedless, fullness, sdr
from upmixer.eval.report import CoverageRow, EvalReport, StemScore
from upmixer.separation.separator import (
    DEFAULT_MODEL,
    SeparationSettings,
    StemSeparator,
)
from upmixer.separation.stem_plan import (
    ENSEMBLE_ALGORITHM,
    MODEL_ENSEMBLE,
    MODEL_PRIMARY,
)

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
    requested controls and the observed model metadata. Optional controls stay
    ``None`` when this single-model boundary cannot resolve them.
    """

    model: str
    sample_rate: int
    segment_size: int | None = None
    overlap: int | None = None
    batch_size: int | None = None
    ensemble_algorithm: str | None = None
    ensemble_models: tuple[str, ...] | None = None
    chunk_duration_s: float | None = None
    tta: bool = False
    pitch_shift: float | None = None
    backend: str | None = None
    model_arch: str | None = None
    model_config_name: str | None = None
    model_native_sample_rate: int | None = None
    stage_settings: tuple[SeparationSettings, ...] = ()


def separate_for_eval(
    mixture_path: str,
    sample_rate: int,
    model: str = DEFAULT_MODEL,
    batch_size: int | None = None,
    segment_size: int | None = None,
    chunk_duration_s: float | None = None,
    overlap: int | None = None,
    stem_ensemble: bool = False,
    tta: bool = False,
    pitch_shift: float | None = None,
) -> tuple[dict[str, np.ndarray], RunSettings]:
    """Separate a mixture with the real ``StemSeparator`` and record settings.

    Args:
        mixture_path: Path to the mixture audio file.
        sample_rate:  Output sample rate for separated stems.
        model:        Model filename (registry name), defaults to the
                      package's default model.
        batch_size, segment_size, chunk_duration_s, overlap: Forwarded to
            ``StemSeparator``; ``None`` selects its backend-aware defaults and
            remains recorded as an unresolved request.
        stem_ensemble: Run the production fixed BS-Roformer-SW + SCNet
            primary-stem ensemble.
        tta: Forward test-time augmentation to the selected separator.
        pitch_shift: Forward the optional pitch-register rescue ratio.

    Returns:
        (stems, settings) — canonical stem name -> (n_samples, 2) float32
        array, and the requested controls plus observed model metadata.
    """
    model_arch = None
    model_config_name = None
    model_native_sample_rate = None
    backend = None
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
            stem_tta=tta,
            stem_pitch_shift=pitch_shift,
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
            tta=tta,
            pitch_shift=pitch_shift,
        )
        try:
            stems = separator.separate(mixture_path)
            backend = separator.backend
        finally:
            separator.close()
        from upmixer.separation.inference.config import load_model_config
        from upmixer.separation.inference.registry import get_model_spec

        spec = get_model_spec(model)
        config = load_model_config(spec.config_name)
        model_arch = spec.arch
        model_config_name = spec.config_name
        model_native_sample_rate = config.sample_rate
    settings = RunSettings(
        model=model,
        sample_rate=sample_rate,
        segment_size=segment_size,
        batch_size=batch_size,
        overlap=overlap,
        ensemble_algorithm=ENSEMBLE_ALGORITHM if stem_ensemble else None,
        ensemble_models=(model, MODEL_ENSEMBLE) if stem_ensemble else None,
        chunk_duration_s=chunk_duration_s,
        tta=tta,
        pitch_shift=pitch_shift,
        backend=backend,
        model_arch=model_arch,
        model_config_name=model_config_name,
        model_native_sample_rate=model_native_sample_rate,
    )
    return stems, settings


def separate_tree_for_eval(
    mixture_path: str,
    sample_rate: int,
    config: UpmixConfig,
) -> tuple[dict[str, np.ndarray], RunSettings]:
    """Run the production stem tree and return its prepared stem audio.

    The pipeline's public preparation method returns a stem summary while its
    plain stem store owns the actual arrays.  Evaluation runs therefore use a
    fresh, isolated store and disable every cache/input shortcut before
    reading the store back.
    """
    from upmixer.separation.stem_pipeline import StemUpmixPipeline
    from upmixer.separation.stem_store import PlainStemStore

    with TemporaryDirectory(prefix="upmixer_eval_stems_") as stem_output_dir:
        eval_config = replace(
            config,
            stem_cache_dir=None,
            stem_input_dir=None,
            stem_output_dir=stem_output_dir,
        )
        with StemUpmixPipeline(eval_config) as pipeline:
            result = pipeline.prepare_stems(mixture_path)
            loaded = PlainStemStore(stem_output_dir).load()
            stage_settings = tuple(pipeline.last_separation_settings)

        if loaded is None:
            raise RuntimeError(
                f"evaluation stem store is missing or unreadable: {stem_output_dir}"
            )
        all_stems, stored_sample_rate = loaded
        if stored_sample_rate != sample_rate:
            raise ValueError(
                f"stem store sample rate {stored_sample_rate} does not match "
                f"evaluation sample rate {sample_rate}"
            )
        requested_stems = frozenset(result.stems or ())
        stems = {
            key: audio
            for key, audio in all_stems.items()
            if key.split("@", 1)[0] in requested_stems
        }
        if not stems:
            requested = ", ".join(sorted(requested_stems)) or "none"
            raise RuntimeError(
                f"evaluation stem store has no requested stems (requested: {requested})"
            )

    return stems, RunSettings(
        model="production-tree",
        sample_rate=sample_rate,
        batch_size=config.stem_batch_size,
        segment_size=config.stem_segment_size,
        chunk_duration_s=config.stem_chunk_duration_s,
        overlap=config.stem_overlap,
        ensemble_algorithm=ENSEMBLE_ALGORITHM if config.stem_ensemble else None,
        ensemble_models=(MODEL_PRIMARY, MODEL_ENSEMBLE)
        if config.stem_ensemble
        else None,
        tta=config.stem_tta,
        pitch_shift=config.stem_pitch_shift,
        stage_settings=stage_settings,
    )


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
    coverage: list[CoverageRow] = []
    settings: RunSettings | None = None
    for item in corpus.items:
        unavailable_stems = tuple(item.unavailable_stems or ())
        if not item.stems and not unavailable_stems:
            raise ValueError(f"item {item.item_id or item.mixture} has no reference stems")
        if len(unavailable_stems) != len(set(unavailable_stems)):
            raise ValueError(f"item {item.item_id or item.mixture} has duplicate unavailable stems")
        overlap = set(item.stems) & set(unavailable_stems)
        if overlap:
            raise ValueError(
                f"item {item.item_id or item.mixture} marks stems as both scored and unavailable: "
                f"{', '.join(sorted(overlap))}"
            )
        estimate_stems, item_settings = separate_fn(item.mixture)
        if not isinstance(estimate_stems, dict):
            raise ValueError(f"item {item.item_id or item.mixture} returned invalid outputs")
        if item.stems and not estimate_stems:
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
        for stem_name in item.stems:
            coverage.append(
                CoverageRow(
                    stem=stem_name,
                    category=item.category,
                    status="scored",
                    recording_id=item.recording_id,
                    item_id=item.item_id,
                    split=item.split,
                )
            )
        for stem_name in unavailable_stems:
            coverage.append(
                CoverageRow(
                    stem=stem_name,
                    category=item.category,
                    status="unavailable",
                    recording_id=item.recording_id,
                    item_id=item.item_id,
                    split=item.split,
                )
            )
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
    return EvalReport(settings=settings, scores=scores, coverage=coverage)

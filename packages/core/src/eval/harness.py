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
from upmixer.eval.corpus import CorpusItem, ReferenceCorpus
from upmixer.eval.metrics import bleedless, fullness, sdr
from upmixer.eval.reference_targets import (
    estimate_components as _estimate_components,
    missing_estimates as _missing_estimates,
    sum_estimate_components as _sum_estimate_components,
    validate_audio as _validate_audio,
    validate_mapping as _validate_mapping,
)
from upmixer.eval.report import CoverageRow, EvalReport, StemScore
from upmixer.separation.separator import (
    DEFAULT_MODEL,
    SeparationSettings,
    StemSeparator,
)
from upmixer.separation.stem_plan import (
    DEFAULT_STEMS,
    ENSEMBLE_ALGORITHM,
    MODEL_ENSEMBLE,
    MODEL_PRIMARY,
    normalize_stems,
    resolve_separation_plan,
)

SeparateFn = Callable[[str], tuple[dict[str, np.ndarray], "RunSettings"]]


class EvaluationSkipped(RuntimeError):
    """Signal that an evaluation item was skipped intentionally."""


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
    device: str | None = None
    input_sample_rate: int | None = None
    separation_sample_rate: int | None = None
    output_sample_rate: int | None = None
    scoring_sample_rate: int | None = None
    plan: dict[str, object] | None = None
    stem_primary_remask: bool | None = None
    stem_drum_remask: bool | None = None
    stem_bleed_reduction: bool | None = None
    stem_ensemble: bool | None = None
    stem_silence_skip: bool | None = None
    stem_silence_threshold_db: float | None = None
    stem_silence_min_duration_s: float | None = None
    stem_silence_crossfade_ms: float | None = None
    stem_silence_pad_ms: float | None = None


@dataclass
class ItemRunSettings:
    """Effective settings recorded for one corpus item."""

    recording_id: str | None
    item_id: str | None
    split: str | None
    category: str
    settings: RunSettings


def _common_stage_setting(
    stage_settings: tuple[SeparationSettings, ...],
    name: str,
) -> object:
    """Return one effective value when an executed stage observed it."""
    if not stage_settings:
        return None
    values = tuple(getattr(stage, name, None) for stage in stage_settings)
    return values[0] if all(value == values[0] for value in values) else None


def _plan_context(
    config: UpmixConfig,
    stage_settings: tuple[SeparationSettings, ...],
) -> tuple[dict[str, object], bool]:
    """Serialize the resolved tree plan and mark stages seen in execution."""
    canonical = normalize_stems(config.stems) if config.stems else list(DEFAULT_STEMS)
    plan = resolve_separation_plan(canonical, config.stem_ensemble)
    observed_models = {stage.model for stage in stage_settings}
    tasks = [
        {
            "model": task.model,
            "input_source": task.input_source,
            "output_stems": sorted(task.output_stems),
            "keep_stems": sorted(task.keep_stems),
            "ensemble_models": list(task.ensemble_models),
            "ensemble_stems": sorted(task.ensemble_stems),
            "ensemble_algorithm": ENSEMBLE_ALGORITHM if task.ensemble_models else None,
            "executed": task.model in observed_models,
        }
        for task in plan.tasks
    ]
    return (
        {
            "requested_stems": sorted(plan.requested_stems),
            "tasks": tasks,
            "stems_hash": plan.stems_hash,
            "inference_hash": plan.inference_hash or None,
        },
        {MODEL_PRIMARY, MODEL_ENSEMBLE} <= observed_models,
    )


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
            ``StemSeparator``; the returned settings use the effective values
            observed after inference completes.
        stem_ensemble: Run the production fixed BS-Roformer-SW + SCNet
            primary-stem ensemble.
        tta: Forward test-time augmentation to the selected separator.
        pitch_shift: Forward the optional pitch-register rescue ratio.

    Returns:
        (stems, settings) — canonical stem name -> (n_samples, 2) float32
        array, and the effective settings observed during inference.
    """
    if stem_ensemble:
        if model != DEFAULT_MODEL:
            raise ValueError(
                "stem_ensemble uses the registered BS-Roformer-SW primary model"
            )
        return separate_tree_for_eval(
            mixture_path,
            sample_rate,
            UpmixConfig(
                output_sample_rate=sample_rate,
                stem_batch_size=batch_size,
                stem_segment_size=segment_size,
                stem_chunk_duration_s=chunk_duration_s,
                stem_overlap=overlap,
                stem_tta=tta,
                stem_pitch_shift=pitch_shift,
                stem_ensemble=True,
            ),
        )
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
        snapshot = getattr(separator, "run_settings", None)
    finally:
        separator.close()
    if snapshot is None:
        raise RuntimeError(
            "StemSeparator completed without a run-settings snapshot"
        )
    try:
        input_sample_rate = sf.info(mixture_path).samplerate
    except (OSError, RuntimeError):
        input_sample_rate = None
    settings = RunSettings(
        model=snapshot.model,
        sample_rate=snapshot.sample_rate,
        segment_size=snapshot.segment_size,
        overlap=snapshot.overlap,
        batch_size=snapshot.batch_size,
        chunk_duration_s=snapshot.chunk_duration_s,
        tta=snapshot.tta,
        pitch_shift=snapshot.pitch_shift,
        backend=snapshot.backend,
        model_arch=snapshot.model_arch,
        model_config_name=snapshot.model_config_name,
        model_native_sample_rate=snapshot.model_native_sample_rate,
        stage_settings=(snapshot,),
        device=snapshot.device,
        input_sample_rate=input_sample_rate,
        separation_sample_rate=snapshot.sample_rate,
        output_sample_rate=snapshot.sample_rate,
        scoring_sample_rate=sample_rate,
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

    plan, ensemble_observed = _plan_context(config, stage_settings)
    output_sample_rate = (
        getattr(result, "output_sample_rate", None) or stored_sample_rate
    )
    return stems, RunSettings(
        model="production-tree",
        sample_rate=stored_sample_rate,
        batch_size=_common_stage_setting(stage_settings, "batch_size"),
        segment_size=_common_stage_setting(stage_settings, "segment_size"),
        chunk_duration_s=_common_stage_setting(stage_settings, "chunk_duration_s"),
        overlap=_common_stage_setting(stage_settings, "overlap"),
        ensemble_algorithm=ENSEMBLE_ALGORITHM if ensemble_observed else None,
        ensemble_models=(MODEL_PRIMARY, MODEL_ENSEMBLE)
        if ensemble_observed
        else None,
        tta=_common_stage_setting(stage_settings, "tta"),
        pitch_shift=_common_stage_setting(stage_settings, "pitch_shift"),
        backend=_common_stage_setting(stage_settings, "backend"),
        model_arch=_common_stage_setting(stage_settings, "model_arch"),
        model_config_name=_common_stage_setting(stage_settings, "model_config_name"),
        model_native_sample_rate=_common_stage_setting(
            stage_settings, "model_native_sample_rate"
        ),
        stage_settings=stage_settings,
        device=_common_stage_setting(stage_settings, "device"),
        input_sample_rate=getattr(result, "input_sample_rate", None),
        separation_sample_rate=stored_sample_rate,
        output_sample_rate=output_sample_rate,
        scoring_sample_rate=sample_rate,
        plan=plan,
        stem_primary_remask=config.stem_primary_remask,
        stem_drum_remask=config.stem_drum_remask,
        stem_bleed_reduction=config.stem_bleed_reduction,
        stem_ensemble=ensemble_observed,
        stem_silence_skip=config.stem_silence_skip,
        stem_silence_threshold_db=config.stem_silence_threshold_db,
        stem_silence_min_duration_s=config.stem_silence_min_duration_s,
        stem_silence_crossfade_ms=config.stem_silence_crossfade_ms,
        stem_silence_pad_ms=config.stem_silence_pad_ms,
    )


def _failure_detail(exc: BaseException) -> str:
    detail = " ".join(str(exc).split()) or type(exc).__name__
    return detail if len(detail) <= 240 else f"{detail[:237]}..."


def _validate_item(item: CorpusItem) -> tuple[str, ...]:
    unavailable_stems = tuple(item.unavailable_stems or ())
    if not item.stems and not unavailable_stems:
        raise ValueError(f"item {item.item_id or item.mixture} has no reference stems")
    if len(unavailable_stems) != len(set(unavailable_stems)):
        raise ValueError(
            f"item {item.item_id or item.mixture} has duplicate unavailable stems"
        )
    overlap = set(item.stems) & set(unavailable_stems)
    if overlap:
        raise ValueError(
            f"item {item.item_id or item.mixture} marks stems as both scored and "
            f"unavailable: {', '.join(sorted(overlap))}"
        )
    _validate_mapping(item)
    return unavailable_stems


def _coverage_rows(
    item: CorpusItem,
    statuses: dict[str, str],
    details: dict[str, str],
    unavailable_stems: tuple[str, ...],
) -> list[CoverageRow]:
    rows = [
        CoverageRow(
            stem=stem_name,
            category=item.category,
            status=statuses[stem_name],
            recording_id=item.recording_id,
            item_id=item.item_id,
            split=item.split,
            detail=details.get(stem_name),
        )
        for stem_name in item.stems
    ]
    rows.extend(
        CoverageRow(
            stem=stem_name,
            category=item.category,
            status="unavailable",
            recording_id=item.recording_id,
            item_id=item.item_id,
            split=item.split,
        )
        for stem_name in unavailable_stems
    )
    return rows


def _score_stem(
    item: CorpusItem,
    stem_name: str,
    ref_path: str,
    estimate: np.ndarray,
    sample_rate: int,
) -> StemScore:
    reference, reference_rate = sf.read(ref_path, dtype="float32", always_2d=True)
    if reference_rate != sample_rate:
        raise ValueError(
            f"reference {ref_path} sample rate {reference_rate} does not "
            f"match evaluation sample rate {sample_rate}"
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


def evaluate_corpus(
    corpus: ReferenceCorpus,
    separate_fn: SeparateFn,
    sample_rate: int,
    *,
    protocol_id: str | None = None,
    code_revision: str | None = None,
    report_failures: bool = False,
) -> EvalReport:
    """Score a separation run over every item in a corpus.

    ``report_failures`` keeps expected-stem coverage when a separator or
    per-stem validation fails; the default remains fail-fast.
    """
    scores: list[StemScore] = []
    coverage: list[CoverageRow] = []
    settings_rows: list[ItemRunSettings] = []
    shared_settings: RunSettings | None = None
    settings_vary = False

    for item in corpus.items:
        unavailable_stems = _validate_item(item)
        statuses = {stem_name: "failed" for stem_name in item.stems}
        details: dict[str, str] = {}
        try:
            estimate_stems, item_settings = separate_fn(item.mixture)
            if not isinstance(estimate_stems, dict):
                raise ValueError(
                    "returned invalid outputs"
                    if report_failures
                    else f"item {item.item_id or item.mixture} returned invalid outputs"
                )
            if not report_failures and item.stems and not estimate_stems:
                raise ValueError(
                    f"item {item.item_id or item.mixture} returned empty outputs"
                )
            missing, missing_components = _missing_estimates(item, estimate_stems)
            if missing and not report_failures:
                if item.estimate_stems:
                    missing_detail = "; ".join(
                        f"{target}: {', '.join(missing_components[target])}"
                        for target in missing
                    )
                    raise ValueError(
                        f"item {item.item_id or item.mixture} missing required "
                        f"output stem(s) for reference target(s): {missing_detail}"
                    )
                raise ValueError(
                    f"item {item.item_id or item.mixture} missing required stem(s): "
                    f"{', '.join(missing)}"
                )
            for stem_name in missing:
                statuses[stem_name] = "absent"
                details[stem_name] = (
                    "missing from separator output"
                    if not item.estimate_stems
                    else "missing from separator output: "
                    + ", ".join(missing_components[stem_name])
                )
            if not isinstance(item_settings, RunSettings):
                raise ValueError("separation returned invalid RunSettings")
            if item_settings.sample_rate != sample_rate:
                raise ValueError(
                    f"RunSettings sample rate {item_settings.sample_rate} does not "
                    f"match evaluation sample rate {sample_rate}"
                )
            settings_rows.append(
                ItemRunSettings(
                    recording_id=item.recording_id,
                    item_id=item.item_id,
                    split=item.split,
                    category=item.category,
                    settings=item_settings,
                )
            )
            if shared_settings is None:
                shared_settings = item_settings
            elif item_settings != shared_settings:
                settings_vary = True
            for stem_name, ref_path in item.stems.items():
                if stem_name in missing:
                    continue
                try:
                    if item.estimate_stems:
                        estimate = _sum_estimate_components(
                            stem_name,
                            _estimate_components(item, stem_name),
                            estimate_stems,
                        )
                    else:
                        estimate = estimate_stems[stem_name]
                    score = _score_stem(
                        item,
                        stem_name,
                        ref_path,
                        estimate,
                        sample_rate,
                    )
                except EvaluationSkipped as exc:
                    if not report_failures:
                        raise
                    statuses[stem_name] = "skipped"
                    details[stem_name] = _failure_detail(exc)
                except Exception as exc:
                    if not report_failures:
                        raise
                    statuses[stem_name] = "failed"
                    details[stem_name] = _failure_detail(exc)
                else:
                    scores.append(score)
                    statuses[stem_name] = "scored"
        except EvaluationSkipped as exc:
            if not report_failures:
                raise
            statuses = {stem_name: "skipped" for stem_name in item.stems}
            detail = _failure_detail(exc)
            details = {stem_name: detail for stem_name in item.stems}
            settings_vary = True
        except Exception as exc:
            if not report_failures:
                raise
            detail = _failure_detail(exc)
            settings_vary = True
            for stem_name in item.stems:
                if statuses[stem_name] != "absent":
                    statuses[stem_name] = "failed"
                    details[stem_name] = detail

        if any(status != "scored" for status in statuses.values()):
            settings_vary = True
        coverage.extend(_coverage_rows(item, statuses, details, unavailable_stems))

    if not corpus.items:
        raise ValueError("corpus has no items to evaluate")
    return EvalReport(
        settings=None if settings_vary else shared_settings,
        scores=scores,
        coverage=coverage,
        item_settings=settings_rows,
        protocol_id=protocol_id,
        corpus_id=corpus.corpus_id,
        code_revision=code_revision,
    )

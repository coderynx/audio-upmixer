"""Evaluation stem retention boundary."""

from __future__ import annotations

import shutil
from pathlib import Path
from typing import Callable

import numpy as np
import soundfile as sf

from upmixer.eval.cascade import CascadeEvaluationResult, write_cascade_arms
from upmixer.eval.origins import OriginEvaluationResult
from upmixer.eval.corpus import ReferenceCorpus
from upmixer.eval.reference_targets import estimate_components
from upmixer.eval.types import RunSettings
from upmixer.execution import write_report
from upmixer.io.atomic import atomic_output_path

_STEM_INDEX_SCHEMA = 1
_RETAINED_SETTINGS_FIELDS = (
    "model",
    "model_native_sample_rate",
    "input_sample_rate",
    "separation_sample_rate",
    "output_sample_rate",
    "scoring_sample_rate",
    "rate_arm",
    "input_frame_count",
    "separation_frame_count",
    "output_frame_count",
    "resampler",
)


def _retained_filename(stem_name: str, used: set[str]) -> str:
    base = stem_name.replace("@", "__").replace("/", "__").replace("\\", "__") or "stem"
    filename = f"{base}.wav"
    suffix = 2
    while filename in used:
        filename = f"{base}__{suffix}.wav"
        suffix += 1
    used.add(filename)
    return filename


def _retaining_separator(
    separate_fn: Callable,
    corpus: ReferenceCorpus,
    output_dir: Path,
    evaluation_sample_rate: int,
) -> Callable:
    """Persist successful separator returns while evaluation advances in order."""
    stems_dir = output_dir / "stems"
    index_path = stems_dir / "index.json"
    stems_dir.mkdir(parents=True, exist_ok=True)
    entries: list[dict[str, object]] = []
    write_report(index_path, {"schema_version": _STEM_INDEX_SCHEMA, "items": entries})
    next_index = 0

    def separate(mixture_path: str):
        nonlocal next_index
        if next_index >= len(corpus.items):
            raise RuntimeError("separator called more times than corpus items")
        item_index = next_index
        item = corpus.items[item_index]
        next_index += 1
        if str(mixture_path) != item.mixture:
            raise ValueError("separator mixture path does not match corpus item")
        result = separate_fn(mixture_path)
        if isinstance(result, OriginEvaluationResult):
            origin_result = result
            stems, settings = result.stems, result.settings
            cascade_result = None
        elif isinstance(result, CascadeEvaluationResult):
            origin_result = None
            cascade_result = result
            stems, settings = result.stems, result.settings
        else:
            origin_result = None
            cascade_result = None
            stems, settings = result
        if not isinstance(stems, dict) or not stems:
            return result
        if not isinstance(settings, RunSettings):
            return result
        sample_rate = settings.sample_rate
        if (
            isinstance(sample_rate, bool)
            or not isinstance(sample_rate, int)
            or sample_rate < 1
        ):
            raise ValueError("retained stems require a positive settings sample rate")
        if sample_rate != evaluation_sample_rate:
            return result
        required_stems = {
            component
            for target in item.stems
            for component in estimate_components(item, target)
        }
        if required_stems and not required_stems.issubset(stems):
            return result
        reference_info = None
        if item.stems:
            try:
                first_reference = item.stems[sorted(item.stems)[0]]
                reference_info = sf.info(first_reference)
            except (OSError, RuntimeError):
                return result
        retained: dict[str, np.ndarray] = {}
        try:
            for stem_name, value in stems.items():
                if not isinstance(stem_name, str):
                    return result
                raw_audio = np.asarray(value)
                if (
                    raw_audio.ndim != 2
                    or not raw_audio.size
                    or not raw_audio.shape[0]
                    or not raw_audio.shape[1]
                    or not np.issubdtype(raw_audio.dtype, np.number)
                ):
                    return result
                audio = np.asarray(raw_audio, dtype=np.float32)
                if not np.all(np.isfinite(audio)):
                    return result
                if reference_info is not None and (
                    reference_info.samplerate != sample_rate
                    or reference_info.frames != audio.shape[0]
                    or reference_info.channels != audio.shape[1]
                ):
                    return result
                if stem_name in item.stems:
                    info = sf.info(item.stems[stem_name])
                    if (
                        info.samplerate != sample_rate
                        or info.frames != audio.shape[0]
                        or info.channels != audio.shape[1]
                    ):
                        return result
                retained[stem_name] = audio
        except (OSError, RuntimeError, TypeError, ValueError):
            return result

        retained_views: list[tuple[object, dict[str, np.ndarray]]] = []
        if origin_result is not None:
            for view in origin_result.view_outputs:
                view_stems: dict[str, np.ndarray] = {}
                for stem_name, value in view.stems.items():
                    audio = np.asarray(value, dtype=np.float32)
                    if (
                        not isinstance(stem_name, str)
                        or audio.ndim != 2
                        or not audio.size
                        or not np.all(np.isfinite(audio))
                        or set(view.stems) != set(retained)
                        or (
                            reference_info is not None
                            and (
                                reference_info.samplerate != sample_rate
                                or reference_info.frames != audio.shape[0]
                                or reference_info.channels != audio.shape[1]
                            )
                        )
                    ):
                        return result
                    view_stems[stem_name] = audio
                retained_views.append((view, view_stems))

        retained_arms: list[tuple[object, dict[str, np.ndarray]]] = []
        if cascade_result is not None:
            arm_ids: set[str] = set()
            for arm in cascade_result.arms:
                if not isinstance(arm.arm_id, str) or arm.arm_id in arm_ids:
                    return result
                arm_ids.add(arm.arm_id)
                if not isinstance(arm.stems, dict) or set(arm.stems) != set(retained):
                    return result
                arm_stems: dict[str, np.ndarray] = {}
                for stem_name, value in arm.stems.items():
                    audio = np.asarray(value, dtype=np.float32)
                    if (
                        not isinstance(stem_name, str)
                        or audio.ndim != 2
                        or not audio.size
                        or not np.all(np.isfinite(audio))
                        or set(arm.stems) != set(retained)
                        or audio.shape != next(iter(retained.values())).shape
                    ):
                        return result
                    arm_stems[stem_name] = audio
                retained_arms.append((arm, arm_stems))

        item_dir = stems_dir / f"{item_index:04d}"
        paths: dict[str, str] = {}
        used_filenames: set[str] = set()
        try:
            item_dir.mkdir(parents=True, exist_ok=False)
            for stem_name in sorted(stems, key=str):
                filename = _retained_filename(str(stem_name), used_filenames)
                destination = item_dir / filename
                with atomic_output_path(destination) as temporary:
                    sf.write(
                        str(temporary),
                        retained[stem_name],
                        sample_rate,
                        subtype="FLOAT",
                    )
                paths[str(stem_name)] = destination.relative_to(output_dir).as_posix()
            view_entries: list[dict[str, object]] = []
            for view, view_stems in retained_views:
                view_dir = item_dir / "views" / str(view.origin_samples)
                view_dir.mkdir(parents=True, exist_ok=False)
                view_paths: dict[str, str] = {}
                used_view_filenames: set[str] = set()
                for stem_name in sorted(view_stems, key=str):
                    filename = _retained_filename(stem_name, used_view_filenames)
                    destination = view_dir / filename
                    with atomic_output_path(destination) as temporary:
                        sf.write(
                            str(temporary),
                            view_stems[stem_name],
                            sample_rate,
                            subtype="FLOAT",
                        )
                    view_paths[stem_name] = destination.relative_to(
                        output_dir
                    ).as_posix()
                view.output_paths = view_paths
                view_entries.append(
                    {"origin_samples": view.origin_samples, "stems": view_paths}
                )
            arm_entries = write_cascade_arms(
                retained_arms,
                item_dir=item_dir,
                output_dir=output_dir,
                sample_rate=sample_rate,
                filename_for=_retained_filename,
            )
            entry = {
                "index": item_index,
                "recording_id": item.recording_id,
                "item_id": item.item_id,
                "split": item.split,
                "category": item.category,
                "sample_rate": sample_rate,
                "stems": paths,
            }
            if view_entries:
                entry["view_outputs"] = view_entries
            if arm_entries:
                entry["cascade_arms"] = arm_entries
            for field in _RETAINED_SETTINGS_FIELDS:
                value = getattr(settings, field, None)
                if value is not None:
                    entry[field] = value
            write_report(
                index_path,
                {"schema_version": _STEM_INDEX_SCHEMA, "items": [*entries, entry]},
            )
        except BaseException:
            shutil.rmtree(item_dir, ignore_errors=True)
            raise
        entries.append(entry)
        return result

    return separate

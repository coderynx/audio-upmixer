"""Reference-target output composition for evaluation."""

from __future__ import annotations

from typing import TYPE_CHECKING

import numpy as np

if TYPE_CHECKING:
    from upmixer.eval.corpus import CorpusItem


def validate_audio(array: np.ndarray, label: str) -> None:
    if not isinstance(array, np.ndarray) or array.ndim != 2:
        raise ValueError(f"{label} must be a 2D array (frames, channels)")
    if not array.size or not array.shape[0] or not array.shape[1]:
        raise ValueError(f"{label} must not be empty")
    if not np.issubdtype(array.dtype, np.number) or not np.all(np.isfinite(array)):
        raise ValueError(f"{label} must contain finite numeric values")


def estimate_components(item: "CorpusItem", target: str) -> tuple[str, ...]:
    return tuple((item.estimate_stems or {}).get(target, (target,)))


def validate_mapping(item: "CorpusItem") -> None:
    mapping = item.estimate_stems or {}
    label = f"item {item.item_id or item.mixture}"
    if not isinstance(mapping, dict):
        raise ValueError(f"{label} estimate_stems must be an object")
    if not all(isinstance(target, str) for target in mapping):
        raise ValueError(f"{label} estimate_stems targets must be strings")
    known_targets = set(item.stems) | set(item.unavailable_stems or ())
    unknown_targets = set(mapping) - known_targets
    if unknown_targets:
        raise ValueError(
            f"{label} estimate_stems has unknown reference target(s): "
            f"{', '.join(sorted(unknown_targets))}"
        )
    for components in mapping.values():
        if (
            not isinstance(components, (list, tuple))
            or not components
            or not all(
                isinstance(component, str) and component.strip()
                for component in components
            )
            or len(set(components)) != len(components)
        ):
            raise ValueError(f"{label} has invalid estimate_stems")


def missing_estimates(
    item: "CorpusItem", estimate_stems: dict[str, np.ndarray]
) -> tuple[list[str], dict[str, tuple[str, ...]]]:
    if not item.estimate_stems:
        missing = sorted(set(item.stems) - set(estimate_stems))
        return missing, {stem_name: (stem_name,) for stem_name in missing}
    missing_by_target = {
        target: tuple(
            component
            for component in estimate_components(item, target)
            if component not in estimate_stems
        )
        for target in item.stems
    }
    missing_by_target = {
        target: values for target, values in missing_by_target.items() if values
    }
    return sorted(missing_by_target), missing_by_target


def sum_estimate_components(
    target: str,
    components: tuple[str, ...],
    estimate_stems: dict[str, np.ndarray],
) -> np.ndarray:
    arrays: list[np.ndarray] = []
    for component in components:
        label = f"estimate component {component} for {target}"
        component_audio = estimate_stems[component]
        validate_audio(component_audio, label)
        with np.errstate(over="ignore", invalid="ignore"):
            component_audio = np.asarray(component_audio, dtype=np.float32)
        validate_audio(component_audio, label)
        if arrays and component_audio.shape != arrays[0].shape:
            kind = (
                "channel" if component_audio.shape[1] != arrays[0].shape[1] else "frame"
            )
            axis = 1 if kind == "channel" else 0
            raise ValueError(
                f"{kind} count mismatch for mapped estimate {target}: component "
                f"{component} has {component_audio.shape[axis]}, expected "
                f"{arrays[0].shape[axis]}"
            )
        arrays.append(component_audio)

    total = np.zeros(arrays[0].shape, dtype=np.float32)
    with np.errstate(over="ignore", invalid="ignore"):
        for component_audio in arrays:
            total += component_audio
    validate_audio(total, f"estimate {target}")
    return total

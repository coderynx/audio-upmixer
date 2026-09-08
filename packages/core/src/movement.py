"""Public movement settings and persisted feature validation.

Movement decisions and routing are compiled by the shared DSP crate.  This
module deliberately contains only the boundary contract used by manifests,
the CLI, and prepared-stem sidecars; it does not analyse audio or compile a
schedule.
"""

from __future__ import annotations

import json
import math
from collections.abc import Mapping
from dataclasses import dataclass
from typing import Any

import numpy as np

MOVEMENT_FEATURE_VERSION = 1
MOVEMENT_SCHEDULE_VERSION = 1
MOVEMENT_FEATURE_WINDOW_US = 10_000
MOVEMENT_GRID_US = 20_000
MOVEMENT_INTERPOLATION_US = 5_208
MOVEMENT_ROLES = ("auto", "supporting", "featured")

# Initial preset depths are deliberately data, so the same values can be
# passed to the shared compiler by each caller without another policy copy.
MOVEMENT_PRESET_DEPTHS: dict[str, float] = {
    "intimate": 0.15,
    "balanced": 0.20,
    "stage": 0.25,
    "wide": 0.30,
    "live": 0.35,
    "immersive": 0.40,
}

_MOVEMENT_AUTO_STEMS = frozenset({"Guitar", "Piano"})
_MOVEMENT_SUPPORTING_STEMS = frozenset({
    "Backing Vocals", "Crowd", "Toms", "Hi-Hat", "Ride", "Crash",
})
_MOVEMENT_ELIGIBLE_STEMS = _MOVEMENT_AUTO_STEMS | _MOVEMENT_SUPPORTING_STEMS

MOVEMENT_TUNING_DEFAULTS: dict[str, float] = {
    "activity_floor_db": -65.0,
    "activity_enter_db": 6.0,
    "activity_leave_db": 3.0,
    "activity_dwell_ms": 80.0,
    "activity_exit_dwell_ms": 250.0,
    "focus_prominence_db": 6.0,
    "focus_share": 0.85,
    "focus_qualification_ms": 750.0,
    "focus_attack_ms": 750.0,
    "focus_release_ms": 1_500.0,
    "vocal_hold_ms": 1_200.0,
    "winner_hold_ms": 1_000.0,
    "challenger_db": 3.0,
    "supporting_span_db": 24.0,
    "percussion_rise_db": 6.0,
    "percussion_gate_db": 6.0,
    "percussion_retrigger_ms": 120.0,
    "percussion_attack_ms": 40.0,
    "percussion_return_ms": 300.0,
    "toms_return_ms": 500.0,
    "crash_return_ms": 1_000.0,
}


@dataclass(frozen=True)
class StemMovementSettings:
    """Validated settings for one stem movement entry."""

    enabled: bool = False
    role: str = "auto"
    depth: float = 0.0
    response: float = 1.0
    sensitivity: float = 0.5
    start_s: float = 0.0
    end_s: float | None = None

    def as_dict(self) -> dict[str, Any]:
        return {
            "enabled": self.enabled,
            "role": self.role,
            "depth": self.depth,
            "response": self.response,
            "sensitivity": self.sensitivity,
            "start_s": self.start_s,
            "end_s": self.end_s,
        }


def _number(value: object, name: str, low: float, high: float) -> float:
    if isinstance(value, bool) or not isinstance(value, (int, float)):
        raise ValueError(f"{name} must be a finite number in {low}..{high}")
    result = float(value)
    if not math.isfinite(result) or not low <= result <= high:
        raise ValueError(f"{name} must be a finite number in {low}..{high}")
    return result


def normalize_stem_movement(value: Mapping[str, Any]) -> dict[str, dict[str, Any]]:
    """Validate and fill defaults for a stem movement mapping."""
    result: dict[str, dict[str, Any]] = {}
    for stem_key, raw in value.items():
        if not isinstance(stem_key, str) or not stem_key.strip():
            raise ValueError("stem movement keys must be non-empty strings")
        if not isinstance(raw, Mapping):
            raise ValueError(f"entry for '{stem_key}' must be a mapping")
        unknown = set(raw) - {
            "enabled", "role", "depth", "response", "sensitivity", "start_s", "end_s",
        }
        if unknown:
            names = ", ".join(sorted(map(str, unknown)))
            raise ValueError(f"entry for '{stem_key}' has unknown field(s): {names}")
        enabled = raw.get("enabled", False)
        if not isinstance(enabled, bool):
            raise ValueError(f"{stem_key}.enabled must be a boolean")
        role = raw.get("role", "auto")
        if not isinstance(role, str) or role not in MOVEMENT_ROLES:
            raise ValueError(f"{stem_key}.role must be one of {MOVEMENT_ROLES}")
        depth = _number(raw.get("depth", 0.0), f"{stem_key}.depth", 0.0, 1.0)
        response = _number(raw.get("response", 1.0), f"{stem_key}.response", 0.5, 2.0)
        sensitivity = _number(raw.get("sensitivity", 0.5), f"{stem_key}.sensitivity", 0.0, 1.0)
        start_s = _number(raw.get("start_s", 0.0), f"{stem_key}.start_s", 0.0, math.inf)
        end_raw = raw.get("end_s")
        end_s = None if end_raw is None else _number(end_raw, f"{stem_key}.end_s", 0.0, math.inf)
        if end_s is not None and not start_s < end_s:
            raise ValueError(f"{stem_key}.start_s must be less than end_s")
        result[stem_key] = StemMovementSettings(
            enabled=enabled,
            role=role,
            depth=depth,
            response=response,
            sensitivity=sensitivity,
            start_s=start_s,
            end_s=end_s,
        ).as_dict()
    return result


def validate_stem_movement(value: object) -> dict[str, dict[str, Any]]:
    """Return normalized settings or raise ``ValueError`` at a boundary."""
    if not isinstance(value, Mapping):
        raise ValueError("stem movement must be a mapping")
    return normalize_stem_movement(value)


def resolve_stem_movement(
    value: Mapping[str, Any] | None, stem_key: str,
) -> StemMovementSettings:
    """Resolve an exact stem key before its base name, with defaults."""
    entries = validate_stem_movement(value or {})
    raw = entries.get(stem_key)
    if raw is None:
        raw = entries.get(stem_key.split("@", 1)[0])
    if raw is None:
        return StemMovementSettings()
    return StemMovementSettings(**raw)


def movement_depth_for_preset(preset: str) -> float:
    """Return the configured initial depth for a named movement preset."""
    try:
        return MOVEMENT_PRESET_DEPTHS[preset]
    except KeyError:
        raise ValueError(
            f"Unknown movement preset '{preset}'. Valid: {tuple(MOVEMENT_PRESET_DEPTHS)}"
        ) from None


def movement_settings_for_preset(
    preset: str, stems: list[str],
) -> dict[str, dict[str, Any]]:
    """Return persisted movement defaults for one routing preset.

    Zone-tagged stems retain their source placement until a producer enables
    movement explicitly.  Unknown and anchored stems are still represented so
    a project has one complete, editable map and the shared compiler receives
    explicit off/depth-zero state.
    """
    depth = movement_depth_for_preset(preset)
    settings: dict[str, dict[str, Any]] = {}
    for stem_key in stems:
        base = stem_key.split("@", 1)[0]
        role = (
            "auto" if base in _MOVEMENT_AUTO_STEMS
            else "supporting" if base in _MOVEMENT_SUPPORTING_STEMS
            else "auto"
        )
        enabled = base in _MOVEMENT_ELIGIBLE_STEMS and "@" not in stem_key
        settings[stem_key] = {
            "enabled": enabled,
            "role": role,
            "depth": depth if enabled else 0.0,
            "response": 1.0,
            "sensitivity": 0.5,
            "start_s": 0.0,
            "end_s": None,
        }
    return settings


def movement_tuning(overrides: Mapping[str, Any] | None = None) -> dict[str, float]:
    """Return core-owned tuning merged with finite manifest overrides."""
    result = dict(MOVEMENT_TUNING_DEFAULTS)
    if overrides is None:
        return result
    if not isinstance(overrides, Mapping):
        raise ValueError("movement tuning must be a mapping")
    unknown = set(overrides) - set(result)
    if unknown:
        raise ValueError(f"unknown movement tuning field(s): {', '.join(sorted(map(str, unknown)))}")
    for key, value in overrides.items():
        if isinstance(value, bool) or not isinstance(value, (int, float)) or not math.isfinite(float(value)):
            raise ValueError(f"movement tuning '{key}' must be a finite number")
        result[key] = float(value)
    bounded = {
        "activity_floor_db": (-120.0, 0.0),
        "activity_enter_db": (0.0, 60.0),
        "activity_leave_db": (0.0, 60.0),
        "focus_prominence_db": (0.0, 60.0),
        "percussion_rise_db": (0.0, 60.0),
        "percussion_gate_db": (0.0, 60.0),
    }
    for key, (low, high) in bounded.items():
        if not low <= result[key] <= high:
            raise ValueError(f"movement tuning '{key}' must be in {low}..{high}")
    if not 0.0 <= result["focus_share"] <= 1.0:
        raise ValueError("movement tuning 'focus_share' must be in 0..1")
    nonnegative = {
        "focus_qualification_ms", "vocal_hold_ms", "winner_hold_ms",
        "challenger_db", "percussion_retrigger_ms", "activity_dwell_ms",
        "activity_exit_dwell_ms",
    }
    positive = {
        "focus_attack_ms", "focus_release_ms", "supporting_span_db",
        "percussion_attack_ms", "percussion_return_ms", "toms_return_ms",
        "crash_return_ms",
    }
    if any(result[key] < 0.0 for key in nonnegative):
        raise ValueError("movement tuning values cannot be negative")
    if any(result[key] <= 0.0 for key in positive):
        raise ValueError("movement tuning transition values must be greater than zero")
    if result["activity_enter_db"] < result["activity_leave_db"]:
        raise ValueError("movement activity entry threshold must be at least its leave threshold")
    return result


def validate_feature_sidecar(
    value: object, stem_keys: list[str] | tuple[str, ...],
) -> dict[str, Any]:
    """Validate a generated sidecar through the shared Rust movement module."""
    if not isinstance(value, Mapping):
        raise ValueError("movement feature sidecar must be a mapping")
    import upmixer_dsp

    encoded = json.dumps(value, separators=(",", ":"), allow_nan=False)
    try:
        upmixer_dsp.validate_movement_sidecar(encoded, list(stem_keys))
    except (TypeError, ValueError) as exc:
        raise ValueError(str(exc)) from exc
    return dict(value)


def validate_feature_sidecar_for_stems(
    value: object, stems: Mapping[str, Any], sample_rate: int,
) -> dict[str, Any]:
    """Validate a sidecar against the prepared PCM it describes.

    The shared validator checks the serialized contract and stem identities;
    this boundary check ties it to the actual prepared arrays so stale data
    cannot survive a rate conversion or stem rewrite.
    """
    result = validate_feature_sidecar(value, list(stems))
    if result.get("sample_rate") != sample_rate:
        raise ValueError("movement sidecar rate does not match prepared stems")
    frame_count = max((len(audio) for audio in stems.values()), default=0)
    if result.get("frame_count") != frame_count:
        raise ValueError("movement sidecar frame count does not match prepared stems")
    entries = {entry.get("stem_key"): entry for entry in result.get("stems", ())}
    for stem_key, audio in stems.items():
        features = entries[stem_key].get("features", {})
        if features.get("sample_rate") != sample_rate:
            raise ValueError(f"movement sidecar rate does not match stem '{stem_key}'")
        if features.get("frame_count") != len(audio):
            raise ValueError(f"movement sidecar frame count does not match stem '{stem_key}'")
    return result


def extract_feature_sidecar(
    stems: Mapping[str, Any], sample_rate: int,
) -> dict[str, Any]:
    """Extract and validate one sidecar from prepared float32 PCM."""
    entries: list[dict[str, Any]] = []
    for stem_key, audio in stems.items():
        array = np.asarray(audio, dtype=np.float32)
        if array.ndim == 1:
            left, right = np.ascontiguousarray(array), None
        elif array.ndim == 2 and array.shape[1] in (1, 2):
            left = np.ascontiguousarray(array[:, 0])
            right = (
                np.ascontiguousarray(array[:, 1])
                if array.shape[1] > 1 else None
            )
        else:
            raise ValueError(f"Prepared stem '{stem_key}' must be mono or stereo")
        entries.append({
            "stem_key": stem_key,
            "features": extract_canonical_features(left, right, sample_rate),
        })
    frame_count = max((len(audio) for audio in stems.values()), default=0)
    sidecar = {
        "version": MOVEMENT_FEATURE_VERSION,
        "sample_rate": sample_rate,
        "frame_count": frame_count,
        "stems": entries,
    }
    return validate_feature_sidecar_for_stems(sidecar, stems, sample_rate)


def extract_canonical_features(left: Any, right: Any | None, sample_rate: int) -> dict[str, Any]:
    """Extract canonical features using the shared PyO3 binding."""
    import upmixer_dsp

    encoded = upmixer_dsp.extract_movement_features_json(left, right, sample_rate)
    return json.loads(encoded)


def compile_movement_schedule(request: Mapping[str, Any]) -> dict[str, Any]:
    """Compile one immutable movement schedule using the shared DSP module."""
    import upmixer_dsp

    encoded = upmixer_dsp.compile_movement_schedule(
        json.dumps(request, separators=(",", ":"), allow_nan=False)
    )
    return json.loads(encoded)

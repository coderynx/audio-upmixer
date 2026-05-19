"""YAML/JSON manifest files for defining upmix jobs.

A manifest file lets all CLI parameters live in a single file, making
it easy to version-control complex upmix jobs and run them reproducibly.

Supported formats
-----------------
* YAML  (``.yaml``, ``.yml``) — requires ``pyyaml``: ``pip install pyyaml``
* JSON  (``.json``) — no extra dependency

Key naming
----------
Manifest keys use the same names as the CLI flags (without the leading
``--``), with hyphens replaced by underscores.  For example:

* ``--output-sample-rate 48000``  →  ``output_sample_rate: 48000``
* ``--loudness-target -18.0``     →  ``loudness_target: -18.0``

Priority order
--------------
CLI flags > manifest values > profile defaults > UpmixConfig defaults.

Example (YAML)::

    input:   stereo.flac
    output:  atmos.adm.bwf
    format:  7.1.2
    mode:    stem
    profile: atmos-music

    stem_model: BS-Roformer-SW.ckpt

    # Override specific profile values
    loudness_target: -18.0
    preview: true
    preview_duration: 30.0

Job keys (``input``, ``output``, ``mode``, ``input_format``,
``stem_model``, ``stem_model_dir``) are returned separately from
:func:`parse_manifest` so the pipeline layer can use them directly.
"""
from __future__ import annotations

import json
import logging
from pathlib import Path
from typing import Any

from upmixer.config import UpmixConfig
from upmixer.profiles import PROFILES

_log = logging.getLogger("upmixer")

# ── Field mapping ──────────────────────────────────────────────────────────────
# manifest key → (UpmixConfig attribute name, Python type for coercion)
# Only non-None values in the manifest are applied; null / omitted keys are
# treated as "not specified" and leave the config default intact.

_FIELD_MAP: dict[str, tuple[str, type]] = {
    # Output format
    "format":                     ("output_format",          str),
    "output_type":                ("output_type",            str),
    "output_subtype":             ("output_subtype",         str),
    "output_sample_rate":         ("output_sample_rate",     int),
    # Channel routing gains
    "center_gain":                ("center_gain",            float),
    "surround_gain":              ("surround_gain",          float),
    "back_gain":                  ("back_gain",              float),
    "height_gain":                ("height_gain",            float),
    "lfe_gain":                   ("lfe_gain",               float),
    # LFE
    "lfe_cutoff":                 ("lfe_cutoff_hz",          float),
    # Center extraction (realtime mode)
    "center_extraction_gain":     ("center_extraction_gain", float),
    "center_attenuation":         ("center_attenuation",     float),
    # Content-aware mixing
    "content_mix_strength":       ("content_mix_strength",   float),
    # Height EQ
    "height_low_rolloff_gain":    ("height_low_rolloff_gain",float),
    "height_high_shelf_gain":     ("height_high_shelf_gain", float),
    # STFT / processing
    "fft_size":                   ("fft_size",               int),
    "block_size":                 ("block_size",             int),
    # Energy normalization (mixing phase)
    "normalize_output":           ("normalize_output",       bool),
    # Mastering — loudness
    "loudness_normalize":         ("loudness_normalize",     bool),
    "loudness_target":            ("loudness_target_lkfs",   float),
    "loudness_max_tp":            ("loudness_max_tp",        float),
    # Mastering — EQ shaping (flat keys; also accessible via mastering: section)
    "mastering_eq_profile":       ("mastering_eq_profile",   str),
    "mastering_eq_strength":      ("mastering_eq_strength",  float),
    # Mastering — bus compressor
    "mastering_comp_profile":     ("mastering_comp_profile",      str),
    "mastering_comp_threshold_db":("mastering_comp_threshold_db", float),
    "mastering_comp_ratio":       ("mastering_comp_ratio",        float),
    "mastering_comp_attack_ms":   ("mastering_comp_attack_ms",    float),
    "mastering_comp_release_ms":  ("mastering_comp_release_ms",   float),
    "mastering_comp_knee_db":     ("mastering_comp_knee_db",      float),
    "mastering_comp_makeup_db":   ("mastering_comp_makeup_db",    float),
    # Mastering — bass control
    "mastering_bass_profile":        ("mastering_bass_profile",        str),
    "mastering_bass_sub_gain_db":    ("mastering_bass_sub_gain_db",    float),
    "mastering_bass_mid_gain_db":    ("mastering_bass_mid_gain_db",    float),
    "mastering_bass_mono_cutoff_hz": ("mastering_bass_mono_cutoff_hz", float),
    "mastering_bass_excite":         ("mastering_bass_excite",         bool),
    "mastering_bass_lfe_gain_db":    ("mastering_bass_lfe_gain_db",    float),
    # Mastering — EQ from reference (Match EQ)
    "mastering_eq_reference":        ("mastering_eq_reference",        str),
    # Mixing — stem rebalance (stem pipeline only)
    "stem_rebalance":                ("stem_rebalance",                dict),
    # Mixing — per-stem EQ (stem pipeline only)
    "stem_eq_profiles":              ("stem_eq_profiles",              dict),
    # Downmix
    "downmix_output":             ("downmix_output_path",    str),
    "downmix_surround_coeff":     ("surround_downmix_coeff", float),
    # Preview
    "preview":                    ("preview",                bool),
    "preview_duration":           ("preview_duration_s",     float),
    "preview_start":              ("preview_start_s",        float),
}

# ── Nested mastering: section ─────────────────────────────────────────────────
# Maps sub-keys inside a ``mastering:`` YAML block to the flat manifest keys
# that feed into _FIELD_MAP above.  This lets users write either:
#
#   mastering_eq_profile: spatial-air        # flat form
#
# or the structured form:
#
#   mastering:
#     eq_profile: spatial-air
#     loudness_normalize: true
#
# Both forms produce identical UpmixConfig state.

_MASTERING_KEY_MAP: dict[str, str] = {
    # EQ
    "eq_profile":       "mastering_eq_profile",
    "eq_strength":      "mastering_eq_strength",
    "eq_reference":     "mastering_eq_reference",
    # Compressor
    "comp_profile":     "mastering_comp_profile",
    "comp_threshold":   "mastering_comp_threshold_db",
    "comp_ratio":       "mastering_comp_ratio",
    "comp_attack":      "mastering_comp_attack_ms",
    "comp_release":     "mastering_comp_release_ms",
    "comp_knee":        "mastering_comp_knee_db",
    "comp_makeup":      "mastering_comp_makeup_db",
    # Bass control
    "bass_profile":     "mastering_bass_profile",
    "bass_sub_gain":    "mastering_bass_sub_gain_db",
    "bass_mid_gain":    "mastering_bass_mid_gain_db",
    "bass_mono_cutoff": "mastering_bass_mono_cutoff_hz",
    "bass_excite":      "mastering_bass_excite",
    "bass_lfe_gain":    "mastering_bass_lfe_gain_db",
    # Loudness (re-uses existing flat keys)
    "loudness_normalize": "loudness_normalize",
    "loudness_target":    "loudness_target",
    "loudness_max_tp":    "loudness_max_tp",
}

# ── Nested mixing: section ────────────────────────────────────────────────────
# Mirrors mastering: section but for mixing-phase params.
# Usage (YAML):
#
#   mixing:
#     stem_rebalance:
#       Vocals: +2.0
#       Drums: -1.0
#     stem_eq:
#       Vocals: vocal-presence
#       Bass: bass-warmth

_MIXING_KEY_MAP: dict[str, str] = {
    "stem_rebalance": "stem_rebalance",    # dict value passes through as-is
    "stem_eq":        "stem_eq_profiles",  # renamed for config
}


def _expand_nested_sections(data: dict) -> dict:
    """Expand ``mastering:`` and ``mixing:`` sub-dicts into flat manifest keys.

    If *data* contains a ``"mastering"`` key whose value is a mapping, its
    sub-keys are translated via :data:`_MASTERING_KEY_MAP` and injected into
    the top-level dict.  Similarly for a ``"mixing"`` key via
    :data:`_MIXING_KEY_MAP`.  Unknown sub-keys are passed through unchanged
    (a warning will be emitted by the caller for unrecognised manifest keys).

    The original ``"mastering"`` / ``"mixing"`` keys are removed.  Existing
    flat keys take priority — nested values do **not** overwrite them.

    Args:
        data: Original manifest dict.

    Returns:
        Expanded flat dict.  A copy is made when expansion occurs; the
        original dict is returned unchanged when neither section is present.
    """
    has_mastering = "mastering" in data and isinstance(data.get("mastering"), dict)
    has_mixing    = "mixing"    in data and isinstance(data.get("mixing"),    dict)

    if not has_mastering and not has_mixing:
        return data

    skip = set()
    if has_mastering:
        skip.add("mastering")
    if has_mixing:
        skip.add("mixing")

    expanded = {k: v for k, v in data.items() if k not in skip}

    if has_mastering:
        for sub_key, value in data["mastering"].items():
            flat_key = _MASTERING_KEY_MAP.get(sub_key, f"mastering_{sub_key}")
            if flat_key not in expanded:
                expanded[flat_key] = value

    if has_mixing:
        for sub_key, value in data["mixing"].items():
            flat_key = _MIXING_KEY_MAP.get(sub_key, sub_key)
            if flat_key not in expanded:
                expanded[flat_key] = value

    return expanded

# Keys handled at the pipeline / CLI level, not mapped into UpmixConfig.
_JOB_KEYS: frozenset[str] = frozenset({
    "input",
    "output",
    "mode",
    "input_format",
    "stem_model",
    "stem_model_dir",
    "profile",
})


# ── Loader ─────────────────────────────────────────────────────────────────────

def load_manifest(path: str | Path) -> dict[str, Any]:
    """Load a YAML or JSON manifest file and return it as a plain dict.

    Args:
        path: Path to a ``.yaml``, ``.yml``, or ``.json`` file.

    Returns:
        Dict of manifest key/value pairs.  The dict is always non-None; an
        empty manifest file returns ``{}``.

    Raises:
        FileNotFoundError: if *path* does not exist.
        ImportError:       if a YAML file is given but PyYAML is not installed.
        ValueError:        if the file extension is not recognised.
        json.JSONDecodeError / yaml.YAMLError: on parse failure.
    """
    path = Path(path)
    if not path.exists():
        raise FileNotFoundError(f"Manifest file not found: {path}")

    suffix = path.suffix.lower()
    text = path.read_text(encoding="utf-8")

    if suffix in (".yaml", ".yml"):
        try:
            import yaml  # type: ignore[import-untyped]
        except ImportError as exc:
            raise ImportError(
                "PyYAML is required to load YAML manifest files. "
                "Install it with: pip install pyyaml"
            ) from exc
        data = yaml.safe_load(text)
    elif suffix == ".json":
        data = json.loads(text)
    else:
        raise ValueError(
            f"Unrecognised manifest extension '{suffix}'. "
            "Use .yaml, .yml, or .json."
        )

    return data or {}


# ── Application ────────────────────────────────────────────────────────────────

def apply_manifest(
    config: UpmixConfig,
    manifest: dict[str, Any],
    *,
    allow_unknown_keys: bool = False,
) -> dict[str, Any]:
    """Apply manifest values to a :class:`~upmixer.config.UpmixConfig`.

    Profile (if specified in manifest) is applied first so that individual
    field values in the manifest can override the profile defaults.

    CLI flag values are NOT applied here — the caller applies them afterwards
    so they win over the manifest.

    Args:
        config:            The config object to modify *in-place*.
        manifest:          Dict loaded by :func:`load_manifest`.
        allow_unknown_keys: If ``False`` (default), logs a warning for keys
                           that are neither known config fields nor job keys.

    Returns:
        ``job_params`` dict containing job-level keys:
        ``input``, ``output``, ``mode``, ``input_format``,
        ``stem_model``, ``stem_model_dir``.  Keys absent from the manifest
        are not present in the returned dict (caller uses ``.get()``).
    """
    # ── Expand nested mastering: section into flat keys ───────────────────────
    manifest = _expand_nested_sections(manifest)

    # ── Apply delivery profile (individual manifest fields override) ──────────
    profile_name: str | None = manifest.get("profile")
    if profile_name is not None:
        if profile_name not in PROFILES:
            raise ValueError(
                f"Unknown profile '{profile_name}' in manifest. "
                f"Valid choices: {sorted(PROFILES.keys())}"
            )
        profile = PROFILES[profile_name]
        config.loudness_normalize   = profile.loudness_normalize
        config.loudness_target_lkfs = profile.loudness_target_lkfs
        config.loudness_max_tp      = profile.loudness_max_tp
        config.output_subtype       = profile.output_subtype
        config.output_type          = profile.output_type
        config.output_sample_rate   = profile.sample_rate
        if profile.lfe_cutoff_hz is not None:
            config.lfe_cutoff_hz = float(profile.lfe_cutoff_hz)
        _log.info(
            "  Manifest profile: %s — %+.1f LKFS / %+.1f dBTP / %d kHz / %s / %s",
            profile.display_name,
            profile.loudness_target_lkfs,
            profile.loudness_max_tp,
            profile.sample_rate // 1000,
            profile.output_subtype,
            profile.output_type,
        )

    # ── Apply config fields ───────────────────────────────────────────────────
    for key, value in manifest.items():
        if value is None:
            continue  # null / omitted → keep config default
        if key in _JOB_KEYS:
            continue  # handled below / by caller

        if key not in _FIELD_MAP:
            if not allow_unknown_keys:
                _log.warning("Unknown manifest key '%s' — ignored", key)
            continue

        config_attr, coerce = _FIELD_MAP[key]
        try:
            coerced = coerce(value)
        except (TypeError, ValueError) as exc:
            raise ValueError(
                f"Manifest key '{key}': cannot convert {value!r} to {coerce.__name__}: {exc}"
            ) from exc
        setattr(config, config_attr, coerced)

    # ── Collect job-level params ──────────────────────────────────────────────
    job_params: dict[str, Any] = {}
    for key in _JOB_KEYS - {"profile"}:
        if key in manifest and manifest[key] is not None:
            job_params[key] = manifest[key]

    return job_params


def list_manifest_keys() -> dict[str, str]:
    """Return a human-readable mapping of manifest keys to config attributes.

    Useful for documentation and ``--manifest-help`` style output.
    """
    out: dict[str, str] = {}
    for mk, (ca, t) in sorted(_FIELD_MAP.items()):
        out[mk] = f"{ca}  ({t.__name__})"
    for jk in sorted(_JOB_KEYS):
        out[jk] = "job parameter"
    return out

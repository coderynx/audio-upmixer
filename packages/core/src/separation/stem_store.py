"""Plain stem store — a caller-owned directory of separated stems.

Unlike :class:`~upmixer.separation.stem_cache.StemCache`, this has no
cache-identity key: the caller (one directory per logical source, e.g. a
project track) owns the identity, so there is nothing to hash and nothing
that can go stale when a model/engine version changes.

Directory layout::

    {stem_dir}/
        stems.json      # schema, stem_keys, sample_rate, source_size
        Vocals.wav      # per-stem float32 WAV
        Bass.wav
        Vocals__front.wav   # zone-tagged: '@' replaced by '__'
        ...
"""
from __future__ import annotations

import json
import os
from pathlib import Path

import numpy as np

from upmixer.movement import (
    extract_feature_sidecar,
    validate_feature_sidecar,
    validate_feature_sidecar_for_stems,
)

_MANIFEST_FILE = "stems.json"
MOVEMENT_FEATURES_FILENAME = "movement-features.json"
_SCHEMA = 1


def _stem_filename(stem_key: str) -> str:
    """Convert a stem key (possibly zone-tagged) to a safe filename.

    ``"Vocals@front"`` → ``"Vocals__front.wav"``
    """
    safe = stem_key.replace("@", "__").replace("/", "__").replace("\\", "__")
    return f"{safe}.wav"


class PlainStemStore:
    """Read/write a directory of separated stems with no cache identity.

    Args:
        stem_dir: Directory to read/write. Created on write if missing.
    """

    def __init__(self, stem_dir: str) -> None:
        self._root = Path(stem_dir)

    def load(self) -> tuple[dict[str, np.ndarray], int] | None:
        """Load previously written stems, or ``None`` if none are present.

        Returns:
            ``(stems_dict, sample_rate)``. Stems are float32 arrays shaped
            ``(n_samples, 2)``.
        """
        manifest_path = self._root / _MANIFEST_FILE
        if not manifest_path.exists():
            return None
        try:
            manifest = json.loads(manifest_path.read_text(encoding="utf-8"))
        except (OSError, ValueError):
            return None
        stem_keys = manifest.get("stem_keys")
        sample_rate = manifest.get("sample_rate")
        if not stem_keys or not sample_rate:
            return None
        try:
            sample_rate = int(sample_rate)
        except (TypeError, ValueError):
            return None
        if sample_rate <= 0:
            return None

        import soundfile as sf  # type: ignore[import-untyped]

        stems: dict[str, np.ndarray] = {}
        for stem_key in stem_keys:
            wav_path = self._root / _stem_filename(stem_key)
            if not wav_path.exists():
                return None
            data, file_sample_rate = sf.read(
                str(wav_path), dtype="float32", always_2d=True,
            )
            if file_sample_rate != sample_rate:
                return None
            stems[stem_key] = data
        if not stems:
            return None
        return stems, sample_rate

    def write(
        self,
        stems: dict[str, np.ndarray],
        sample_rate: int,
        *,
        source_size: int | None = None,
        movement_features: dict | None = None,
    ) -> None:
        """Write stems, replacing any previous contents of this directory."""
        import soundfile as sf  # type: ignore[import-untyped]

        if movement_features is None and stems:
            movement_features = extract_feature_sidecar(stems, sample_rate)
        elif movement_features is not None:
            validate_feature_sidecar_for_stems(movement_features, stems, sample_rate)
        self._root.mkdir(parents=True, exist_ok=True)
        previous_stem_files: set[str] = set()
        try:
            previous_manifest = json.loads(
                (self._root / _MANIFEST_FILE).read_text(encoding="utf-8")
            )
        except (OSError, ValueError):
            previous_manifest = None
        if isinstance(previous_manifest, dict):
            previous_keys = previous_manifest.get("stem_keys")
            if isinstance(previous_keys, list):
                previous_stem_files = {
                    _stem_filename(key)
                    for key in previous_keys
                    if isinstance(key, str)
                }

        for stem_key, audio in stems.items():
            wav_path = self._root / _stem_filename(stem_key)
            arr = audio if audio.ndim == 2 else audio[:, np.newaxis]
            temp_path = self._root / f".{wav_path.stem}.tmp.wav"
            sf.write(str(temp_path), arr.astype(np.float32, copy=False), sample_rate, subtype="FLOAT")
            os.replace(temp_path, wav_path)

        manifest = {
            "schema": _SCHEMA,
            "stem_keys": list(stems.keys()),
            "sample_rate": sample_rate,
            "source_size": source_size,
        }
        manifest_path = self._root / _MANIFEST_FILE
        temp_manifest = self._root / f".{_MANIFEST_FILE}.tmp"
        temp_manifest.write_text(json.dumps(manifest, indent=2), encoding="utf-8")
        os.replace(temp_manifest, manifest_path)

        current_stem_files = {_stem_filename(stem_key) for stem_key in stems}
        for filename in previous_stem_files - current_stem_files:
            (self._root / filename).unlink(missing_ok=True)

        # Empty stores have no analysis; remove any stale sidecar.
        if movement_features is None:
            (self._root / MOVEMENT_FEATURES_FILENAME).unlink(missing_ok=True)
        else:
            self.write_features(movement_features, stem_keys=list(stems))

    def write_features(
        self, features: dict, *, stem_keys: list[str] | tuple[str, ...] | None = None,
    ) -> None:
        """Atomically persist canonical features produced by the shared DSP."""
        expected = list(stem_keys) if stem_keys is not None else self._manifest_stem_keys()
        validate_feature_sidecar(features, expected)
        self._root.mkdir(parents=True, exist_ok=True)
        destination = self._root / MOVEMENT_FEATURES_FILENAME
        temporary = self._root / f".{MOVEMENT_FEATURES_FILENAME}.tmp"
        temporary.write_text(
            json.dumps(features, indent=2, allow_nan=False), encoding="utf-8"
        )
        os.replace(temporary, destination)

    def load_features(
        self, stem_keys: list[str] | tuple[str, ...] | None = None,
    ) -> dict | None:
        """Load and shared-DSP-validate canonical features, or return ``None``."""
        path = self._root / MOVEMENT_FEATURES_FILENAME
        try:
            features = json.loads(path.read_text(encoding="utf-8"))
            expected = list(stem_keys) if stem_keys is not None else self._manifest_stem_keys()
            return validate_feature_sidecar(features, expected)
        except (OSError, ValueError, TypeError):
            return None

    def _manifest_stem_keys(self) -> list[str]:
        try:
            manifest = json.loads((self._root / _MANIFEST_FILE).read_text(encoding="utf-8"))
        except (OSError, ValueError, TypeError):
            return []
        keys = manifest.get("stem_keys") if isinstance(manifest, dict) else None
        return [str(key) for key in keys] if isinstance(keys, list) else []

"""Stem separation cache — skip re-separating unchanged input files.

Cache structure on disk::

    {cache_dir}/
        {key}/
            metadata.json          # cache-key components for validation
            Vocals.wav             # per-stem PCM_24 WAV
            Bass.wav
            Drums.wav
            Other.wav
            Vocals__front.wav      # zone-tagged: '@' replaced by '__'
            ...

Cache key: SHA-256 of ``abs_path|mtime|model|sep_sr`` (first 20 hex chars).

Cache invalidation: the key encodes the input file's absolute path, mtime
(float, 6 decimal places), stem model name, and target sample rate.  Any
change to these four factors produces a different key → cold miss.

Stems are stored as float32 PCM_24 WAV (soundfile).  On load, arrays are
returned as float64 to match the rest of the pipeline.
"""
from __future__ import annotations

import hashlib
import json
import logging
import os
from pathlib import Path

import numpy as np

_log = logging.getLogger("upmixer")

_METADATA_FILE = "metadata.json"
_MTIME_TOLERANCE = 2.0  # seconds — FAT32 / network FS mtime granularity


def _cache_key(input_path: str, model: str, sep_sr: int) -> str:
    """Return a 20-char hex cache key for the given separation parameters."""
    abs_path = str(Path(input_path).resolve())
    mtime = os.path.getmtime(abs_path)
    raw = f"{abs_path}|{mtime:.6f}|{model}|{sep_sr}"
    return hashlib.sha256(raw.encode()).hexdigest()[:20]


def _stem_filename(stem_key: str) -> str:
    """Convert a stem key (possibly zone-tagged) to a safe filename.

    ``"Vocals@front"`` → ``"Vocals__front.wav"``
    """
    safe = stem_key.replace("@", "__").replace("/", "__").replace("\\", "__")
    return f"{safe}.wav"


class StemCache:
    """On-disk cache for separated stems.

    Args:
        cache_dir: Root directory for the cache.  Created if it does not exist.
    """

    def __init__(self, cache_dir: str) -> None:
        self._root = Path(cache_dir)
        self._root.mkdir(parents=True, exist_ok=True)

    # ── load ──────────────────────────────────────────────────────────────────

    def load(
        self,
        input_path: str,
        model: str,
        sep_sr: int,
    ) -> tuple[dict[str, np.ndarray], int] | None:
        """Try to load cached stems for the given parameters.

        Args:
            input_path: Original input audio file path.
            model:      Stem separation model name.
            sep_sr:     Target separation sample rate in Hz.

        Returns:
            ``(stems_dict, sample_rate)`` on cache hit, or ``None`` on miss.
            Stems are returned as float64 arrays shaped ``(n_samples, 2)``.
        """
        key = _cache_key(input_path, model, sep_sr)
        entry_dir = self._root / key

        if not entry_dir.exists():
            return None

        meta_path = entry_dir / _METADATA_FILE
        if not meta_path.exists():
            _log.debug("  StemCache: metadata missing for key %s", key)
            return None

        try:
            meta = json.loads(meta_path.read_text(encoding="utf-8"))
        except Exception as exc:  # noqa: BLE001
            _log.debug("  StemCache: corrupt metadata (%s), ignoring", exc)
            return None

        # Validate mtime with tolerance (FAT32, NFS, samba can drift ±2 s)
        try:
            current_mtime = os.path.getmtime(str(Path(input_path).resolve()))
        except OSError:
            return None  # input file gone
        stored_mtime = float(meta.get("mtime", 0.0))
        if abs(current_mtime - stored_mtime) > _MTIME_TOLERANCE:
            _log.debug(
                "  StemCache: mtime mismatch (stored=%.3f, current=%.3f)",
                stored_mtime, current_mtime,
            )
            return None

        # Load stem WAV files
        try:
            import soundfile as sf  # type: ignore[import-untyped]
        except ImportError:
            _log.debug("  StemCache: soundfile not available, skipping cache")
            return None

        stems: dict[str, np.ndarray] = {}
        stem_keys: list[str] = meta.get("stem_keys", [])
        for stem_key in stem_keys:
            wav_path = entry_dir / _stem_filename(stem_key)
            if not wav_path.exists():
                _log.debug(
                    "  StemCache: missing file %s for key %s", wav_path.name, key
                )
                return None  # partial cache → cold miss
            data, _ = sf.read(str(wav_path), dtype="float64", always_2d=True)
            stems[stem_key] = data

        if not stems:
            return None

        _log.info(
            "  StemCache: HIT — loaded %d stems from %s",
            len(stems), entry_dir,
        )
        return stems, sep_sr

    # ── save ──────────────────────────────────────────────────────────────────

    def save(
        self,
        input_path: str,
        model: str,
        sep_sr: int,
        stems: dict[str, np.ndarray],
        sample_rate: int,
    ) -> None:
        """Write stems to the cache.

        Args:
            input_path: Original input audio file path.
            model:      Stem separation model name.
            sep_sr:     Target separation sample rate in Hz.
            stems:      Dict stem_key → ``(n_samples, 2)`` float array.
            sample_rate: Actual sample rate of the stems (should equal sep_sr).
        """
        try:
            import soundfile as sf  # type: ignore[import-untyped]
        except ImportError:
            _log.debug("  StemCache: soundfile not available, skipping cache write")
            return

        abs_path = str(Path(input_path).resolve())
        mtime = os.path.getmtime(abs_path)
        key = _cache_key(input_path, model, sep_sr)
        entry_dir = self._root / key
        entry_dir.mkdir(parents=True, exist_ok=True)

        # Write stems
        for stem_key, audio in stems.items():
            wav_path = entry_dir / _stem_filename(stem_key)
            # Ensure 2-D for soundfile
            arr = audio if audio.ndim == 2 else audio[:, np.newaxis]
            sf.write(str(wav_path), arr.astype(np.float32), sample_rate, subtype="PCM_24")

        # Write metadata
        meta = {
            "input_path": abs_path,
            "mtime": round(mtime, 6),
            "model": model,
            "sep_sr": sep_sr,
            "stem_keys": list(stems.keys()),
        }
        (entry_dir / _METADATA_FILE).write_text(
            json.dumps(meta, indent=2), encoding="utf-8"
        )
        _log.info(
            "  StemCache: saved %d stems → %s",
            len(stems), entry_dir,
        )

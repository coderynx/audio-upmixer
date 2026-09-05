#!/usr/bin/env python3
"""Prepare a relative MUSDB18-HQ evaluation manifest and provenance file.

The input is either ``DATASET_ROOT/{tuning,heldout}/<track>`` or
``DATASET_ROOT/subset/{tuning,heldout}/<track>``. Audio is inspected in place;
no audio files are copied to the output directory.

Example:
    uv run python scripts/prepare_musdb18hq_corpus.py \
        --dataset-root /data/musdb18-hq --output-dir /data/musdb18-hq-manifest
"""

from __future__ import annotations

import argparse
import hashlib
import json
import math
import os
from contextlib import ExitStack
from pathlib import Path
from typing import Any

import numpy as np
import soundfile as sf

from upmixer.io.atomic import atomic_output_path

SAMPLE_RATE = 44_100
BLOCK_SIZE = 65_536
CATEGORY = "musdb18-hq-baseline"
SPLITS = ("tuning", "heldout")
REQUIRED_FILES = (
    ("mixture", "mixture.wav"),
    ("Vocals", "vocals.wav"),
    ("Bass", "bass.wav"),
    ("Drums", "drums.wav"),
    ("Other", "other.wav"),
)


def _split_dir(dataset_root: Path, split: str) -> Path:
    for candidate in (dataset_root / split, dataset_root / "subset" / split):
        if candidate.is_dir():
            return candidate
    raise ValueError(f"missing {split} split under {dataset_root}")


def _track_dirs(split_dir: Path, split: str) -> list[Path]:
    tracks = sorted(
        (
            path
            for path in split_dir.iterdir()
            if path.is_dir() and not path.name.startswith(".")
        ),
        key=lambda path: path.name,
    )
    if len(tracks) < 12:
        raise ValueError(
            f"{split} requires at least 12 independent tracks, found {len(tracks)}"
        )
    return tracks


def _selection_manifest(path: Path) -> tuple[str, list[str], dict[str, Any]]:
    try:
        selection = json.loads(path.read_text(encoding="utf-8"))
    except (OSError, json.JSONDecodeError) as exc:
        raise ValueError(f"invalid selection manifest {path}: {exc}") from exc
    if not isinstance(selection, dict):
        raise ValueError(f"invalid selection manifest {path}: expected an object")
    selection_id = selection.get("selection_id")
    if not isinstance(selection_id, str) or not selection_id.strip():
        raise ValueError(f"selection manifest {path} requires a nonempty selection_id")

    tracks = selection.get("tracks")
    if not isinstance(tracks, dict):
        raise ValueError(f"selection manifest {path} requires tracks by split")
    membership = []
    for split in SPLITS:
        names = tracks.get(split)
        if not isinstance(names, list) or not all(
            isinstance(name, str) and name for name in names
        ):
            raise ValueError(f"selection manifest {path} requires {split} track names")
        membership.extend(f"{split}/{name}" for name in names)
    if len(membership) != len(set(membership)):
        raise ValueError(f"selection manifest {path} contains duplicate tracks")

    archive = selection.get("archive")
    if not isinstance(archive, dict):
        archive = {}

    def value(*names: str) -> Any:
        for name in names:
            candidate = selection.get(name, archive.get(name))
            if candidate is not None and candidate != "":
                return candidate
        return "unknown"

    metadata = {
        "archive_subset": value("archive_subset", "subset"),
        "archive_identity": value("archive_identity", "identity"),
        "benchmark_overlap": value("benchmark_overlap", "benchmark_overlap_statement"),
    }
    return selection_id.strip(), sorted(membership), metadata


def _membership_hash(membership: list[str]) -> str:
    payload = ("\n".join(membership) + "\n").encode("utf-8")
    return hashlib.sha256(payload).hexdigest()


def _sha256(path: Path) -> str:
    digest = hashlib.sha256()
    with path.open("rb") as handle:
        for block in iter(lambda: handle.read(1 << 20), b""):
            digest.update(block)
    return digest.hexdigest()


def _audio_metadata(path: Path, info: Any) -> dict[str, Any]:
    if info.samplerate <= 0:
        raise ValueError(f"{path}: sample rate must be positive")
    duration_s = info.frames / float(info.samplerate)
    if info.frames <= 0 or not math.isfinite(duration_s) or duration_s <= 0.0:
        raise ValueError(f"{path}: duration must be finite and positive")
    if info.samplerate != SAMPLE_RATE:
        raise ValueError(f"{path}: expected {SAMPLE_RATE} Hz, got {info.samplerate} Hz")
    if info.channels != 2:
        raise ValueError(f"{path}: expected stereo audio, got {info.channels} channels")
    return {
        "sha256": _sha256(path),
        "bytes": path.stat().st_size,
        "sample_rate": info.samplerate,
        "channels": info.channels,
        "frames": info.frames,
        "duration_s": duration_s,
    }


def _residual(files: dict[str, Path], frames: int, label: str) -> dict[str, Any]:
    total_squared = 0.0
    peak = 0.0
    samples = 0
    with ExitStack() as stack:
        handles = [
            stack.enter_context(sf.SoundFile(str(files[name]), mode="r"))
            for name, _ in REQUIRED_FILES
        ]
        while True:
            blocks = [
                handle.read(BLOCK_SIZE, dtype="float32", always_2d=True)
                for handle in handles
            ]
            mixture = blocks[0]
            if len(mixture) == 0:
                break
            if any(block.shape != mixture.shape for block in blocks[1:]):
                raise ValueError(f"{label}: audio frame shape changed while reading")
            if not all(np.isfinite(block).all() for block in blocks):
                raise ValueError(f"{label}: audio contains non-finite samples")
            difference = mixture.astype(np.float64, copy=True)
            for stem in blocks[1:]:
                difference -= stem
            total_squared += float(np.sum(difference * difference, dtype=np.float64))
            peak = max(peak, float(np.max(np.abs(difference))))
            samples += difference.size
    if samples != frames * 2:
        raise ValueError(
            f"{label}: expected {frames} stereo frames, read {samples // 2}"
        )
    return {
        "block_size": BLOCK_SIZE,
        "frames": frames,
        "channels": 2,
        "rms": math.sqrt(total_squared / samples),
        "peak": peak,
    }


def _relative(path: Path, output_dir: Path) -> str:
    return Path(os.path.relpath(path, output_dir)).as_posix()


def _inspect_recording(
    split: str, track_dir: Path, output_dir: Path
) -> tuple[dict[str, Any], dict[str, Any]]:
    label = f"{split}/{track_dir.name}"
    files = {name: track_dir / filename for name, filename in REQUIRED_FILES}
    missing = [
        filename for name, filename in REQUIRED_FILES if not files[name].is_file()
    ]
    if missing:
        raise ValueError(f"{label}: missing required files: {', '.join(missing)}")

    try:
        infos = {name: sf.info(str(path)) for name, path in files.items()}
    except (OSError, RuntimeError) as exc:
        raise ValueError(f"{label}: invalid audio file: {exc}") from exc
    metadata = {
        name: _audio_metadata(path, infos[name]) for name, path in files.items()
    }
    frames = metadata["mixture"]["frames"]
    for name, file_metadata in metadata.items():
        if file_metadata["frames"] != frames:
            raise ValueError(
                f"{label}: {name} has {file_metadata['frames']} frames, expected {frames}"
            )
    residual = _residual(files, frames, label)
    recording_id = f"musdb18-hq/{track_dir.name}"
    item_id = f"musdb18-hq/{split}/{track_dir.name}"
    item = {
        "mixture": _relative(files["mixture"], output_dir),
        "stems": {
            name: _relative(files[name], output_dir) for name, _ in REQUIRED_FILES[1:]
        },
        "category": CATEGORY,
        "recording_id": recording_id,
        "item_id": item_id,
        "split": split,
    }
    provenance = {
        "recording_id": recording_id,
        "item_id": item_id,
        "split": split,
        "category": CATEGORY,
        "recording_group": track_dir.name,
        "mixture_construction": (
            "provided mixture; residual measured against the sum of the four references"
        ),
        "common_gain": "unknown",
        "master_bus_processing": "unknown",
        "overlap_with_pretrained_benchmarks": "unknown",
        "available_references": [name for name, _ in REQUIRED_FILES[1:]],
        "files": {
            name: {
                "path": _relative(files[name], output_dir),
                **metadata[name],
            }
            for name, _ in REQUIRED_FILES
        },
        "residual": residual,
    }
    return item, provenance


def _write_json(path: Path, payload: dict[str, Any]) -> None:
    with atomic_output_path(path) as temporary:
        temporary.write_text(
            json.dumps(payload, indent=2, sort_keys=True) + "\n",
            encoding="utf-8",
        )


def prepare_corpus(
    dataset_root: str | Path, output_dir: str | Path | None = None
) -> tuple[Path, Path]:
    """Write ``corpus.json`` and ``provenance.json`` for MUSDB18-HQ audio."""
    dataset_root = Path(dataset_root).expanduser().resolve()
    if not dataset_root.is_dir():
        raise ValueError(f"dataset root is not a directory: {dataset_root}")
    output_dir = (
        dataset_root if output_dir is None else Path(output_dir).expanduser().resolve()
    )
    split_dirs = {split: _split_dir(dataset_root, split) for split in SPLITS}
    tracks = {split: _track_dirs(split_dirs[split], split) for split in SPLITS}
    seen: dict[str, str] = {}
    for split in SPLITS:
        for track in tracks[split]:
            key = track.name.casefold()
            if key in seen:
                raise ValueError(
                    f"track {track.name!r} is present in both {seen[key]} and {split}"
                )
            seen[key] = split

    membership = [
        f"{split}/{track.name}" for split in SPLITS for track in tracks[split]
    ]
    membership.sort()
    membership_sha256 = _membership_hash(membership)
    selection_path = dataset_root / "selection.json"
    selection = None
    selection_metadata = {
        "archive_subset": "unknown",
        "archive_identity": "MUSDB18-HQ@10.5281/zenodo.3338373",
        "benchmark_overlap": "unknown",
    }
    if selection_path.exists():
        selection_id, selected_membership, selected_metadata = _selection_manifest(
            selection_path
        )
        expected = set(membership)
        selected = set(selected_membership)
        if expected != selected:
            missing = sorted(expected - selected)
            extra = sorted(selected - expected)
            details = []
            if missing:
                details.append(f"missing={missing}")
            if extra:
                details.append(f"extra={extra}")
            raise ValueError(
                "selection.json membership mismatch: " + "; ".join(details)
            )
        corpus_id = selection_id
        selection_metadata.update(selected_metadata)
        selection = {
            "selection_id": corpus_id,
            **selection_metadata,
        }
    else:
        corpus_id = f"musdb18-hq-{membership_sha256}"

    items = []
    recordings = []
    mixture_hashes: dict[str, str] = {}
    for split in SPLITS:
        for track in tracks[split]:
            item, provenance = _inspect_recording(split, track, output_dir)
            mixture_hash = provenance["files"]["mixture"]["sha256"]
            previous = mixture_hashes.get(mixture_hash)
            if previous is not None:
                raise ValueError(
                    f"duplicate mixture SHA-256 for {previous} and {split}/{track.name}"
                )
            mixture_hashes[mixture_hash] = f"{split}/{track.name}"
            items.append(item)
            recordings.append(provenance)

    manifest = {
        "schema_version": 1,
        "corpus_id": corpus_id,
        "membership_sha256": membership_sha256,
        "items": items,
    }
    dataset_metadata = {
        "name": "MUSDB18-HQ",
        "doi": "10.5281/zenodo.3338373",
        "source": "MUSDB18-HQ archive on Zenodo",
        "source_url": "https://doi.org/10.5281/zenodo.3338373",
        "license": "educational-only",
        **selection_metadata,
    }
    provenance = {
        "schema_version": 1,
        "corpus_id": corpus_id,
        "membership_sha256": membership_sha256,
        "dataset": dataset_metadata,
        "splits": {split: {"n_recordings": len(tracks[split])} for split in SPLITS},
        "recordings": recordings,
    }
    if selection is not None:
        provenance["selection"] = selection
    output_dir.mkdir(parents=True, exist_ok=True)
    corpus_path = output_dir / "corpus.json"
    provenance_path = output_dir / "provenance.json"
    _write_json(corpus_path, manifest)
    _write_json(provenance_path, provenance)
    return corpus_path, provenance_path


def _parser() -> argparse.ArgumentParser:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--dataset-root", required=True, type=Path, metavar="PATH")
    parser.add_argument("--output-dir", type=Path, metavar="DIR")
    return parser


def main(argv: list[str] | None = None) -> int:
    args = _parser().parse_args(argv)
    corpus_path, provenance_path = prepare_corpus(args.dataset_root, args.output_dir)
    print(f"Wrote {corpus_path}")
    print(f"Wrote {provenance_path}")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())

#!/usr/bin/env python3
"""Run a reproducible evaluation report.

Examples:
    uv run python scripts/run_eval.py --corpus synthetic \
        --variant synthetic-reference --output-dir /tmp/upmixer-eval
    uv run python scripts/run_eval.py --corpus PATH \
        --variant real-model --output-dir /tmp/upmixer-eval \
        --model BS-Roformer-SW.ckpt
    uv run python scripts/run_eval.py --corpus synthetic \
        --variant production-tree --output-dir /tmp/upmixer-eval-tree \
        --stems vocals,bass,drums,other
"""
from __future__ import annotations

import argparse
import math
import shutil
import subprocess
from functools import partial
from pathlib import Path
from typing import Callable, Sequence

import numpy as np
import soundfile as sf

from upmixer.config import UpmixConfig
from upmixer.execution import write_report
from upmixer.eval import (
    ReferenceCorpus,
    RunSettings,
    evaluate_corpus,
    format_report,
    separate_for_eval,
    separate_tree_for_eval,
    synthetic_corpus,
)
from upmixer.separation.stem_plan import normalize_stems
from upmixer.io.atomic import atomic_output_path

_PROTOCOL_ID = "upmixer-separation-q00-v1"
_REPOSITORY_ROOT = Path(__file__).resolve().parents[1]
_STEM_INDEX_SCHEMA = 1


def _positive_int(value: str) -> int:
    parsed = int(value)
    if parsed < 1:
        raise argparse.ArgumentTypeError("must be at least 1")
    return parsed


def _finite_positive(value: str) -> float:
    parsed = float(value)
    if not math.isfinite(parsed) or parsed <= 0.0:
        raise argparse.ArgumentTypeError("must be a finite positive number")
    return parsed


def _comma_separated_stems(value: str) -> list[str]:
    stems = [stem.strip() for stem in value.split(",") if stem.strip()]
    if not stems:
        raise argparse.ArgumentTypeError("must include at least one stem")
    return stems


def _parser() -> argparse.ArgumentParser:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--corpus", required=True, metavar="PATH|synthetic")
    parser.add_argument(
        "--variant",
        required=True,
        choices=("synthetic-reference", "real-model", "production-tree"),
    )
    parser.add_argument("--output-dir", required=True, type=Path, metavar="DIR")
    parser.add_argument("--sample-rate", type=_positive_int, default=44_100)
    parser.add_argument("--model", default=None)
    parser.add_argument("--batch-size", type=_positive_int, default=None)
    parser.add_argument("--segment-size", type=_positive_int, default=None)
    parser.add_argument("--chunk-duration-s", type=_finite_positive, default=None)
    parser.add_argument("--overlap", type=_positive_int, default=None)
    parser.add_argument("--stem-ensemble", action="store_true")
    parser.add_argument("--tta", action="store_true")
    parser.add_argument("--pitch-shift", type=_finite_positive, default=None)
    parser.add_argument(
        "--stems",
        type=_comma_separated_stems,
        default=None,
        metavar="NAME[,NAME...]",
        help="Requested production-tree stems; use manifest or canonical names.",
    )
    parser.add_argument(
        "--retain-stems",
        action="store_true",
        help="Keep each completed item's pre-routing float32 stems.",
    )
    return parser


def _reference_separator(corpus: ReferenceCorpus, sample_rate: int) -> Callable:
    items = {item.mixture: item for item in corpus.items}

    def separate(mixture_path: str):
        item = items[mixture_path]
        stems = {
            name: sf.read(path, dtype="float32", always_2d=True)[0]
            for name, path in item.stems.items()
        }
        return stems, RunSettings(model="synthetic-reference", sample_rate=sample_rate)

    return separate


def _real_separator(args: argparse.Namespace) -> Callable:
    if args.variant == "production-tree":
        config = UpmixConfig(
            output_sample_rate=args.sample_rate,
            stems=args.stems,
            stem_batch_size=args.batch_size,
            stem_segment_size=args.segment_size,
            stem_chunk_duration_s=args.chunk_duration_s,
            stem_overlap=args.overlap,
            stem_ensemble=args.stem_ensemble,
            stem_tta=args.tta,
            stem_pitch_shift=args.pitch_shift,
        )
        return partial(
            separate_tree_for_eval,
            sample_rate=args.sample_rate,
            config=config,
        )
    options = {
        "sample_rate": args.sample_rate,
        "batch_size": args.batch_size,
        "segment_size": args.segment_size,
        "chunk_duration_s": args.chunk_duration_s,
        "overlap": args.overlap,
        "stem_ensemble": args.stem_ensemble,
        "tta": args.tta,
        "pitch_shift": args.pitch_shift,
    }
    if args.model is not None:
        options["model"] = args.model
    return partial(separate_for_eval, **options)


def _fresh_output_dir(path: Path, parser: argparse.ArgumentParser) -> None:
    if path.exists() and (not path.is_dir() or any(path.iterdir())):
        parser.error(f"output directory must be fresh and empty: {path}")
    path.mkdir(parents=True, exist_ok=True)


def _git_revision() -> str | None:
    """Return the repository HEAD, marking a dirty working tree when known."""
    try:
        head = subprocess.run(
            ["git", "rev-parse", "--verify", "HEAD"],
            cwd=_REPOSITORY_ROOT,
            check=True,
            capture_output=True,
            text=True,
        ).stdout.strip()
        if not head:
            return None
        status = subprocess.run(
            ["git", "status", "--porcelain"],
            cwd=_REPOSITORY_ROOT,
            check=True,
            capture_output=True,
            text=True,
        )
    except (OSError, subprocess.CalledProcessError):
        return None
    return f"{head}-dirty" if status.stdout else head


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
        stems, settings = separate_fn(mixture_path)
        if not isinstance(stems, dict) or not stems:
            return stems, settings
        if not isinstance(settings, RunSettings):
            return stems, settings
        sample_rate = settings.sample_rate
        if isinstance(sample_rate, bool) or not isinstance(sample_rate, int) or sample_rate < 1:
            raise ValueError("retained stems require a positive settings sample rate")
        if sample_rate != evaluation_sample_rate:
            return stems, settings
        if item.stems and not set(item.stems).issubset(stems):
            return stems, settings
        retained: dict[str, np.ndarray] = {}
        try:
            for stem_name, value in stems.items():
                if not isinstance(stem_name, str):
                    return stems, settings
                raw_audio = np.asarray(value)
                if (
                    raw_audio.ndim != 2
                    or not raw_audio.size
                    or not raw_audio.shape[0]
                    or not raw_audio.shape[1]
                    or not np.issubdtype(raw_audio.dtype, np.number)
                ):
                    return stems, settings
                audio = np.asarray(raw_audio, dtype=np.float32)
                if not np.all(np.isfinite(audio)):
                    return stems, settings
                if stem_name in item.stems:
                    info = sf.info(item.stems[stem_name])
                    if (
                        info.samplerate != sample_rate
                        or info.frames != audio.shape[0]
                        or info.channels != audio.shape[1]
                    ):
                        return stems, settings
                retained[stem_name] = audio
        except (OSError, RuntimeError, TypeError, ValueError):
            return stems, settings

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
                        str(temporary), retained[stem_name], sample_rate, subtype="FLOAT"
                    )
                paths[str(stem_name)] = destination.relative_to(output_dir).as_posix()
            entry = {
                "index": item_index,
                "recording_id": item.recording_id,
                "item_id": item.item_id,
                "split": item.split,
                "category": item.category,
                "sample_rate": sample_rate,
                "stems": paths,
            }
            write_report(
                index_path,
                {"schema_version": _STEM_INDEX_SCHEMA, "items": [*entries, entry]},
            )
        except BaseException:
            shutil.rmtree(item_dir, ignore_errors=True)
            raise
        entries.append(entry)
        return stems, settings

    return separate


def main(argv: Sequence[str] | None = None) -> int:
    parser = _parser()
    args = parser.parse_args(argv)
    if args.variant == "production-tree" and args.model is not None:
        parser.error(
            "production-tree does not accept --model; the plan owns model selection"
        )
    if args.stems is not None and args.variant != "production-tree":
        parser.error("--stems requires --variant production-tree")
    if args.variant == "synthetic-reference":
        if any(
            value is not None
            for value in (
                args.model,
                args.batch_size,
                args.segment_size,
                args.chunk_duration_s,
                args.overlap,
                args.pitch_shift,
            )
        ) or args.stem_ensemble or args.tta:
            parser.error("model settings require --variant real-model")
        if args.retain_stems:
            parser.error("--retain-stems requires --variant real-model or production-tree")
    if args.stems is not None:
        try:
            args.stems = normalize_stems(args.stems)
        except ValueError as exc:
            parser.error(str(exc))

    code_revision = _git_revision()
    _fresh_output_dir(args.output_dir, parser)

    corpus = (
        synthetic_corpus(args.sample_rate, str(args.output_dir / "corpus"))
        if args.corpus == "synthetic"
        else ReferenceCorpus.from_dir(args.corpus)
    )
    if args.variant == "synthetic-reference":
        separate_fn = _reference_separator(corpus, args.sample_rate)
    else:
        separate_fn = _real_separator(args)
    if args.retain_stems:
        separate_fn = _retaining_separator(
            separate_fn, corpus, args.output_dir, args.sample_rate
        )

    report = evaluate_corpus(
        corpus,
        separate_fn,
        sample_rate=args.sample_rate,
        protocol_id=_PROTOCOL_ID,
        code_revision=code_revision,
        report_failures=True,
    )
    report.write_json(args.output_dir / "report.json")
    text = format_report(report)
    (args.output_dir / "report.txt").write_text(text + "\n", encoding="utf-8")
    print(text)
    print(f"\nWrote {args.output_dir / 'report.json'}")
    print(f"Wrote {args.output_dir / 'report.txt'}")
    return int(
        any(
            row.status in {"failed", "absent"}
            for row in getattr(report, "coverage", ())
        )
    )


if __name__ == "__main__":
    raise SystemExit(main())

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
from functools import partial
from pathlib import Path
from typing import Callable, Sequence

import soundfile as sf

from upmixer.config import UpmixConfig
from upmixer.eval import (
    ReferenceCorpus,
    RunSettings,
    evaluate_corpus,
    format_report,
    separate_for_eval,
    separate_tree_for_eval,
    synthetic_corpus,
)


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


def main(argv: Sequence[str] | None = None) -> int:
    parser = _parser()
    args = parser.parse_args(argv)
    _fresh_output_dir(args.output_dir, parser)
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

    corpus = (
        synthetic_corpus(args.sample_rate, str(args.output_dir / "corpus"))
        if args.corpus == "synthetic"
        else ReferenceCorpus.from_dir(args.corpus)
    )
    if args.variant == "synthetic-reference":
        separate_fn = _reference_separator(corpus, args.sample_rate)
    else:
        separate_fn = _real_separator(args)

    report = evaluate_corpus(corpus, separate_fn, sample_rate=args.sample_rate)
    report.write_json(args.output_dir / "report.json")
    text = format_report(report)
    (args.output_dir / "report.txt").write_text(text + "\n", encoding="utf-8")
    print(text)
    print(f"\nWrote {args.output_dir / 'report.json'}")
    print(f"Wrote {args.output_dir / 'report.txt'}")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())

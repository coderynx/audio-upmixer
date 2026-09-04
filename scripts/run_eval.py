#!/usr/bin/env python3
"""Run a reproducible evaluation report.

Examples:
    uv run python scripts/run_eval.py --corpus synthetic \
        --variant synthetic-reference --output-dir /tmp/upmixer-eval
    uv run python scripts/run_eval.py --corpus PATH \
        --variant real-model --output-dir /tmp/upmixer-eval \
        --model BS-Roformer-SW.ckpt
"""
from __future__ import annotations

import argparse
from functools import partial
from pathlib import Path
from typing import Callable, Sequence

import soundfile as sf

from upmixer.eval import (
    ReferenceCorpus,
    RunSettings,
    evaluate_corpus,
    format_report,
    separate_for_eval,
    synthetic_corpus,
)


def _positive_int(value: str) -> int:
    parsed = int(value)
    if parsed < 1:
        raise argparse.ArgumentTypeError("must be at least 1")
    return parsed


def _parser() -> argparse.ArgumentParser:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--corpus", required=True, metavar="PATH|synthetic")
    parser.add_argument(
        "--variant",
        required=True,
        choices=("synthetic-reference", "real-model"),
    )
    parser.add_argument("--output-dir", required=True, type=Path, metavar="DIR")
    parser.add_argument("--sample-rate", type=_positive_int, default=44_100)
    parser.add_argument("--model", default=None)
    parser.add_argument("--batch-size", type=_positive_int, default=None)
    parser.add_argument("--segment-size", type=_positive_int, default=None)
    parser.add_argument("--chunk-duration-s", type=float, default=None)
    parser.add_argument("--overlap", type=_positive_int, default=None)
    parser.add_argument("--stem-ensemble", action="store_true")
    parser.add_argument("--tta", action="store_true")
    parser.add_argument("--pitch-shift", type=float, default=None)
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

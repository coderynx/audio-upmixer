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
import hashlib
import math
import subprocess
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
    separate_with_deux_cascade,
    separate_with_extra_origin,
    separate_tree_for_eval,
    synthetic_corpus,
)
from upmixer.eval.cascade import CASCADE_RECIPE
from upmixer.eval.rate_experiment import (
    separate_model_for_rate_experiment,
    separate_tree_for_rate_experiment,
)
from upmixer.eval.retention import _retaining_separator
from upmixer.separation.stem_plan import normalize_stems
from upmixer.separation.separator import DEFAULT_MODEL

_PROTOCOL_ID = "upmixer-separation-q00-v1"
_REPOSITORY_ROOT = Path(__file__).resolve().parents[1]
_CASCADE_CORPUS_SHA256 = (
    "b835022cb61d8b61197f4d0521686e3f42628c864a36138f46921c0cef0c694e"
)
_CASCADE_CORPUS_ID = "upmixer-musdb18hq-v1-q40-deux-counterfactual-half-v1"
_CASCADE_ITEM_ID = "musdb18-hq/tuning/Hollow Ground - Ill Fate#60s-72s"
_CASCADE_RECORDING_ID = "musdb18-hq/Hollow Ground - Ill Fate"
_CASCADE_CATEGORY = "q40-deux-counterfactual-half"
_CASCADE_FILE_SHA256 = {
    "mixture": "734b9aa62075732cc19e1eec8918e9b6f8e28cfe03340506e2f2d7ac28d46d7d",
    "Vocals": "81e7f9322bfdb71bc68cadef053505ad611374373d26a4633790969e6a15b53b",
    "Instrumental": "6cc635e43829471b3fdb81eca38c0971b8c91b3fd9b08776cb73886322ae46e7",
}


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
    parser.add_argument(
        "--rate-arm",
        choices=("delivery", "native"),
        default=None,
        help="Q20 experiment arm: infer at delivery or model-native rate.",
    )
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
    parser.add_argument(
        "--extra-origin-samples",
        type=_positive_int,
        default=None,
        metavar="N",
        help="Q30: add one zero-padded origin offset by N mixture samples.",
    )
    parser.add_argument(
        "--cascade-vocal-repair",
        action="store_true",
        help="Q40: run the frozen direct-Deux counterfactual vocal recipe.",
    )
    return parser


def _validate_cascade_args(
    args: argparse.Namespace, parser: argparse.ArgumentParser
) -> None:
    if not args.cascade_vocal_repair:
        return
    checks = (
        (
            not args.retain_stems,
            "--cascade-vocal-repair requires --retain-stems",
        ),
        (
            args.variant != "real-model",
            "--cascade-vocal-repair requires --variant real-model",
        ),
        (
            args.model != "becruily_deux.ckpt",
            "--cascade-vocal-repair requires --model becruily_deux.ckpt",
        ),
        (
            args.sample_rate != 44_100,
            "--cascade-vocal-repair requires --sample-rate 44100",
        ),
        (
            args.batch_size not in (None, 1),
            "--cascade-vocal-repair requires batch size 1",
        ),
        (args.overlap not in (None, 2), "--cascade-vocal-repair requires overlap 2"),
        (
            args.segment_size is not None,
            "--cascade-vocal-repair requires the default segment size",
        ),
        (
            args.chunk_duration_s is not None,
            "--cascade-vocal-repair forbids outer chunking",
        ),
        (args.tta, "--cascade-vocal-repair forbids TTA"),
        (args.pitch_shift is not None, "--cascade-vocal-repair forbids pitch shift"),
        (args.rate_arm is not None, "--cascade-vocal-repair forbids rate arms"),
        (args.stem_ensemble, "--cascade-vocal-repair forbids stem ensembles"),
        (
            args.extra_origin_samples is not None,
            "--cascade-vocal-repair cannot be combined with --extra-origin-samples",
        ),
    )
    for invalid, message in checks:
        if invalid:
            parser.error(message)


def _sha256_file(path: Path) -> str:
    return hashlib.sha256(path.read_bytes()).hexdigest()


def _validate_frozen_cascade_corpus(
    corpus: ReferenceCorpus, corpus_dir: Path, parser: argparse.ArgumentParser
) -> None:
    """Require the single licensed tuning item frozen by Q40."""
    manifest = corpus_dir / "corpus.json"
    try:
        manifest_hash = _sha256_file(manifest)
    except OSError as exc:
        parser.error(f"--cascade-vocal-repair requires frozen corpus.json: {exc}")
    if manifest_hash != _CASCADE_CORPUS_SHA256:
        parser.error(
            "--cascade-vocal-repair requires frozen corpus.json "
            f"SHA256 {_CASCADE_CORPUS_SHA256}"
        )
    if corpus.corpus_id != _CASCADE_CORPUS_ID:
        parser.error(
            f"--cascade-vocal-repair requires corpus ID {_CASCADE_CORPUS_ID!r}"
        )
    if len(corpus.items) != 1:
        parser.error("--cascade-vocal-repair requires exactly one tuning item")
    item = corpus.items[0]
    for field, expected in (
        ("item_id", _CASCADE_ITEM_ID),
        ("recording_id", _CASCADE_RECORDING_ID),
        ("category", _CASCADE_CATEGORY),
        ("split", "tuning"),
    ):
        if getattr(item, field) != expected:
            parser.error(
                f"--cascade-vocal-repair requires frozen item {field}={expected!r}"
            )
    if set(item.stems) != {"Vocals", "Instrumental"}:
        parser.error(
            "--cascade-vocal-repair requires Vocals and Instrumental references"
        )
    if item.estimate_stems != {"Instrumental": ("_deux_inst",)}:
        parser.error(
            "--cascade-vocal-repair requires Instrumental mapped to _deux_inst"
        )
    paths = {"mixture": item.mixture, **item.stems}
    for name, expected_hash in _CASCADE_FILE_SHA256.items():
        path = Path(paths[name])
        try:
            info = sf.info(str(path))
            actual_hash = _sha256_file(path)
        except (OSError, RuntimeError) as exc:
            parser.error(f"--cascade-vocal-repair cannot read frozen {name}: {exc}")
        if actual_hash != expected_hash:
            parser.error(f"--cascade-vocal-repair requires frozen {name} content")
        if (info.samplerate, info.channels, info.frames) != (44_100, 2, 529_200):
            parser.error(
                f"--cascade-vocal-repair requires {name} at 44100 Hz stereo/529200 frames"
            )


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
    if args.cascade_vocal_repair:
        return partial(
            separate_with_deux_cascade,
            separate_fn=partial(
                separate_for_eval,
                model="becruily_deux.ckpt",
                sample_rate=44_100,
                batch_size=1,
                segment_size=None,
                chunk_duration_s=None,
                overlap=2,
                stem_ensemble=False,
                tta=False,
                pitch_shift=None,
            ),
        )
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
        if args.rate_arm is not None:
            return partial(
                separate_tree_for_rate_experiment,
                delivery_sample_rate=args.sample_rate,
                config=config,
                rate_arm=args.rate_arm,
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
    if args.rate_arm is not None:
        separate = partial(
            separate_model_for_rate_experiment,
            delivery_sample_rate=args.sample_rate,
            model=args.model or DEFAULT_MODEL,
            rate_arm=args.rate_arm,
            batch_size=args.batch_size,
            segment_size=args.segment_size,
            chunk_duration_s=args.chunk_duration_s,
            overlap=args.overlap,
            tta=args.tta,
            pitch_shift=args.pitch_shift,
        )
    else:
        separate = partial(separate_for_eval, **options)
    if args.extra_origin_samples is not None:
        return partial(
            separate_with_extra_origin,
            separate_fn=separate,
            origin_samples=args.extra_origin_samples,
        )
    return separate


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


def main(argv: Sequence[str] | None = None) -> int:
    parser = _parser()
    args = parser.parse_args(argv)
    if args.variant == "production-tree" and args.model is not None:
        parser.error(
            "production-tree does not accept --model; the plan owns model selection"
        )
    if (
        args.variant == "real-model"
        and args.rate_arm is not None
        and args.stem_ensemble
    ):
        parser.error(
            "--rate-arm with --stem-ensemble requires --variant production-tree"
        )
    if args.stems is not None and args.variant != "production-tree":
        parser.error("--stems requires --variant production-tree")
    if args.extra_origin_samples is not None and args.variant != "real-model":
        parser.error("--extra-origin-samples requires --variant real-model")
    if args.extra_origin_samples is not None and args.stem_ensemble:
        parser.error("--extra-origin-samples is for direct model evaluation")
    if args.extra_origin_samples is not None and args.rate_arm is not None:
        parser.error("--extra-origin-samples cannot be combined with --rate-arm")
    if args.extra_origin_samples is not None and (
        args.tta or args.pitch_shift is not None
    ):
        parser.error(
            "--extra-origin-samples cannot be combined with --tta or --pitch-shift"
        )
    _validate_cascade_args(args, parser)
    if args.variant == "synthetic-reference":
        if (
            any(
                value is not None
                for value in (
                    args.model,
                    args.batch_size,
                    args.segment_size,
                    args.chunk_duration_s,
                    args.overlap,
                    args.pitch_shift,
                )
            )
            or args.stem_ensemble
            or args.tta
        ):
            parser.error("model settings require --variant real-model")
        if args.rate_arm is not None:
            parser.error("--rate-arm requires --variant real-model")
        if args.retain_stems:
            parser.error(
                "--retain-stems requires --variant real-model or production-tree"
            )
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
    if args.cascade_vocal_repair:
        _validate_frozen_cascade_corpus(corpus, Path(args.corpus), parser)
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
        protocol_id=(CASCADE_RECIPE if args.cascade_vocal_repair else _PROTOCOL_ID),
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

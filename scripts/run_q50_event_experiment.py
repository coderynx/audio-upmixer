#!/usr/bin/env python3
"""Run the Q50 aggregate-Drums tuning-only pre-screen.

The experiment deliberately reuses retained Q03 output instead of running a
separator.  It is therefore a small, reproducible post-pass screen, not a
production or held-out quality result.
"""

from __future__ import annotations

import argparse
import hashlib
import json
import time
from dataclasses import fields, replace
from pathlib import Path
from typing import Any, Sequence

import numpy as np
import soundfile as sf

from upmixer.eval import (
    ReferenceCorpus,
    RunSettings,
    evaluate_corpus,
    format_report,
    repair_aggregate_drums,
)
from upmixer.eval.report import EvalReport
from upmixer.separation.separator import SeparationSettings

_DEFAULT_Q03_DIR = Path(
    "/Volumes/External SSD/upmixer-eval/separation-quality/"
    "q03/excerpts-60s-72s/tuning-44k-7a2806c"
)
_DEFAULT_CORPUS_DIR = Path(
    "/Volumes/External SSD/upmixer-datasets/musdb18-hq/excerpts-60s-72s/44.1k/tuning"
)
_DEFAULT_OUTPUT_DIR = Path(
    "/Volumes/External SSD/upmixer-eval/separation-quality/"
    "q50/event-repair-v1/prescreen-12s"
)
_PROTOCOL_ID = "q50-aggregate-drums-event-repetition-v1-prescreen-12s"
_EXPECTED_CORPUS_ID = (
    "upmixer-musdb18hq-v1-c5ba6b34513f-q03-q20-excerpt-60s-72s-44.1k-tuning"
)
_EXPECTED_Q03_CODE_REVISION = "7a2806cf5fa7f0d8a84af3b8139b92c68a8c242b"
_SAMPLE_RATE = 44_100
_FRAME_COUNT = 529_200
_PRIMARY_SIBLINGS = ("Bass", "Drums", "Guitar", "Piano", "Other")
_SUPPORTED_TARGETS = ("Bass", "Drums", "Other")
_Q03_OUTPUT_STEMS = (*_PRIMARY_SIBLINGS, "Vocals")
_REFERENCE_STEMS = {"Bass", "Drums", "Other", "Vocals"}
_OTHER_COMPONENTS = ("Guitar", "Piano", "Other")
_DRUMS_SDR_FLOOR_DB = 0.2
_METRIC_WIN_FLOOR = 0.01
_SIBLING_SDR_FLOOR_DB = -0.1
_MATERIAL_METRIC_FLOOR = -0.02
_CONSERVATION_LIMIT = 1e-6


def _sha256(path: Path) -> str:
    digest = hashlib.sha256()
    with path.open("rb") as source:
        for chunk in iter(lambda: source.read(1024 * 1024), b""):
            digest.update(chunk)
    return digest.hexdigest()


def _read_json(path: Path) -> dict[str, Any]:
    try:
        value = json.loads(path.read_text(encoding="utf-8"))
    except (OSError, json.JSONDecodeError) as exc:
        raise ValueError(f"cannot read JSON {path}: {exc}") from exc
    if not isinstance(value, dict):
        raise ValueError(f"JSON root must be an object: {path}")
    return value


def _path_under(root: Path, relative: str, label: str) -> Path:
    path = (root / relative).resolve()
    try:
        path.relative_to(root.resolve())
    except ValueError as exc:
        raise ValueError(f"{label} escapes its artifact root: {relative}") from exc
    if not path.is_file():
        raise ValueError(f"{label} does not exist: {path}")
    return path


def _validate_audio(path: Path, label: str) -> tuple[np.ndarray, dict[str, Any]]:
    try:
        info = sf.info(str(path))
        audio, sample_rate = sf.read(str(path), dtype="float32", always_2d=True)
    except (OSError, RuntimeError) as exc:
        raise ValueError(f"cannot read {label} {path}: {exc}") from exc
    metadata = {
        "path": str(path),
        "sha256": _sha256(path),
        "sample_rate": int(sample_rate),
        "channels": int(info.channels),
        "frames": int(info.frames),
        "subtype": info.subtype,
        "dtype": str(audio.dtype),
        "finite": bool(np.all(np.isfinite(audio))),
    }
    if (
        metadata["sample_rate"],
        metadata["channels"],
        metadata["frames"],
    ) != (_SAMPLE_RATE, 2, _FRAME_COUNT):
        raise ValueError(
            f"{label} must be {_SAMPLE_RATE} Hz stereo/{_FRAME_COUNT} frames: {path}"
        )
    if not metadata["finite"] or audio.dtype != np.dtype("float32"):
        raise ValueError(f"{label} must be finite float32 audio: {path}")
    return audio, metadata


def _stage_settings(raw: list[dict[str, Any]]) -> tuple[SeparationSettings, ...]:
    stages = []
    for stage in raw:
        stages.append(
            SeparationSettings(
                model=stage["model"],
                sample_rate=int(stage["sample_rate"]),
                batch_size=int(stage["batch_size"]),
                segment_size=stage.get("segment_size"),
                chunk_duration_s=stage.get("chunk_duration_s"),
                overlap=stage.get("overlap"),
                tta=bool(stage.get("tta", False)),
                pitch_shift=stage.get("pitch_shift"),
                backend=stage["backend"],
                model_arch=stage.get("model_arch"),
                model_config_name=stage.get("model_config_name"),
                model_native_sample_rate=stage.get("model_native_sample_rate"),
                device=stage.get("device"),
                checkpoint_sha256=stage.get("checkpoint_sha256"),
                model_config_sha256=stage.get("model_config_sha256"),
                runtime_precision=stage.get("runtime_precision"),
                normalization_policy=stage.get("normalization_policy"),
                oom_fallback_attempts=tuple(stage.get("oom_fallback_attempts") or ()),
                oom_fallback_count=int(stage.get("oom_fallback_count", 0)),
            )
        )
    return tuple(stages)


def _run_settings(raw: dict[str, Any]) -> RunSettings:
    values = {field.name: raw.get(field.name) for field in fields(RunSettings)}
    values.update(
        model=raw["model"],
        sample_rate=int(raw["sample_rate"]),
        ensemble_models=(
            tuple(raw["ensemble_models"])
            if raw.get("ensemble_models") is not None
            else None
        ),
        stage_settings=_stage_settings(raw.get("stage_settings") or []),
    )
    return RunSettings(**values)


def _validate_provenance(
    corpus_dir: Path,
    q03_dir: Path,
) -> tuple[ReferenceCorpus, dict[str, dict[str, np.ndarray]], dict[str, Any]]:
    """Load only the frozen tuning corpus and retained Q03 outputs."""
    corpus_manifest = corpus_dir / "corpus.json"
    corpus_provenance = corpus_dir / "provenance.json"
    q03_report_path = q03_dir / "report.json"
    q03_index_path = q03_dir / "stems" / "index.json"
    corpus = ReferenceCorpus.from_dir(str(corpus_dir))
    q03_report = _read_json(q03_report_path)
    q03_index = _read_json(q03_index_path)
    source_provenance = _read_json(corpus_provenance)

    if corpus.corpus_id != _EXPECTED_CORPUS_ID:
        raise ValueError(f"unexpected corpus ID: {corpus.corpus_id!r}")
    if q03_report.get("corpus_id") != _EXPECTED_CORPUS_ID:
        raise ValueError("Q03 report corpus ID does not match the tuning corpus")
    if q03_report.get("code_revision") != _EXPECTED_Q03_CODE_REVISION:
        raise ValueError("Q03 report is not the retained 7a2806c incumbent")
    q03_settings = q03_report.get("settings") or {}
    if (
        q03_settings.get("model") != "production-tree"
        or q03_settings.get("sample_rate") != _SAMPLE_RATE
        or q03_settings.get("overlap") != 2
        or q03_settings.get("batch_size") != 1
    ):
        raise ValueError("Q03 report settings do not match the frozen incumbent")
    if q03_index.get("schema_version") != 1:
        raise ValueError("unsupported Q03 stem index schema")
    index_items = q03_index.get("items")
    if not isinstance(index_items, list) or len(index_items) != 12:
        raise ValueError("Q03 stem index must contain exactly 12 items")
    if len(corpus.items) != 12:
        raise ValueError("tuning corpus must contain exactly 12 items")

    provenance_checks = source_provenance.get("checks") or {}
    if (
        provenance_checks.get("expected_recordings") != 12
        or provenance_checks.get("actual_recordings") != 12
        or provenance_checks.get("heldout_inference") is not False
        or not provenance_checks.get("all_sha256_checked")
        or not provenance_checks.get("all_frame_counts_checked")
        or not provenance_checks.get("all_finite_stereo_float32")
    ):
        raise ValueError("source provenance does not certify the tuning corpus")
    if source_provenance.get("heldout_inference") != "not run":
        raise ValueError("source provenance does not certify heldout as not run")
    if source_provenance.get("corpus_id") != _EXPECTED_CORPUS_ID:
        raise ValueError("source provenance corpus ID does not match")

    arrays: dict[str, dict[str, np.ndarray]] = {}
    output_metadata: dict[str, dict[str, Any]] = {}
    expected_ids = []
    for index, (item, indexed) in enumerate(
        zip(corpus.items, index_items, strict=True)
    ):
        if indexed.get("index") != index:
            raise ValueError(f"Q03 stem index order mismatch at item {index}")
        for field in ("recording_id", "item_id", "split", "category"):
            if indexed.get(field) != getattr(item, field):
                raise ValueError(f"Q03/source {field} mismatch at item {index}")
        if item.split != "tuning" or "heldout" in str(item.item_id).lower():
            raise ValueError("Q50 pre-screen refuses non-tuning or heldout items")
        if set(item.stems) != _REFERENCE_STEMS:
            raise ValueError(f"unsupported reference mapping at item {index}")
        if item.estimate_stems != {"Other": _OTHER_COMPONENTS}:
            raise ValueError(f"unsupported Other mapping at item {index}")
        indexed_stems = indexed.get("stems")
        if set(indexed_stems or ()) != set(_Q03_OUTPUT_STEMS):
            raise ValueError(f"Q03 output stem mapping mismatch at item {index}")

        for stem_name, reference_path in item.stems.items():
            _validate_audio(Path(reference_path), f"reference {stem_name}")
        item_arrays: dict[str, np.ndarray] = {}
        item_metadata: dict[str, Any] = {}
        for stem_name in _Q03_OUTPUT_STEMS:
            relative = indexed_stems[stem_name]
            output_path = _path_under(q03_dir, relative, f"Q03 {stem_name}")
            audio, metadata = _validate_audio(output_path, f"Q03 {stem_name}")
            item_arrays[stem_name] = audio
            item_metadata[stem_name] = metadata
        arrays[item.mixture] = item_arrays
        output_metadata[item.item_id] = item_metadata
        expected_ids.append(item.item_id)

    report_ids = [row.get("item_id") for row in q03_report.get("item_settings", [])]
    if report_ids != expected_ids:
        raise ValueError("Q03 report item order does not match the stem index")
    if len(q03_report.get("scores") or []) != 48:
        raise ValueError("Q03 report must contain 48 scored rows")
    if any(row.get("split") != "tuning" for row in q03_report["scores"]):
        raise ValueError("Q03 report contains a non-tuning score")

    metadata = {
        "corpus_manifest": {
            "path": str(corpus_manifest),
            "sha256": _sha256(corpus_manifest),
        },
        "corpus_provenance": {
            "path": str(corpus_provenance),
            "sha256": _sha256(corpus_provenance),
        },
        "q03_report": {
            "path": str(q03_report_path),
            "sha256": _sha256(q03_report_path),
            "code_revision": q03_report["code_revision"],
            "settings": q03_settings,
        },
        "q03_stem_index": {
            "path": str(q03_index_path),
            "sha256": _sha256(q03_index_path),
        },
        "q03_outputs": output_metadata,
        "heldout_inference": False,
    }
    return corpus, arrays, metadata


def _scoring_corpus(corpus: ReferenceCorpus) -> ReferenceCorpus:
    return ReferenceCorpus(
        corpus_id=corpus.corpus_id,
        items=[
            replace(
                item,
                stems={name: item.stems[name] for name in _SUPPORTED_TARGETS},
                estimate_stems={"Other": _OTHER_COMPONENTS},
                unavailable_stems=(),
            )
            for item in corpus.items
        ],
    )


def _evaluate(
    corpus: ReferenceCorpus,
    arrays: dict[str, dict[str, np.ndarray]],
    settings: RunSettings,
    arm_id: str,
    code_revision: str,
) -> EvalReport:
    def separate(mixture: str) -> tuple[dict[str, np.ndarray], RunSettings]:
        return arrays[mixture], settings

    return evaluate_corpus(
        corpus,
        separate,
        sample_rate=_SAMPLE_RATE,
        protocol_id=f"{_PROTOCOL_ID}:{arm_id}",
        code_revision=code_revision,
    )


def _repair(
    arrays: dict[str, dict[str, np.ndarray]],
    recording_ids: dict[str, str],
) -> tuple[dict[str, dict[str, np.ndarray]], list[dict[str, Any]]]:
    repaired: dict[str, dict[str, np.ndarray]] = {}
    accounting: list[dict[str, Any]] = []
    for mixture, stems in arrays.items():
        primary = {name: stems[name] for name in _PRIMARY_SIBLINGS}
        parent = sum(primary.values(), np.zeros_like(primary["Drums"]))
        started = time.perf_counter()
        result = repair_aggregate_drums(parent, primary, _SAMPLE_RATE)
        runtime_s = time.perf_counter() - started
        candidate_parent = sum(result.stems.values(), np.zeros_like(parent))
        conservation_error = float(
            np.max(
                np.abs(
                    np.asarray(candidate_parent, dtype=np.float64)
                    - np.asarray(parent, dtype=np.float64)
                )
            )
        )
        repaired[mixture] = {**result.stems, "Vocals": stems["Vocals"]}
        accounting.append(
            {
                "mixture": mixture,
                "recording_id": recording_ids[mixture],
                "transfer_count": len(result.transfers),
                "transfers": [
                    {
                        "donor": transfer.donor,
                        "donor_frame": transfer.donor_frame,
                        "reference_frame": transfer.reference_frame,
                        "sample_start": transfer.sample_start,
                        "sample_end": transfer.sample_end,
                        "gain": transfer.gain,
                        "score": transfer.score,
                    }
                    for transfer in result.transfers
                ],
                "no_op": not result.transfers,
                "post_pass_runtime_s": runtime_s,
                "conservation_max_abs_error": conservation_error,
            }
        )
    return repaired, accounting


def _medians(
    bootstrap: dict[str, Any],
) -> dict[str, dict[str, float]]:
    values: dict[str, dict[str, list[float]]] = {}
    for row in bootstrap["pairs"]:
        stem = row["stem"]
        values.setdefault(
            stem, {metric: [] for metric in ("sdr", "fullness", "bleedless")}
        )
        for metric in values[stem]:
            values[stem][metric].append(float(row["delta"][metric]))
    return {
        stem: {metric: float(np.median(numbers)) for metric, numbers in metrics.items()}
        for stem, metrics in values.items()
    }


def _gate(
    baseline: EvalReport,
    candidate: EvalReport,
    accounting: list[dict[str, Any]],
    bootstrap: dict[str, Any],
) -> dict[str, Any]:
    medians = _medians(bootstrap)
    drums = medians["Drums"]
    target_quality_win = (
        drums["sdr"] >= _DRUMS_SDR_FLOOR_DB
        or drums["fullness"] >= _METRIC_WIN_FLOOR
        or drums["bleedless"] >= _METRIC_WIN_FLOOR
    )
    sibling_safety = {
        stem: {
            "sdr_median_delta_db": medians[stem]["sdr"],
            "sdr_pass": medians[stem]["sdr"] >= _SIBLING_SDR_FLOOR_DB,
            "fullness_median_delta": medians[stem]["fullness"],
            "bleedless_median_delta": medians[stem]["bleedless"],
            "metrics_pass": medians[stem]["fullness"] >= _MATERIAL_METRIC_FLOOR
            and medians[stem]["bleedless"] >= _MATERIAL_METRIC_FLOOR,
        }
        for stem in _SUPPORTED_TARGETS
    }
    conservation = {
        "max_abs_error": max(row["conservation_max_abs_error"] for row in accounting),
        "limit": _CONSERVATION_LIMIT,
        "pass": all(
            row["conservation_max_abs_error"] <= _CONSERVATION_LIMIT
            for row in accounting
        ),
    }
    safety_pass = all(
        row["sdr_pass"] and row["metrics_pass"] for row in sibling_safety.values()
    )
    coverage_pass = all(
        row.status == "scored" for row in (*baseline.coverage, *candidate.coverage)
    )
    material_pass = bool(
        target_quality_win and safety_pass and conservation["pass"] and coverage_pass
    )
    return {
        "name": "Q50 material tuning pre-screen",
        "status": "PASS" if material_pass else "FAIL",
        "promotion_eligible": False,
        "target": {
            "stem": "Drums",
            "median_deltas": drums,
            "sdr_floor_db": _DRUMS_SDR_FLOOR_DB,
            "metric_win_floor": _METRIC_WIN_FLOOR,
            "quality_win_pass": target_quality_win,
            "fullness_regression_guard": drums["fullness"] >= _MATERIAL_METRIC_FLOOR,
        },
        "supported_sibling_safety": sibling_safety,
        "conservation": conservation,
        "coverage_pass": coverage_pass,
        "transfer_count": int(sum(row["transfer_count"] for row in accounting)),
        "noop_count": int(sum(row["no_op"] for row in accounting)),
        "next_action": (
            "Do not run 60s inference; reject this Q50 candidate at the material gate."
            if not material_pass
            else "No 60s inference in this script; held-out, listening, and Q100 gates remain."
        ),
    }


def _runtime_summary(accounting: list[dict[str, Any]]) -> dict[str, float | int]:
    values = np.asarray(
        [row["post_pass_runtime_s"] for row in accounting], dtype=np.float64
    )
    return {
        "items": int(values.size),
        "total_s": float(np.sum(values)),
        "mean_s": float(np.mean(values)),
        "median_s": float(np.median(values)),
        "min_s": float(np.min(values)),
        "max_s": float(np.max(values)),
    }


def _write_hashes(output_dir: Path) -> None:
    paths = sorted(
        path
        for path in output_dir.rglob("*")
        if path.is_file() and path.name != "SHA256SUMS"
    )
    text = "".join(
        f"{_sha256(path)}  {path.relative_to(output_dir)}\n" for path in paths
    )
    (output_dir / "SHA256SUMS").write_text(text, encoding="utf-8")


def _parser() -> argparse.ArgumentParser:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--q03-dir", type=Path, default=_DEFAULT_Q03_DIR)
    parser.add_argument("--corpus-dir", type=Path, default=_DEFAULT_CORPUS_DIR)
    parser.add_argument("--output-dir", type=Path, default=_DEFAULT_OUTPUT_DIR)
    parser.add_argument("--bootstrap-resamples", type=int, default=10_000)
    parser.add_argument("--bootstrap-seed", type=int, default=0)
    return parser


def main(argv: Sequence[str] | None = None) -> int:
    args = _parser().parse_args(argv)
    if args.bootstrap_resamples < 1 or args.bootstrap_seed < 0:
        raise SystemExit("bootstrap arguments must be non-negative; resamples >= 1")
    if args.output_dir.exists() and any(args.output_dir.iterdir()):
        raise SystemExit(f"output directory must be fresh and empty: {args.output_dir}")
    args.output_dir.mkdir(parents=True, exist_ok=True)

    corpus, incumbent_arrays, provenance = _validate_provenance(
        args.corpus_dir, args.q03_dir
    )
    scoring_corpus = _scoring_corpus(corpus)
    settings = _run_settings(provenance["q03_report"]["settings"])
    recording_ids = {item.mixture: item.recording_id for item in corpus.items}
    incumbent_report = _evaluate(
        scoring_corpus,
        incumbent_arrays,
        settings,
        "incumbent-q03-retained",
        _EXPECTED_Q03_CODE_REVISION,
    )
    candidate_arrays, accounting = _repair(incumbent_arrays, recording_ids)
    candidate_report = _evaluate(
        scoring_corpus,
        candidate_arrays,
        settings,
        "candidate-q50-event-repair",
        "94bb197",
    )
    bootstrap = incumbent_report.paired_bootstrap(
        candidate_report,
        n_resamples=args.bootstrap_resamples,
        seed=args.bootstrap_seed,
    )
    gate = _gate(incumbent_report, candidate_report, accounting, bootstrap)
    donor_counts: dict[str, int] = {}
    for row in accounting:
        for transfer in row["transfers"]:
            donor_counts[transfer["donor"]] = donor_counts.get(transfer["donor"], 0) + 1

    payload = {
        "schema_version": 1,
        "protocol_id": _PROTOCOL_ID,
        "experiment_type": "tuning pre-screen",
        "target_commit": "94bb197",
        "limitations": [
            "12-second tuning context only; no held-out inference was read or run.",
            "Primary parent is the approximate sum Bass+Drums+Guitar+Piano+Other of retained Q03 estimates.",
            "Incumbent stems are retained from the older Q03 revision 7a2806c, not rerun at target commit 94bb197.",
            "Guitar and Piano have no separate references; only supported Bass, Drums, and aggregate Other targets are scored.",
            "This material screen does not establish production promotion, listening acceptance, or Q100 completion.",
        ],
        "settings": {
            "sample_rate": _SAMPLE_RATE,
            "frames": _FRAME_COUNT,
            "duration_s": _FRAME_COUNT / _SAMPLE_RATE,
            "bootstrap_resamples": args.bootstrap_resamples,
            "bootstrap_seed": args.bootstrap_seed,
        },
        "provenance": provenance,
        "repair_accounting": {
            "items": accounting,
            "donor_transfer_counts": donor_counts,
            "runtime": _runtime_summary(accounting),
        },
        "gate": gate,
        "incumbent_report": incumbent_report.to_dict(),
        "candidate_report": candidate_report.to_dict(),
        "paired_bootstrap": bootstrap,
    }
    report_path = args.output_dir / "report.json"
    report_path.write_text(
        json.dumps(payload, indent=2, allow_nan=False) + "\n", encoding="utf-8"
    )
    report_text = "\n".join(
        [
            "Q50 aggregate-Drums event/repetition repair — tuning pre-screen",
            "Material gate: " + gate["status"],
            gate["next_action"],
            "",
            "Limitations:",
            *[f"- {limitation}" for limitation in payload["limitations"]],
            "",
            "Repair accounting:",
            f"- transfers={gate['transfer_count']} no_op_items={gate['noop_count']} "
            f"runtime_median_s={payload['repair_accounting']['runtime']['median_s']:.6f}",
            f"- donor_transfer_counts={json.dumps(donor_counts, sort_keys=True)}",
            "",
            "Incumbent (retained Q03):",
            format_report(incumbent_report),
            "",
            "Candidate (Q50 repair):",
            format_report(candidate_report),
            "",
            "Paired bootstrap recording-level deltas (candidate - incumbent):",
            json.dumps(bootstrap["confidence_intervals"], indent=2, sort_keys=True),
        ]
    )
    (args.output_dir / "report.txt").write_text(report_text + "\n", encoding="utf-8")
    _write_hashes(args.output_dir)
    print(report_text)
    print(f"\nWrote {report_path}")
    print(f"Wrote {args.output_dir / 'report.txt'}")
    print(f"Wrote {args.output_dir / 'SHA256SUMS'}")
    return 0 if gate["status"] == "PASS" else 1


if __name__ == "__main__":
    raise SystemExit(main())

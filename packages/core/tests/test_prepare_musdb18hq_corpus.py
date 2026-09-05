import hashlib
import importlib.util
import json
from pathlib import Path

import numpy as np
import pytest
import soundfile as sf

from upmixer.eval.corpus import ReferenceCorpus


def _load_preparer():
    path = (
        Path(__file__).resolve().parents[3] / "scripts" / "prepare_musdb18hq_corpus.py"
    )
    spec = importlib.util.spec_from_file_location("upmixer_musdb18hq_preparer", path)
    assert spec is not None and spec.loader is not None
    module = importlib.util.module_from_spec(spec)
    spec.loader.exec_module(module)
    return module


PREPARER = _load_preparer()


def _write_dataset(
    root: Path,
    *,
    tuning_count: int = 12,
    heldout_count: int = 12,
    nested: bool = True,
    residual: float = 0.0,
    invalid: str | None = None,
) -> Path:
    frames = 16
    stems = ("vocals", "bass", "drums", "other")
    for split, count, offset in (
        ("tuning", tuning_count, 0),
        ("heldout", heldout_count, 100),
    ):
        split_root = root / "subset" / split if nested else root / split
        for index in range(count):
            track_root = split_root / f"track-{offset + index:02d}"
            track_root.mkdir(parents=True, exist_ok=True)
            signals = {
                stem: np.full(
                    (frames, 2),
                    value + (offset + index) * 0.001,
                    dtype=np.float32,
                )
                for stem, value in zip(stems, (0.1, 0.2, -0.05, 0.03))
            }
            mixture = np.zeros((frames, 2), dtype=np.float32)
            for signal in signals.values():
                mixture += signal
            mixture[0, 0] += residual
            for stem, signal in signals.items():
                sample_rate = 48_000 if invalid == "rate" and stem == "bass" else 44_100
                if invalid == "shape" and stem == "drums":
                    signal = signal[:, :1]
                sf.write(
                    track_root / f"{stem}.wav", signal, sample_rate, subtype="FLOAT"
                )
            sf.write(track_root / "mixture.wav", mixture, 44_100, subtype="FLOAT")
    if invalid == "missing":
        missing = root / "subset" / "tuning" / "track-00" / "other.wav"
        missing.unlink()
    return root


def test_prepare_writes_reloadable_manifest_with_stable_relative_items(tmp_path):
    dataset = _write_dataset(tmp_path / "dataset")
    output = tmp_path / "prepared"

    corpus_path, provenance_path = PREPARER.prepare_corpus(dataset, output)

    manifest = json.loads(corpus_path.read_text(encoding="utf-8"))
    assert manifest["schema_version"] == 1
    assert manifest["corpus_id"] == "musdb18-hq-v1"
    assert len(manifest["items"]) == 24
    assert {item["split"] for item in manifest["items"]} == {"tuning", "heldout"}
    assert set(manifest["items"][0]["stems"]) == {"Vocals", "Bass", "Drums", "Other"}
    assert all(
        not Path(path).is_absolute()
        for item in manifest["items"]
        for path in [item["mixture"], *item["stems"].values()]
    )
    assert not list(output.rglob("*.wav"))

    corpus = ReferenceCorpus.from_dir(str(output))
    assert corpus.corpus_id == "musdb18-hq-v1"
    assert len(corpus.items) == 24
    assert all(Path(item.mixture).is_file() for item in corpus.items)
    assert all(
        set(item.stems) == {"Vocals", "Bass", "Drums", "Other"} for item in corpus.items
    )

    provenance = json.loads(provenance_path.read_text(encoding="utf-8"))
    assert provenance["dataset"]["doi"] == "10.5281/zenodo.3338373"
    assert provenance["dataset"]["license"] == "educational-only"
    assert (
        provenance["dataset"]["archive_identity"] == "MUSDB18-HQ@10.5281/zenodo.3338373"
    )
    assert provenance["splits"] == {
        "heldout": {"n_recordings": 12},
        "tuning": {"n_recordings": 12},
    }
    assert provenance["recordings"][0]["category"] == "musdb18-hq-baseline"
    assert provenance["recordings"][0]["residual"]["peak"] == pytest.approx(
        0.0, abs=1e-7
    )

    second_output = tmp_path / "prepared-second"
    PREPARER.prepare_corpus(dataset, second_output)
    first_ids = [(item["recording_id"], item["item_id"]) for item in manifest["items"]]
    second_ids = [
        (item["recording_id"], item["item_id"])
        for item in json.loads((second_output / "corpus.json").read_text())["items"]
    ]
    assert first_ids == second_ids


def test_prepare_records_hash_audio_metadata_and_nonzero_residual(tmp_path):
    dataset = _write_dataset(tmp_path / "dataset", residual=0.125)
    output = tmp_path / "prepared"

    PREPARER.prepare_corpus(dataset, output)

    provenance = json.loads((output / "provenance.json").read_text(encoding="utf-8"))
    record = next(
        item for item in provenance["recordings"] if item["split"] == "tuning"
    )
    file_path = dataset / "subset" / "tuning" / "track-00" / "vocals.wav"
    metadata = record["files"]["Vocals"]
    assert metadata["sha256"] == hashlib.sha256(file_path.read_bytes()).hexdigest()
    assert metadata["bytes"] == file_path.stat().st_size
    assert metadata["sample_rate"] == 44_100
    assert metadata["channels"] == 2
    assert metadata["frames"] == 16
    assert metadata["duration_s"] == pytest.approx(16 / 44_100)
    assert record["residual"]["block_size"] == PREPARER.BLOCK_SIZE
    assert record["residual"]["peak"] == pytest.approx(0.125, abs=1e-7)
    assert record["residual"]["rms"] == pytest.approx(0.125 / np.sqrt(16 * 2), abs=1e-7)


def test_prepare_requires_independent_tracks_per_split(tmp_path):
    dataset = _write_dataset(tmp_path / "dataset", tuning_count=11)

    with pytest.raises(
        ValueError, match="tuning requires at least 12 independent tracks"
    ):
        PREPARER.prepare_corpus(dataset, tmp_path / "prepared")


@pytest.mark.parametrize("invalid", ("missing", "rate", "shape"))
def test_prepare_rejects_invalid_required_audio(tmp_path, invalid):
    dataset = _write_dataset(tmp_path / "dataset", invalid=invalid)

    with pytest.raises(ValueError):
        PREPARER.prepare_corpus(dataset, tmp_path / "prepared")

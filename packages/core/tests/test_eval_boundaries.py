import json

import numpy as np
import pytest
import soundfile as sf

import upmixer.eval.harness as harness
from upmixer.eval.corpus import CorpusItem, ReferenceCorpus, synthetic_corpus
from upmixer.eval.harness import RunSettings, evaluate_corpus


def _corpus(tmp_path, sample_rate=8000, frames=8, **item_kwargs):
    signal = np.ones((frames, 2), dtype=np.float32)
    mixture = tmp_path / "mix.wav"
    reference = tmp_path / "vocals.wav"
    sf.write(mixture, signal, sample_rate, subtype="FLOAT")
    sf.write(reference, signal, sample_rate, subtype="FLOAT")
    return ReferenceCorpus([
        CorpusItem(
            mixture=str(mixture),
            stems={"Vocals": str(reference)},
            **item_kwargs,
        )
    ])


def _separate(corpus, settings, stems=None):
    def separate(_mixture):
        item = corpus.items[0]
        if stems is None:
            output = {}
            for name, path in item.stems.items():
                output[name], _ = sf.read(path, dtype="float32", always_2d=True)
        else:
            output = stems
        return output, settings

    return separate


def test_synthetic_corpus_has_stable_identity(tmp_path):
    corpus = synthetic_corpus(sample_rate=8000, out_dir=str(tmp_path / "synthetic"))

    assert [(item.recording_id, item.item_id, item.split) for item in corpus.items] == [
        ("synthetic-default", "default", "synthetic"),
        ("synthetic-dense-synth", "dense_synth", "synthetic"),
        ("synthetic-choir-cluster", "choir_cluster", "synthetic"),
    ]


def test_from_dir_loads_identity_fields(tmp_path):
    (tmp_path / "song").mkdir()
    manifest = {
        "items": [{
            "mixture": "song/mix.wav",
            "stems": {"Vocals": "song/vocals.wav"},
            "category": "vocal",
            "recording_id": "rec-01",
            "item_id": "rec-01-full",
            "split": "holdout",
            "unavailable_stems": ["Crowd"],
        }]
    }
    (tmp_path / "corpus.json").write_text(json.dumps(manifest), encoding="utf-8")

    item = ReferenceCorpus.from_dir(str(tmp_path)).items[0]

    assert (item.recording_id, item.item_id, item.split) == (
        "rec-01",
        "rec-01-full",
        "holdout",
    )
    assert item.unavailable_stems == ("Crowd",)


def test_from_dir_loads_reference_target_output_mapping(tmp_path):
    (tmp_path / "song").mkdir()
    manifest = {
        "items": [
            {
                "mixture": "song/mix.wav",
                "stems": {"Other": "song/other.wav"},
                "estimate_stems": {"Other": ["Guitar", "Piano", "Other"]},
            }
        ]
    }
    (tmp_path / "corpus.json").write_text(json.dumps(manifest), encoding="utf-8")

    item = ReferenceCorpus.from_dir(str(tmp_path)).items[0]

    assert item.estimate_stems == {
        "Other": ("Guitar", "Piano", "Other"),
    }


def test_from_dir_loads_item_with_only_unavailable_stems(tmp_path):
    (tmp_path / "song").mkdir()
    manifest = {
        "items": [{
            "mixture": "song/mix.wav",
            "category": "live",
            "recording_id": "rec-01",
            "item_id": "rec-01-live",
            "split": "holdout",
            "unavailable_stems": ["Crowd", "Room"],
        }]
    }
    (tmp_path / "corpus.json").write_text(json.dumps(manifest), encoding="utf-8")

    item = ReferenceCorpus.from_dir(str(tmp_path)).items[0]

    assert item.stems == {}
    assert item.unavailable_stems == ("Crowd", "Room")


def test_evaluate_corpus_propagates_identity_to_score(tmp_path):
    corpus = _corpus(
        tmp_path,
        recording_id="rec-01",
        item_id="rec-01-excerpt-01",
        split="tuning",
    )
    settings = RunSettings(model="identity-test", sample_rate=8000)

    report = evaluate_corpus(corpus, _separate(corpus, settings), sample_rate=8000)

    score = report.scores[0]
    assert (score.recording_id, score.item_id, score.split) == (
        "rec-01",
        "rec-01-excerpt-01",
        "tuning",
    )


def test_evaluate_corpus_sums_mapped_outputs_under_reference_target(tmp_path):
    frames = 16
    sample_rate = 8000
    reference = np.full((frames, 2), 6.0, dtype=np.float32)
    mixture = tmp_path / "mix.wav"
    reference_path = tmp_path / "other.wav"
    sf.write(mixture, reference, sample_rate, subtype="FLOAT")
    sf.write(reference_path, reference, sample_rate, subtype="FLOAT")
    corpus = ReferenceCorpus(
        [
            CorpusItem(
                mixture=str(mixture),
                stems={"Other": str(reference_path)},
                estimate_stems={"Other": ("Guitar", "Piano", "Other")},
            )
        ]
    )
    estimates = {
        "Guitar": np.full((frames, 2), 1.0, dtype=np.float64),
        "Piano": np.full((frames, 2), 2.0, dtype=np.float64),
        "Other": np.full((frames, 2), 3.0, dtype=np.float64),
    }

    report = evaluate_corpus(
        corpus,
        lambda _mixture: (
            estimates,
            RunSettings(model="mapped", sample_rate=sample_rate),
        ),
        sample_rate=sample_rate,
    )

    assert [
        (score.stem, score.fullness, score.bleedless) for score in report.scores
    ] == [("Other", pytest.approx(1.0), pytest.approx(1.0))]
    assert [(row.stem, row.status) for row in report.coverage] == [("Other", "scored")]


def test_evaluate_corpus_mapping_can_exclude_reference_target_name(tmp_path):
    frames = 16
    sample_rate = 8000
    reference = np.full((frames, 2), 3.0, dtype=np.float32)
    mixture = tmp_path / "mix.wav"
    reference_path = tmp_path / "instrumental.wav"
    sf.write(mixture, reference, sample_rate, subtype="FLOAT")
    sf.write(reference_path, reference, sample_rate, subtype="FLOAT")
    corpus = ReferenceCorpus(
        [
            CorpusItem(
                mixture=str(mixture),
                stems={"Instrumental": str(reference_path)},
                estimate_stems={"Instrumental": ("Guitar", "Piano")},
            )
        ]
    )

    report = evaluate_corpus(
        corpus,
        lambda _mixture: (
            {
                "Guitar": np.ones((frames, 2), dtype=np.float32),
                "Piano": np.full((frames, 2), 2.0, dtype=np.float32),
            },
            RunSettings(model="mapped", sample_rate=sample_rate),
        ),
        sample_rate=sample_rate,
    )

    assert [score.stem for score in report.scores] == ["Instrumental"]


def test_evaluate_corpus_reports_scored_and_unavailable_coverage(tmp_path):
    corpus = _corpus(
        tmp_path,
        recording_id="rec-01",
        item_id="rec-01-excerpt-01",
        split="tuning",
        unavailable_stems=("Crowd",),
    )
    settings = RunSettings(model="coverage-test", sample_rate=8000)

    report = evaluate_corpus(corpus, _separate(corpus, settings), sample_rate=8000)

    assert [(row.stem, row.status) for row in report.coverage] == [
        ("Vocals", "scored"),
        ("Crowd", "unavailable"),
    ]
    assert len(report.scores) == 1


def test_evaluate_corpus_reports_item_with_only_unavailable_stems(tmp_path):
    mixture = tmp_path / "mix.wav"
    sf.write(mixture, np.ones((8, 2), dtype=np.float32), 8000, subtype="FLOAT")
    corpus = ReferenceCorpus([
        CorpusItem(
            mixture=str(mixture),
            stems={},
            unavailable_stems=("Crowd",),
            recording_id="rec-01",
            item_id="rec-01-crowd",
            split="holdout",
        )
    ])
    settings = RunSettings(model="coverage-test", sample_rate=8000)

    report = evaluate_corpus(corpus, _separate(corpus, settings), sample_rate=8000)

    assert report.scores == []
    assert len(report.coverage) == 1
    assert report.coverage[0].status == "unavailable"
    assert report.coverage[0].item_id == "rec-01-crowd"


def test_evaluate_corpus_rejects_scored_and_unavailable_overlap(tmp_path):
    corpus = _corpus(tmp_path, unavailable_stems=("Vocals",))
    settings = RunSettings(model="coverage-test", sample_rate=8000)

    with pytest.raises(ValueError, match="both scored and unavailable"):
        evaluate_corpus(corpus, _separate(corpus, settings), sample_rate=8000)


def test_evaluate_corpus_rejects_missing_required_estimate(tmp_path):
    corpus = _corpus(tmp_path)
    settings = RunSettings(model="test", sample_rate=8000)
    estimate = np.ones((8, 2), dtype=np.float32)

    with pytest.raises(ValueError, match="missing required stem"):
        evaluate_corpus(
            corpus,
            _separate(corpus, settings, stems={"Other": estimate}),
            sample_rate=8000,
        )


def test_evaluate_corpus_rejects_empty_outputs(tmp_path):
    corpus = _corpus(tmp_path)
    settings = RunSettings(model="test", sample_rate=8000)

    with pytest.raises(ValueError, match="empty outputs"):
        evaluate_corpus(corpus, _separate(corpus, settings, stems={}), sample_rate=8000)


@pytest.mark.parametrize(
    "estimate",
    [
        np.empty((0, 2), dtype=np.float32),
        np.ones(8, dtype=np.float32),
        np.ones((7, 2), dtype=np.float32),
        np.ones((8, 1), dtype=np.float32),
        np.full((8, 2), np.nan, dtype=np.float32),
    ],
    ids=["empty", "non-2d", "wrong-length", "wrong-channels", "non-finite"],
)
def test_evaluate_corpus_rejects_invalid_estimate(tmp_path, estimate):
    corpus = _corpus(tmp_path)
    settings = RunSettings(model="test", sample_rate=8000)

    with pytest.raises(ValueError):
        evaluate_corpus(
            corpus,
            _separate(corpus, settings, stems={"Vocals": estimate}),
            sample_rate=8000,
        )


@pytest.mark.parametrize(
    "reference",
    [
        np.empty((0, 2), dtype=np.float32),
        np.ones(8, dtype=np.float32),
        np.full((8, 2), np.inf, dtype=np.float32),
    ],
    ids=["empty", "non-2d", "non-finite"],
)
def test_evaluate_corpus_rejects_invalid_reference(tmp_path, monkeypatch, reference):
    corpus = _corpus(tmp_path)
    settings = RunSettings(model="test", sample_rate=8000)

    def fake_read(*_args, **_kwargs):
        return reference, 8000

    monkeypatch.setattr(harness.sf, "read", fake_read)

    with pytest.raises(ValueError):
        evaluate_corpus(corpus, _separate(corpus, settings), sample_rate=8000)


def test_evaluate_corpus_rejects_reference_rate_mismatch(tmp_path):
    corpus = _corpus(tmp_path, sample_rate=16000)
    settings = RunSettings(model="test", sample_rate=8000)

    with pytest.raises(ValueError, match="sample rate"):
        evaluate_corpus(corpus, _separate(corpus, settings), sample_rate=8000)


def test_evaluate_corpus_retains_inconsistent_settings_per_item(tmp_path):
    signal = np.ones((8, 2), dtype=np.float32)
    items = []
    for index in range(2):
        mixture = tmp_path / f"mix-{index}.wav"
        reference = tmp_path / f"vocals-{index}.wav"
        sf.write(mixture, signal, 8000, subtype="FLOAT")
        sf.write(reference, signal, 8000, subtype="FLOAT")
        items.append(CorpusItem(str(mixture), {"Vocals": str(reference)}))
    corpus = ReferenceCorpus(items)

    def separate(mixture):
        index = int(mixture.rsplit("-", 1)[1].split(".")[0])
        ref, _ = sf.read(corpus.items[index].stems["Vocals"], dtype="float32", always_2d=True)
        return {"Vocals": ref}, RunSettings(model=f"test-{index}", sample_rate=8000)

    report = evaluate_corpus(corpus, separate, sample_rate=8000)

    assert report.settings is None
    assert [row.settings.model for row in report.item_settings] == [
        "test-0",
        "test-1",
    ]

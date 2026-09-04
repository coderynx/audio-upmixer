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
        }]
    }
    (tmp_path / "corpus.json").write_text(json.dumps(manifest), encoding="utf-8")

    item = ReferenceCorpus.from_dir(str(tmp_path)).items[0]

    assert (item.recording_id, item.item_id, item.split) == (
        "rec-01",
        "rec-01-full",
        "holdout",
    )


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


def test_evaluate_corpus_rejects_inconsistent_settings(tmp_path):
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

    with pytest.raises(ValueError, match="inconsistent RunSettings"):
        evaluate_corpus(corpus, separate, sample_rate=8000)

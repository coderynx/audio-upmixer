"""Per-item settings retained by the evaluation harness."""
from __future__ import annotations

import json

import numpy as np
import soundfile as sf

from upmixer.eval import ItemRunSettings, RunSettings, evaluate_corpus
from upmixer.eval.corpus import CorpusItem, ReferenceCorpus
from upmixer.eval.report import EvalReport, StemScore, format_report
from upmixer.separation.separator import SeparationSettings


def _stage(model: str) -> SeparationSettings:
    return SeparationSettings(
        model=model,
        sample_rate=8000,
        batch_size=1,
        segment_size=64,
        chunk_duration_s=None,
        overlap=2,
        tta=False,
        pitch_shift=None,
        backend="cpu",
        model_arch="bs_roformer",
        model_config_name=f"{model}-config",
        model_native_sample_rate=8000,
        device="cpu",
    )


def _settings(model: str, *stages: str) -> RunSettings:
    return RunSettings(
        model=model,
        sample_rate=8000,
        segment_size=64,
        overlap=2,
        batch_size=1,
        chunk_duration_s=None,
        tta=False,
        pitch_shift=None,
        backend="cpu",
        stage_settings=tuple(_stage(stage) for stage in stages),
        device="cpu",
    )


def _corpus(tmp_path) -> ReferenceCorpus:
    signal = np.ones((8, 2), dtype=np.float32)
    items = []
    for index in range(2):
        mixture = tmp_path / f"mix-{index}.wav"
        reference = tmp_path / f"vocals-{index}.wav"
        sf.write(mixture, signal, 8000, subtype="FLOAT")
        sf.write(reference, signal, 8000, subtype="FLOAT")
        items.append(
            CorpusItem(
                mixture=str(mixture),
                stems={"Vocals": str(reference)},
                category=f"category-{index}",
                recording_id=f"recording-{index}",
                item_id=f"item-{index}",
                split="holdout",
            )
        )
    return ReferenceCorpus(items)


def _separate(corpus: ReferenceCorpus, settings: list[RunSettings]):
    by_item = {item.item_id: item_settings for item, item_settings in zip(corpus.items, settings)}

    def separate(mixture_path: str):
        item = next(item for item in corpus.items if item.mixture == mixture_path)
        reference, _ = sf.read(item.stems["Vocals"], dtype="float32", always_2d=True)
        return {"Vocals": reference}, by_item[item.item_id]

    return separate


def test_matching_settings_keep_shared_report_settings_and_retain_rows(tmp_path):
    corpus = _corpus(tmp_path)
    settings = _settings("incumbent", "incumbent-stage")

    report = evaluate_corpus(
        corpus,
        _separate(corpus, [settings, settings]),
        sample_rate=8000,
    )

    assert report.settings == settings
    assert [row.settings for row in report.item_settings] == [settings, settings]
    assert [(row.recording_id, row.item_id, row.split, row.category) for row in report.item_settings] == [
        ("recording-0", "item-0", "holdout", "category-0"),
        ("recording-1", "item-1", "holdout", "category-1"),
    ]
    assert all(isinstance(row, ItemRunSettings) for row in report.item_settings)


def test_varying_settings_clear_shared_setting_and_format_each_item_and_stage(tmp_path):
    corpus = _corpus(tmp_path)
    first = _settings("first", "first-stage")
    second = _settings("second", "second-stage-a", "second-stage-b")

    report = evaluate_corpus(
        corpus,
        _separate(corpus, [first, second]),
        sample_rate=8000,
    )

    assert report.settings is None
    assert [row.settings for row in report.item_settings] == [first, second]
    text = format_report(report)
    assert "Settings vary by item:" in text
    assert "Item recording_id=recording-0 item_id=item-0 split=holdout category=category-0" in text
    assert "model=first" in text
    assert "  Stage 1: model=first-stage" in text
    assert "Item recording_id=recording-1 item_id=item-1 split=holdout category=category-1" in text
    assert "  Stage 1: model=second-stage-a" in text
    assert "  Stage 2: model=second-stage-b" in text


def test_varying_item_settings_are_json_serialized_with_stages():
    first = _settings("first", "first-stage")
    second = _settings("second", "second-stage")
    report = EvalReport(
        settings=None,
        scores=[
            StemScore(
                "Vocals",
                "default",
                1.0,
                0.5,
                0.5,
                recording_id="recording-0",
                item_id="item-0",
            )
        ],
        item_settings=[
            ItemRunSettings("recording-0", "item-0", "holdout", "default", first),
            ItemRunSettings("recording-1", "item-1", "holdout", "default", second),
        ],
    )

    payload = report.to_dict()

    assert payload["settings"] is None
    assert [row["item_id"] for row in payload["item_settings"]] == ["item-0", "item-1"]
    assert payload["item_settings"][0]["settings"]["stage_settings"][0]["model"] == "first-stage"
    assert json.loads(report.to_json()) == payload

"""Readable and JSON-safe rendering of production evaluation context."""
import json

from upmixer.eval.harness import RunSettings
from upmixer.eval.report import EvalReport, StemScore, format_report
from upmixer.separation.separator import SeparationSettings


def test_report_serializes_and_renders_tree_and_runtime_context():
    plan = {
        "requested_stems": ["Bass", "Kick"],
        "tasks": [
            {
                "model": "first.ckpt",
                "input_source": "original",
                "output_stems": ["Bass", "Drums"],
                "keep_stems": ["Drums"],
                "ensemble_models": ["second.ckpt"],
                "ensemble_stems": ["Bass"],
                "ensemble_algorithm": "avg_wave",
                "executed": True,
            }
        ],
        "stems_hash": "0123456789abcdef0123",
        "inference_hash": "fedcba9876543210fedc",
    }
    stage = SeparationSettings(
        model="first.ckpt",
        sample_rate=44_100,
        batch_size=2,
        segment_size=256,
        chunk_duration_s=600.0,
        overlap=2,
        tta=False,
        pitch_shift=None,
        backend="cpu",
        checkpoint_sha256="a" * 64,
        model_config_sha256="b" * 64,
        runtime_precision="float32",
        normalization_policy="peak-downscale-to-0.9; inverse-output-scale",
        oom_fallback_attempts=(
            {"batch_size": 4, "segment_size": 512, "chunk_duration_s": 600.0},
            {"batch_size": 2, "segment_size": 512, "chunk_duration_s": 600.0},
        ),
        oom_fallback_count=2,
    )
    settings = RunSettings(
        model="production-tree",
        sample_rate=44_100,
        input_sample_rate=48_000,
        separation_sample_rate=44_100,
        output_sample_rate=44_100,
        scoring_sample_rate=44_100,
        plan=plan,
        stage_settings=(stage,),
        stem_primary_remask=True,
        stem_drum_remask=False,
        stem_bleed_reduction=True,
        stem_ensemble=True,
        stem_silence_skip=True,
        stem_silence_threshold_db=-90.0,
        stem_silence_min_duration_s=2.0,
        stem_silence_crossfade_ms=10.0,
        stem_silence_pad_ms=200.0,
    )
    report = EvalReport(
        settings=settings,
        scores=[
            StemScore(
                stem="Bass",
                category="default",
                sdr=1.0,
                fullness=0.5,
                bleedless=0.6,
                recording_id="recording-0",
                item_id="item-0",
            )
        ],
    )

    payload = report.to_dict()
    text = format_report(report)

    assert json.loads(report.to_json()) == payload
    serialized = payload["settings"]
    assert serialized["plan"] == plan
    assert serialized["stage_settings"][0]["checkpoint_sha256"] == "a" * 64
    assert "input_sample_rate=48000 separation_sample_rate=44100" in text
    assert "output_sample_rate=44100 scoring_sample_rate=44100" in text
    assert '"input_source":"original"' in text
    assert '"output_stems":["Bass","Drums"]' in text
    assert '"keep_stems":["Drums"]' in text
    assert '"ensemble_models":["second.ckpt"]' in text
    assert '"stems_hash":"0123456789ab..."' in text
    assert '"inference_hash":"fedcba987654..."' in text
    assert "stem_primary_remask=True" in text
    assert "stem_drum_remask=False" in text
    assert "stem_bleed_reduction=True" in text
    assert "stem_ensemble=True" in text
    assert "checkpoint_sha256=aaaaaaaaaaaa..." in text
    assert "model_config_sha256=bbbbbbbbbbbb..." in text
    assert "runtime_precision=float32" in text
    assert "normalization_policy=peak-downscale-to-0.9; inverse-output-scale" in text
    assert 'oom_fallback_attempts=[{"batch_size":4,"chunk_duration_s":600.0' in text
    assert "oom_fallback_count=2" in text

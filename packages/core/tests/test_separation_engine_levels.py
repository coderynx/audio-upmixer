"""Level-domain invariants of SeparationEngine's write path.

The pre-demix peak normalization must be divided back out of every stem, so
stems stay in the input's level domain regardless of how loud the input was.
"""
from __future__ import annotations

import numpy as np
import pytest
import soundfile as sf

torch = pytest.importorskip("torch")

from upmixer.separation.inference.config import ModelConfig  # noqa: E402
from upmixer.separation.inference.device import DeviceManager  # noqa: E402
from upmixer.separation.inference.engine import SeparationEngine  # noqa: E402


def _make_config() -> ModelConfig:
    return ModelConfig(
        audio={"sample_rate": 44100, "hop_length": 100},
        model={"stft_hop_length": 100},
        training={"instruments": ["vocals", "other"], "target_instrument": None},
        inference={"dim_t": 6},
    )


def _make_engine(output_dir: str, sample_rate: int = 44100) -> SeparationEngine:
    engine = SeparationEngine(
        model=torch.nn.Identity(),
        config=_make_config(),
        arch="bs_roformer",
        model_filename="test.ckpt",
        device=DeviceManager("cpu"),
        output_dir=output_dir,
        sample_rate=sample_rate,
        batch_size=1,
        segment_size=None,
        chunk_duration_s=None,
    )
    # Stand-in for a real checkpoint: a linear split that sums back to its
    # input, so any level error the engine introduces is the only thing the
    # assertions below can see.
    engine._demix_arch = lambda mix: {
        "vocals": mix * 0.25,
        "other": mix * 0.75,
    }
    return engine


def _write_source(path, audio: np.ndarray, sample_rate: int = 44100) -> str:
    sf.write(str(path), audio.T, sample_rate, subtype="FLOAT")
    return str(path)


def _separate(tmp_path, name: str, audio: np.ndarray) -> dict[str, np.ndarray]:
    source = _write_source(tmp_path / f"{name}.wav", audio)
    out_dir = tmp_path / f"{name}_out"
    paths = _make_engine(str(out_dir)).separate(source)
    stems = {}
    for path in paths:
        data, _ = sf.read(path, dtype="float32", always_2d=True)
        stems[path] = data.T
    return stems


def _loud_mix(n_samples: int = 2000) -> np.ndarray:
    rng = np.random.default_rng(0)
    mix = rng.standard_normal((2, n_samples)).astype(np.float32)
    return (mix / np.abs(mix).max() * 1.073).astype(np.float32)


def _diagnostic_mix(sample_rate: int, n_samples: int = 2001) -> np.ndarray:
    t = np.arange(n_samples, dtype=np.float64) / sample_rate
    high_hz = 18_000.0
    mix = np.stack(
        (
            0.35 * np.sin(2 * np.pi * 220.0 * t)
            + 0.18 * np.sin(2 * np.pi * high_hz * t),
            0.22 * np.sin(2 * np.pi * 330.0 * t)
            - 0.12 * np.sin(2 * np.pi * high_hz * t),
        )
    ).astype(np.float32)
    mix[0, n_samples // 2] += 1.2
    return mix


def _tone_amplitude(audio: np.ndarray, sample_rate: int, frequency: float) -> float:
    t = np.arange(audio.shape[-1], dtype=np.float64) / sample_rate
    basis = np.stack(
        (
            np.sin(2 * np.pi * frequency * t),
            np.cos(2 * np.pi * frequency * t),
        ),
        axis=1,
    )
    coefficients = np.linalg.lstsq(basis, audio, rcond=None)[0]
    return float(np.hypot(*coefficients))


_RATE_CASES = (
    ("native-44k", 44_100, 44_100),
    ("native-48k", 48_000, 48_000),
    ("resampled-48k", 48_000, 44_100),
    ("resampled-to-48k", 44_100, 48_000),
    ("native-96k", 96_000, 96_000),
    ("resampled-96k", 44_100, 96_000),
)
_RESAMPLED_RATE_CASES = tuple(case for case in _RATE_CASES if case[1] != case[2])


def test_hot_input_restores_gain_and_stems_sum_to_input_level(tmp_path):
    mix = _loud_mix()
    stems = _separate(tmp_path, "loud", mix)

    assert np.max(np.abs(mix)) > 0.9
    vocals = next(audio for path, audio in stems.items() if "(vocals)" in path.casefold())
    other = next(audio for path, audio in stems.items() if "(other)" in path.casefold())
    np.testing.assert_allclose(vocals, mix * 0.25, atol=1e-5)
    np.testing.assert_allclose(other, mix * 0.75, atol=1e-5)
    np.testing.assert_allclose(vocals + other, mix, atol=1e-5)


def test_halving_input_halves_output_exactly(tmp_path):
    mix = _loud_mix()
    full = _separate(tmp_path, "full", mix)
    half = _separate(tmp_path, "half", (mix * 0.5).astype(np.float32))

    assert len(full) == len(half) == 2
    for full_path, half_path in zip(sorted(full), sorted(half)):
        assert np.allclose(half[half_path], full[full_path] * 0.5, atol=1e-6)


@pytest.mark.parametrize(
    ("case", "source_rate", "engine_rate"),
    _RATE_CASES,
    ids=[case[0] for case in _RATE_CASES],
)
def test_rate_length_parent_and_high_band_contract(
    tmp_path, case: str, source_rate: int, engine_rate: int
):
    from upmixer.separation.inference.audio_io import load_audio

    source = _write_source(
        tmp_path / f"{case}.wav",
        _diagnostic_mix(source_rate),
        source_rate,
    )
    source_audio, source_file_rate = sf.read(
        source, dtype="float32", always_2d=True
    )
    assert source_file_rate == source_rate
    engine = _make_engine(str(tmp_path / f"{case}_out"), engine_rate)

    paths = engine.separate(source, retain_parent=True)
    parent = engine.take_last_parent()
    expected = load_audio(source, engine_rate)

    np.testing.assert_array_equal(parent, expected.T)
    outputs = {}
    for path in paths:
        data, output_rate = sf.read(path, dtype="float32", always_2d=True)
        assert output_rate == engine_rate
        assert data.shape == (expected.shape[1], 2)
        outputs[path] = data.T

    vocals = next(audio for path, audio in outputs.items() if "(vocals)" in path.casefold())
    other = next(audio for path, audio in outputs.items() if "(other)" in path.casefold())
    np.testing.assert_allclose(vocals, expected * 0.25, atol=2e-5)
    np.testing.assert_allclose(other, expected * 0.75, atol=2e-5)
    np.testing.assert_allclose(vocals + other, expected, atol=2e-5)
    if source_rate == engine_rate:
        np.testing.assert_array_equal(expected.T, source_audio)
        np.testing.assert_array_equal(parent, source_audio)
        np.testing.assert_allclose(vocals, source_audio.T * 0.25, atol=2e-5)
        np.testing.assert_allclose(other, source_audio.T * 0.75, atol=2e-5)
    if source_rate == engine_rate == 96_000:
        assert _tone_amplitude(vocals[0], engine_rate, 18_000.0) > 0.02

    with pytest.raises(RuntimeError, match="No completed separation input"):
        engine.take_last_parent()


@pytest.mark.parametrize(
    ("case", "source_rate", "engine_rate"),
    _RESAMPLED_RATE_CASES,
    ids=[case[0] for case in _RESAMPLED_RATE_CASES],
)
def test_resampled_impulse_has_no_material_position_delay(
    tmp_path, case: str, source_rate: int, engine_rate: int
):
    from upmixer.separation.inference.audio_io import load_audio

    n_samples = 2001
    impulse_index = n_samples // 2
    impulse = np.zeros((2, n_samples), dtype=np.float32)
    impulse[:, impulse_index] = 1.0
    source = _write_source(tmp_path / f"{case}-impulse.wav", impulse, source_rate)

    loaded = load_audio(source, engine_rate)
    expected_index = round(impulse_index * engine_rate / source_rate)
    peak_index = int(np.argmax(np.abs(loaded[0])))

    assert abs(peak_index - expected_index) <= 2


@pytest.mark.parametrize(
    ("case", "source_rate", "engine_rate"),
    _RESAMPLED_RATE_CASES,
    ids=[case[0] for case in _RESAMPLED_RATE_CASES],
)
def test_resampled_audio_retains_low_and_high_tones(
    tmp_path, case: str, source_rate: int, engine_rate: int
):
    from upmixer.separation.inference.audio_io import load_audio

    n_samples = source_rate
    time = np.arange(n_samples, dtype=np.float64) / source_rate
    signal = (
        0.35 * np.sin(2 * np.pi * 220.0 * time)
        + 0.18 * np.sin(2 * np.pi * 18_000.0 * time)
    ).astype(np.float32)
    source = _write_source(
        tmp_path / f"{case}-tones.wav",
        np.stack((signal, signal)),
        source_rate,
    )

    loaded = load_audio(source, engine_rate)

    assert _tone_amplitude(loaded[0], engine_rate, 220.0) == pytest.approx(
        0.35, rel=0.02, abs=0.002
    )
    assert _tone_amplitude(loaded[0], engine_rate, 18_000.0) == pytest.approx(
        0.18, rel=0.02, abs=0.002
    )

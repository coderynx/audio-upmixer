"""Physical-frequency calibration of the shipped binaural filter banks."""

import runpy
from pathlib import Path

import numpy as np
import pytest

from measured_xtc_fixture import MEASURED_HRIRS
from upmixer.binaural.decoder import HRIR_DIR
from upmixer.binaural.renderer import render_binaural
from upmixer.formats import FORMAT_MAP, MEASURED_HRIR_LAYOUTS


@pytest.mark.parametrize("layout", MEASURED_HRIR_LAYOUTS)
@pytest.mark.parametrize("channel,index", (("FL", 0), ("FR", 1)))
def test_conditioned_bank_preserves_measured_magnitudes_and_itd(layout, channel, index):
    impulse = np.zeros(8192)
    impulse[0] = 1.0
    ears = np.array(render_binaural({channel: impulse}, FORMAT_MAP[layout], 48000, "flat"))
    actual = np.fft.rfft(ears, axis=1)
    measured = np.fft.rfft(MEASURED_HRIRS[index], n=len(impulse), axis=1)
    frequencies = np.fft.rfftfreq(len(impulse), 1 / 48000)
    band = (frequencies >= 80) & (frequencies <= 16000)
    error_db = 20 * np.log10(np.abs(actual[:, band] / measured[:, band]))
    assert np.max(np.abs(error_db)) < 0.25

    timing_band = (frequencies >= 200) & (frequencies <= 1500)
    timing_error = np.unwrap(np.angle(actual[1] * measured[0] / (actual[0] * measured[1])))
    delay_error_s = np.polyfit(2 * np.pi * frequencies[timing_band], timing_error[timing_band], 1)[0]
    assert abs(delay_error_s) < 2e-6


def test_phase_conditioning_removes_common_measurement_delay():
    script = Path(__file__).resolve().parents[3] / "scripts/build_binaural_filters.py"
    condition = runpy.run_path(str(script))["condition_direct_hrirs"]
    original = MEASURED_HRIRS[:2].astype(np.float64)
    shifted = np.pad(original, ((0, 0), (0, 0), (37, 0)))
    np.testing.assert_allclose(condition(shifted), condition(original), atol=1e-12)


def test_preview_and_export_share_conditioned_assets():
    web = Path(__file__).resolve().parents[3] / "apps/web/public/hrir"
    for part in HRIR_DIR.glob("*.wav"):
        assert part.read_bytes() == (web / part.name).read_bytes(), part.name


@pytest.mark.parametrize("profile", ("flat", "studio", "listening"))
def test_coherent_front_bed_does_not_cancel_presence(profile):
    impulse = np.zeros(8192)
    impulse[0] = 1.0
    ears = np.array(render_binaural(
        {channel: impulse for channel in ("FL", "C", "FR")},
        FORMAT_MAP["7.1.4"], 48000, profile,
    ))
    frequencies = np.fft.rfftfreq(len(impulse), 1 / 48000)
    power = np.abs(np.fft.rfft(ears, axis=1)) ** 2
    bass = np.mean(power[:, (frequencies >= 80) & (frequencies < 300)])
    presence = np.mean(power[:, (frequencies >= 1000) & (frequencies < 4000)])
    assert 10 * np.log10(bass / presence) < 2.0


@pytest.mark.parametrize("profile", ("flat", "studio", "listening"))
@pytest.mark.parametrize("sample_rate", (44100, 88200, 96000))
def test_binaural_resampling_preserves_speaker_gain_relative_to_lfe(profile, sample_rate):
    frequencies = np.array([80.0, 250.0, 1000.0, 8000.0])

    def response(rate, channel):
        impulse = np.zeros(round(rate * 0.05))
        impulse[0] = 1.0
        ears = np.array(render_binaural(
            {"FL": np.zeros_like(impulse), channel: impulse},
            FORMAT_MAP["7.1.4"], rate, profile,
            lfe_already_processed=True,
        ))
        basis = np.exp(-2j * np.pi * frequencies[:, None] * np.arange(len(impulse)) / rate)
        return np.abs(ears @ basis.T)

    reference = response(48000, "FL") / response(48000, "LFE")
    actual = response(sample_rate, "FL") / response(sample_rate, "LFE")
    error_db = 20 * np.log10(actual / reference)
    assert np.max(np.abs(error_db)) < 0.1, error_db

"""Tests for upmixer.mastering_eq_match — EQMatcher + _CHANNEL_PROXIES."""
from __future__ import annotations

import json
import tempfile
from pathlib import Path

import numpy as np
import pytest

from upmixer.mastering_eq_match import (
    _CHANNEL_PROXIES,
    _gaussian_smooth_log,
    EQMatcher,
)


# ---------------------------------------------------------------------------
# Helpers
# ---------------------------------------------------------------------------

def _write_wav(path: str, data: np.ndarray, sr: int = 44100) -> str:
    """Write a float64 numpy array to a WAV file. Requires soundfile."""
    sf = pytest.importorskip("soundfile", reason="soundfile not installed")
    sf.write(path, data, sr, subtype="FLOAT")
    return path


def _sine(n: int, freq: float = 440.0, amplitude: float = 0.2) -> np.ndarray:
    t = np.linspace(0, n / 44100, n, endpoint=False)
    return (amplitude * np.sin(2 * np.pi * freq * t)).astype(np.float64)


def _stereo_wav(tmp_path: Path, n: int = 44100 * 3) -> str:
    """Create a stereo WAV test file and return path."""
    pytest.importorskip("soundfile", reason="soundfile not installed")
    sig = np.stack([_sine(n), _sine(n, freq=880.0)], axis=1)
    path = str(tmp_path / "ref_stereo.wav")
    _write_wav(path, sig)
    return path


def _51_wav(tmp_path: Path, n: int = 44100 * 3) -> str:
    """Create a 5.1 WAV test file and return path."""
    pytest.importorskip("soundfile", reason="soundfile not installed")
    freqs = [440, 660, 880, 110, 550, 770]
    sig = np.stack([_sine(n, f) for f in freqs], axis=1)
    path = str(tmp_path / "ref_51.wav")
    _write_wav(path, sig)
    return path


# ---------------------------------------------------------------------------
# _CHANNEL_PROXIES sanity
# ---------------------------------------------------------------------------

class TestChannelProxies:
    def test_supported_channel_counts(self):
        assert 1 in _CHANNEL_PROXIES
        assert 2 in _CHANNEL_PROXIES
        assert 6 in _CHANNEL_PROXIES
        assert 8 in _CHANNEL_PROXIES

    def test_stereo_proxies_fl_fr(self):
        assert _CHANNEL_PROXIES[2]["FL"] == 0
        assert _CHANNEL_PROXIES[2]["FR"] == 1

    def test_stereo_proxy_c_is_mid(self):
        assert _CHANNEL_PROXIES[2]["C"] == "mid"

    def test_stereo_proxy_lfe_is_mid_lp(self):
        assert _CHANNEL_PROXIES[2]["LFE"] == "mid_lp"

    def test_51_proxies_direct_channels(self):
        assert _CHANNEL_PROXIES[6]["FL"]  == 0
        assert _CHANNEL_PROXIES[6]["FR"]  == 1
        assert _CHANNEL_PROXIES[6]["C"]   == 2
        assert _CHANNEL_PROXIES[6]["LFE"] == 3
        assert _CHANNEL_PROXIES[6]["SL"]  == 4
        assert _CHANNEL_PROXIES[6]["SR"]  == 5

    def test_mono_maps_all_to_zero(self):
        proxy = _CHANNEL_PROXIES[1]
        for ch, idx in proxy.items():
            assert idx == 0, f"Mono proxy for {ch} is not 0"


# ---------------------------------------------------------------------------
# _gaussian_smooth_log
# ---------------------------------------------------------------------------

class TestGaussianSmoothLog:
    def test_flat_signal_stays_flat(self):
        log_f = np.linspace(1, 4, 100)  # log2 of some freqs
        vals = np.zeros(100)
        smoothed = _gaussian_smooth_log(log_f, vals, 0.25)
        np.testing.assert_allclose(smoothed, 0.0, atol=1e-10)

    def test_output_same_length(self):
        log_f = np.linspace(1, 4, 200)
        vals = np.random.default_rng(0).standard_normal(200)
        smoothed = _gaussian_smooth_log(log_f, vals, 0.25)
        assert len(smoothed) == 200

    def test_smoothing_reduces_variance(self):
        log_f = np.linspace(1, 4, 200)
        rng = np.random.default_rng(42)
        noisy = rng.standard_normal(200)
        smoothed = _gaussian_smooth_log(log_f, noisy, 0.5)
        assert float(np.var(smoothed)) < float(np.var(noisy))


# ---------------------------------------------------------------------------
# EQMatcher construction
# ---------------------------------------------------------------------------

class TestEQMatcherInit:
    def test_constructs(self):
        m = EQMatcher(44100)
        assert m is not None

    def test_custom_n_fft(self):
        m = EQMatcher(48000, n_fft=4096)
        assert m._n_fft == 4096


# ---------------------------------------------------------------------------
# EQMatcher.analyze — stereo reference
# ---------------------------------------------------------------------------

class TestEQMatcherAnalyzeStereo:
    def test_returns_dict(self, tmp_path):
        ref_path = _stereo_wav(tmp_path)
        m = EQMatcher(44100)
        result = m.analyze(ref_path, ["FL", "FR", "C"])
        assert isinstance(result, dict)

    def test_all_target_channels_present(self, tmp_path):
        ref_path = _stereo_wav(tmp_path)
        m = EQMatcher(44100)
        targets = ["FL", "FR", "C", "LFE", "SL", "SR"]
        result = m.analyze(ref_path, targets)
        for ch in targets:
            assert ch in result, f"Missing channel {ch}"

    def test_breakpoints_are_tuples(self, tmp_path):
        ref_path = _stereo_wav(tmp_path)
        m = EQMatcher(44100)
        result = m.analyze(ref_path, ["FL"])
        for f, g in result["FL"]:
            assert isinstance(f, float)
            assert isinstance(g, float)

    def test_breakpoints_ascending(self, tmp_path):
        ref_path = _stereo_wav(tmp_path)
        m = EQMatcher(44100)
        result = m.analyze(ref_path, ["FL", "FR"])
        for ch, bps in result.items():
            freqs = [f for f, _ in bps]
            assert freqs == sorted(freqs), f"Breakpoints not ascending for {ch}"

    def test_breakpoints_freqs_positive(self, tmp_path):
        ref_path = _stereo_wav(tmp_path)
        m = EQMatcher(44100)
        result = m.analyze(ref_path, ["FL"])
        for f, _ in result["FL"]:
            assert f > 0.0

    def test_breakpoints_count(self, tmp_path):
        """Should return approximately N_BREAKPOINTS breakpoints."""
        from upmixer.mastering_eq_match import _N_BREAKPOINTS
        ref_path = _stereo_wav(tmp_path)
        m = EQMatcher(44100)
        result = m.analyze(ref_path, ["FL"])
        assert len(result["FL"]) == _N_BREAKPOINTS

    def test_gains_are_finite(self, tmp_path):
        ref_path = _stereo_wav(tmp_path)
        m = EQMatcher(44100)
        result = m.analyze(ref_path, ["FL", "FR", "C", "LFE"])
        for ch, bps in result.items():
            for f, g in bps:
                assert np.isfinite(g), f"Non-finite gain {g} in {ch}"


# ---------------------------------------------------------------------------
# EQMatcher.analyze — 5.1 reference
# ---------------------------------------------------------------------------

class TestEQMatcherAnalyze51:
    def test_51_reference_direct_channels(self, tmp_path):
        ref_path = _51_wav(tmp_path)
        m = EQMatcher(44100)
        result = m.analyze(ref_path, ["FL", "FR", "C", "LFE", "SL", "SR"])
        for ch in ["FL", "FR", "C", "LFE", "SL", "SR"]:
            assert ch in result


# ---------------------------------------------------------------------------
# EQMatcher.save_profile / load_profile
# ---------------------------------------------------------------------------

class TestEQMatcherSaveLoad:
    def _sample_bps(self) -> dict[str, list[tuple[float, float]]]:
        freqs = np.logspace(np.log10(20), np.log10(20000), 10)
        return {
            "FL": [(float(f), float(g)) for f, g in zip(freqs, np.zeros(10))],
            "FR": [(float(f), float(g)) for f, g in zip(freqs, np.ones(10) * 0.5)],
        }

    def test_save_load_json(self, tmp_path):
        path = str(tmp_path / "profile.json")
        bps = self._sample_bps()
        m = EQMatcher(44100)
        m.save_profile(bps, path)
        loaded = EQMatcher.load_profile(path)
        for ch in bps:
            assert ch in loaded
            for (f1, g1), (f2, g2) in zip(bps[ch], loaded[ch]):
                assert f1 == pytest.approx(f2)
                assert g1 == pytest.approx(g2)

    def test_save_load_yaml(self, tmp_path):
        pytest.importorskip("yaml", reason="pyyaml not installed")
        path = str(tmp_path / "profile.yaml")
        bps = self._sample_bps()
        m = EQMatcher(44100)
        m.save_profile(bps, path)
        loaded = EQMatcher.load_profile(path)
        assert set(loaded.keys()) == set(bps.keys())

    def test_json_file_created(self, tmp_path):
        path = str(tmp_path / "out.json")
        bps = self._sample_bps()
        EQMatcher(44100).save_profile(bps, path)
        assert Path(path).exists()

    def test_load_preserves_tuple_type(self, tmp_path):
        path = str(tmp_path / "prof.json")
        bps = self._sample_bps()
        EQMatcher(44100).save_profile(bps, path)
        loaded = EQMatcher.load_profile(path)
        for ch, bp_list in loaded.items():
            for item in bp_list:
                assert len(item) == 2
                assert isinstance(item[0], float)
                assert isinstance(item[1], float)


# ---------------------------------------------------------------------------
# SpectralShaper per-channel mode (integration)
# ---------------------------------------------------------------------------

class TestSpectralShaperPerChannel:
    """Verify SpectralShaper works in per-channel mode with EQMatcher output."""

    def test_per_channel_mode_runs(self, tmp_path):
        from upmixer.mastering_eq import SpectralShaper

        bps: dict[str, list[tuple[float, float]]] = {
            "FL": [(20., 0.), (1000., 0.), (10000., 1.5), (20000., 1.5)],
            "FR": [(20., 0.), (1000., 0.), (10000., 1.5), (20000., 1.5)],
            "C":  [(20., 0.), (20000., 0.)],
        }
        t = np.linspace(0, 1, 44100, endpoint=False)
        sig = 0.3 * np.sin(2 * np.pi * 440 * t).astype(np.float64)
        chs = {"FL": sig.copy(), "FR": sig.copy(), "C": sig.copy(),
               "LFE": sig.copy() * 0.5, "SL": sig.copy(), "SR": sig.copy()}

        shaper = SpectralShaper(
            profile=None, strength=1.0, sample_rate=44100,
            per_channel_breakpoints=bps,
        )
        out = shaper.process(chs)
        for arr in out.values():
            assert np.all(np.isfinite(arr))

    def test_per_channel_lfe_bypassed(self, tmp_path):
        from upmixer.mastering_eq import SpectralShaper

        bps = {"FL": [(20., 0.), (20000., 2.)]}
        t = np.linspace(0, 1, 44100, endpoint=False)
        sig = 0.3 * np.sin(2 * np.pi * 440 * t).astype(np.float64)
        chs = {"FL": sig.copy(), "LFE": sig.copy()}

        shaper = SpectralShaper(
            profile=None, strength=1.0, sample_rate=44100,
            per_channel_breakpoints=bps,
        )
        out = shaper.process(chs)
        np.testing.assert_array_equal(out["LFE"], chs["LFE"])

    def test_missing_channel_in_bps_passes_through(self):
        from upmixer.mastering_eq import SpectralShaper

        bps = {"FL": [(20., 0.), (20000., 2.)]}  # FR not in bps
        t = np.linspace(0, 1, 44100, endpoint=False)
        sig = 0.3 * np.sin(2 * np.pi * 440 * t).astype(np.float64)
        chs = {"FL": sig.copy(), "FR": sig.copy()}

        shaper = SpectralShaper(
            profile=None, strength=1.0, sample_rate=44100,
            per_channel_breakpoints=bps,
        )
        out = shaper.process(chs)
        assert out["FR"] is chs["FR"]  # pass-through, not filtered

    def test_neither_profile_nor_bps_raises(self):
        from upmixer.mastering_eq import SpectralShaper
        with pytest.raises(ValueError, match="profile.*per_channel"):
            SpectralShaper(profile=None, strength=1.0, sample_rate=44100)

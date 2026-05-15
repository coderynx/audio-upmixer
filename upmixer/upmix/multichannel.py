import numpy as np
from scipy.signal import butter, sosfilt

from upmixer.config import UpmixConfig
from upmixer.formats import ChannelLabel, InputFormat, OutputFormat


def _allpass_decorr(signal: np.ndarray, seed: int, n_taps: int = 1024) -> np.ndarray:
    """Decorrelate via convolution with a random-phase unit-magnitude FIR.

    numpy's irfft of a unit-magnitude spectrum already yields sum(h²) ≈ 1,
    so no additional normalization is needed — dividing by peak would amplify
    filter energy by ~100x, making derived channels far too loud.
    """
    rng = np.random.default_rng(seed)
    n_freq = n_taps // 2
    phase = rng.uniform(0, 2 * np.pi, n_freq + 1)
    h = np.fft.irfft(np.exp(1j * phase), n=n_taps)
    return np.convolve(signal, h, mode="same")


def _apply_delay(signal: np.ndarray, delay_samples: int) -> np.ndarray:
    return np.pad(signal, (delay_samples, 0))[: len(signal)]


def _lfe_filter(
    signal: np.ndarray, sr: int, cutoff_hz: float, gain: float, order: int
) -> np.ndarray:
    sos = butter(order, cutoff_hz / (sr / 2.0), btype="low", output="sos")
    return sosfilt(sos, signal) * gain


def _high_shelf(
    signal: np.ndarray,
    sr: int,
    crossover_hz: float,
    low_gain: float,
    high_gain: float,
) -> np.ndarray:
    """High-shelf: low_gain below crossover, high_gain above.

    Implemented as: original*low_gain + hp*(high_gain - low_gain).
    """
    sos_hp = butter(2, crossover_hz / (sr / 2.0), btype="high", output="sos")
    hp = sosfilt(sos_hp, signal)
    return signal * low_gain + hp * (high_gain - low_gain)


class MultichannelUpmixer:
    """Upmix multichannel audio to a higher format.

    Passes through existing channels unchanged and derives missing channels
    from spatially appropriate sources using time-domain processing.
    """

    def __init__(
        self,
        config: UpmixConfig,
        input_fmt: InputFormat,
        output_fmt: OutputFormat,
        sample_rate: int,
    ):
        self._cfg = config
        self._input_fmt = input_fmt
        self._output_fmt = output_fmt
        self._sr = sample_rate

    def process(
        self, input_channels: dict[ChannelLabel, np.ndarray]
    ) -> dict[str, np.ndarray]:
        """Pass through existing channels and derive any missing output channels."""
        cfg = self._cfg
        sr = self._sr
        fmt = self._output_fmt

        out: dict[str, np.ndarray] = {
            label.value: arr.copy() for label, arr in input_channels.items()
        }

        FL = out.get("FL")
        FR = out.get("FR")
        C = out.get("C")
        SL = out.get("SL")
        SR = out.get("SR")
        BL = out.get("BL")
        BR = out.get("BR")

        # --- Center ---
        if "C" not in out and FL is not None and FR is not None:
            out["C"] = 0.35 * (FL + FR)
            C = out["C"]

        # --- LFE ---
        if "LFE" not in out:
            src = C if C is not None else ((FL + FR) * 0.5 if FL is not None else None)
            if src is not None:
                out["LFE"] = _lfe_filter(
                    src, sr, cfg.lfe_cutoff_hz, cfg.lfe_gain, cfg.lfe_filter_order
                )

        # --- Surround (SL/SR) ---
        # Normally present for multichannel inputs; synthesised for quad-like formats.
        if "SL" not in out:
            src = FL if FL is not None else (BL if BL is not None else None)
            if src is not None:
                out["SL"] = cfg.surround_gain * _allpass_decorr(src, seed=0)
                SL = out["SL"]
        if "SR" not in out:
            src = FR if FR is not None else (BR if BR is not None else None)
            if src is not None:
                out["SR"] = cfg.surround_gain * _allpass_decorr(src, seed=1)
                SR = out["SR"]

        # --- Back surround (BL/BR) ---
        if fmt.has_back:
            delay = int(cfg.back_delay_ms * sr / 1000.0)
            if "BL" not in out and SL is not None:
                out["BL"] = cfg.back_gain * _apply_delay(
                    _allpass_decorr(SL, seed=2), delay
                )
                BL = out["BL"]
            if "BR" not in out and SR is not None:
                out["BR"] = cfg.back_gain * _apply_delay(
                    _allpass_decorr(SR, seed=3), delay
                )
                BR = out["BR"]

        # --- Height channels ---
        if fmt.has_height:
            n = len(next(iter(out.values())))

            if FL is not None:
                sl_L = SL * 0.3 if SL is not None else np.zeros_like(FL)
                sl_R = SR * 0.3 if SR is not None else np.zeros_like(FR)
                h_src_L = FL * 0.5 + sl_L
                h_src_R = FR * 0.5 + sl_R
            elif SL is not None:
                h_src_L = SL
                h_src_R = SR if SR is not None else SL
            else:
                h_src_L = h_src_R = np.zeros(n)

            if "TFL" not in out:
                shelved = _high_shelf(
                    h_src_L, sr,
                    cfg.height_crossover_hz, cfg.height_low_shelf_gain, cfg.height_max_gain,
                )
                out["TFL"] = cfg.height_gain * _allpass_decorr(shelved, seed=4)
            if "TFR" not in out:
                shelved = _high_shelf(
                    h_src_R, sr,
                    cfg.height_crossover_hz, cfg.height_low_shelf_gain, cfg.height_max_gain,
                )
                out["TFR"] = cfg.height_gain * _allpass_decorr(shelved, seed=5)

            if fmt.n_height_channels == 4:
                if SL is not None:
                    bl_L = BL * 0.3 if BL is not None else np.zeros_like(SL)
                    bl_R = BR * 0.3 if BR is not None else np.zeros_like(SR)
                    hb_src_L = SL * 0.5 + bl_L
                    hb_src_R = SR * 0.5 + bl_R
                else:
                    hb_src_L, hb_src_R = h_src_L, h_src_R

                delay = int(cfg.height_back_delay_ms * sr / 1000.0)
                if "TBL" not in out:
                    shelved = _high_shelf(
                        hb_src_L, sr,
                        cfg.height_crossover_hz, cfg.height_low_shelf_gain, cfg.height_max_gain,
                    )
                    out["TBL"] = cfg.height_gain * _apply_delay(
                        _allpass_decorr(shelved, seed=6), delay
                    )
                if "TBR" not in out:
                    shelved = _high_shelf(
                        hb_src_R, sr,
                        cfg.height_crossover_hz, cfg.height_low_shelf_gain, cfg.height_max_gain,
                    )
                    out["TBR"] = cfg.height_gain * _apply_delay(
                        _allpass_decorr(shelved, seed=7), delay
                    )

        return {label.value: out[label.value] for label in fmt.channels}

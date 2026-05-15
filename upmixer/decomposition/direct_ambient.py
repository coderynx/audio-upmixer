from dataclasses import dataclass

import numpy as np

from upmixer.config import UpmixConfig


@dataclass
class SoftMatrixResult:
    """Result of perceptual spectral decomposition for one STFT frame."""

    center: np.ndarray          # Center channel spectrum (n_freq,)
    front_L: np.ndarray         # Front left spectrum (n_freq,)
    front_R: np.ndarray         # Front right spectrum (n_freq,)
    ambient_L: np.ndarray       # M-S side = (L-R)/2
    ambient_R: np.ndarray       # -(L-R)/2
    signal_L: np.ndarray        # Raw left input (for height extraction)
    signal_R: np.ndarray        # Raw right input
    width: np.ndarray           # Per-bin diffuseness = 1 - coherence (n_freq,)
    transient_score: float      # 0=steady-state, 1=transient (spectral flux)


@dataclass
class SoftMatrixBatchResult:
    """Result of perceptual spectral decomposition for full spectrogram (batch mode)."""

    center: np.ndarray          # (n_freq, n_frames)
    front_L: np.ndarray
    front_R: np.ndarray
    ambient_L: np.ndarray
    ambient_R: np.ndarray
    signal_L: np.ndarray
    signal_R: np.ndarray
    width: np.ndarray           # (n_freq, n_frames)
    transient_score: np.ndarray # (n_frames,) — all zeros in batch mode


class SoftMatrixDecomposer:
    """Perceptual spectral decomposer for music remix upmixing.

    Per STFT frame, performs:
    1. Panning-aware center extraction: center only for coherent, center-panned bins.
    2. Width per bin: diffuseness = 1 - coherence. Router uses this to send only
       diffuse content to surrounds (reverb tails, room, stereo instruments).
    3. Transient detection via spectral flux: spikes in forward flux indicate
       transients. Router uses this to anchor transients in the front field
       and let only sustained/reverberant tails spread to surrounds.
    """

    def __init__(self, config: UpmixConfig):
        self._center_extraction_gain = config.center_extraction_gain
        self._center_attenuation = config.center_attenuation
        self._eps = config.epsilon
        self._flux_threshold = config.transient_flux_threshold

        # Transient detection state
        self._prev_mag: np.ndarray | None = None

    def decompose_frame(
        self,
        X_L_frame: np.ndarray,
        X_R_frame: np.ndarray,
        coherence_frame: np.ndarray,
    ) -> SoftMatrixResult:
        mid = (X_L_frame + X_R_frame) * 0.5
        side = (X_L_frame - X_R_frame) * 0.5

        mag_L = np.abs(X_L_frame)
        mag_R = np.abs(X_R_frame)
        pan = (mag_L - mag_R) / (mag_L + mag_R + self._eps)

        # Center only where coherent AND center-panned
        center_weight = coherence_frame * (1.0 - np.abs(pan))
        center = self._center_extraction_gain * center_weight * mid

        reduction = self._center_attenuation * center_weight * 0.5
        front_L = X_L_frame * (1.0 - reduction)
        front_R = X_R_frame * (1.0 - reduction)

        # Per-bin diffuseness: 1=fully diffuse/ambient, 0=fully coherent/direct
        width = 1.0 - coherence_frame

        # Transient score via spectral flux (positive-only onset detector)
        mag = mag_L + mag_R
        if self._prev_mag is not None:
            positive_flux = np.sum(np.maximum(0.0, mag - self._prev_mag))
            total_energy = np.sum(mag) + self._eps
            flux = positive_flux / total_energy
            transient_score = float(np.clip(flux / self._flux_threshold, 0.0, 1.0))
        else:
            transient_score = 0.0
        self._prev_mag = mag

        return SoftMatrixResult(
            center=center,
            front_L=front_L,
            front_R=front_R,
            ambient_L=side,
            ambient_R=-side,
            signal_L=X_L_frame,
            signal_R=X_R_frame,
            width=width,
            transient_score=transient_score,
        )

    def decompose(
        self,
        X_L: np.ndarray,
        X_R: np.ndarray,
        coherence: np.ndarray,
    ) -> SoftMatrixBatchResult:
        """Batch mode: process full spectrograms. Transient detection disabled."""
        mid = (X_L + X_R) * 0.5
        side = (X_L - X_R) * 0.5

        mag_L = np.abs(X_L)
        mag_R = np.abs(X_R)
        pan = (mag_L - mag_R) / (mag_L + mag_R + self._eps)

        center_weight = coherence * (1.0 - np.abs(pan))
        center = self._center_extraction_gain * center_weight * mid

        reduction = self._center_attenuation * center_weight * 0.5
        front_L = X_L * (1.0 - reduction)
        front_R = X_R * (1.0 - reduction)

        n_frames = X_L.shape[1] if X_L.ndim > 1 else 1
        return SoftMatrixBatchResult(
            center=center,
            front_L=front_L,
            front_R=front_R,
            ambient_L=side,
            ambient_R=-side,
            signal_L=X_L,
            signal_R=X_R,
            width=1.0 - coherence,
            transient_score=np.zeros(n_frames),
        )

    def reset(self) -> None:
        self._prev_mag = None

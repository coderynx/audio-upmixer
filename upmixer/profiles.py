"""Delivery target profiles for multichannel audio output.

Each profile bundles loudness, sample rate, bit depth, and container
requirements for a specific distribution platform. Individual CLI flags
always override profile defaults.

Built-in profiles
-----------------
atmos-music    Dolby Atmos Music Master Delivery Specification v2022.07
               Apple Music, Amazon Music, Tidal, etc.
               -18 LKFS / -1 dBTP / 48 kHz / ADM-BWF / LFE 120 Hz

atmos-bluray   Dolby Atmos Blu-ray via TrueHD/MLP
               -27 LKFS / -2 dBTP / 48 kHz / WAV → TrueHD encoder
               (Netflix home-mix convention, BS.1770-1 dialog-gated)

Extending
---------
Add a new ``DeliveryProfile`` instance to the ``PROFILES`` dict.
Individual parameters not controlled by the profile (e.g. channel
routing gains) keep their ``UpmixConfig`` defaults.
"""
from __future__ import annotations

from dataclasses import dataclass


@dataclass(frozen=True)
class DeliveryProfile:
    """All platform-specific delivery constants for one target format.

    The tool enforces loudness, sample rate, and container.  Codec encoding
    (TrueHD) is handled downstream; ``codec_note`` describes the expected
    next step so users know what to feed this output to.
    """

    name: str
    display_name: str
    description: str

    # Loudness (ITU-R BS.1770-4 unless noted in description)
    loudness_target_lkfs: float
    loudness_max_tp: float

    # Audio format
    sample_rate: int   # required output sample rate in Hz
    bit_depth: int     # PCM bit depth (16 / 24 / 32)
    output_type: str   # "wav" or "adm-bwf"

    # LFE low-pass cutoff (Hz). None = use UpmixConfig default (120 Hz).
    # Spec-mandated range: 100–150 Hz (Dolby Atmos Music Master Delivery
    # Specification v2022.07 §4.3).
    lfe_cutoff_hz: int | None = None

    # Informational — describes the downstream encoding step
    codec_note: str = ""

    @property
    def output_subtype(self) -> str:
        """UpmixConfig output_subtype string derived from bit_depth."""
        return f"PCM_{self.bit_depth}"


# ── Built-in profiles ─────────────────────────────────────────────────────────

PROFILES: dict[str, DeliveryProfile] = {
    "atmos-music": DeliveryProfile(
        name="atmos-music",
        display_name="Dolby Atmos Music (Streaming)",
        description=(
            "Dolby Atmos Music Master Delivery Specification v2022.07. "
            "Apple Music, Amazon Music, Tidal, etc. "
            "ADM-BWF container; loudness -18 LKFS / -1 dBTP / LFE ≤120 Hz."
        ),
        loudness_target_lkfs=-18.0,
        loudness_max_tp=-1.0,
        sample_rate=48_000,
        bit_depth=24,
        output_type="adm-bwf",
        lfe_cutoff_hz=120,
        codec_note=(
            "ADM-BWF / ITU-R BS.2076-2 (24-bit LPCM + XML metadata). "
            "Feed to a Dolby Media Encoder (DME)."
        ),
    ),

    "atmos-bluray": DeliveryProfile(
        name="atmos-bluray",
        display_name="Dolby Atmos Blu-ray (TrueHD/MLP)",
        description=(
            "Dolby Atmos for Blu-ray disc via TrueHD/MLP. "
            "Loudness convention: -27 LKFS dialog-gated (BS.1770-1), ±2 LU tolerance. "
            "Outputs multichannel WAV for ingestion into a TrueHD encoder."
        ),
        loudness_target_lkfs=-27.0,
        loudness_max_tp=-2.0,
        sample_rate=48_000,
        bit_depth=24,
        output_type="wav",
        codec_note=(
            "Feed output WAV to a Dolby Media Encore (DME)."
            "96 kHz supported on Blu-ray (≤8ch); override with --output-sample-rate 96000."
        ),
    ),
}

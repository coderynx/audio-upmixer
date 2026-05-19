"""Mastering package — post-mixing tonal, dynamic, and loudness processing.

Public API (re-exported for backward compatibility)::

    from upmixer.mastering import MasteringChain, MasteringResult

Sub-modules:
    chain       — MasteringChain orchestrator
    eq          — SpectralShaper + EQ_PROFILES
    eq_match    — EQMatcher + scale_breakpoints
    compressor  — BusCompressor + COMP_PROFILES
    bass        — BassController + BASS_PROFILES
"""
from .chain import MasteringChain, MasteringResult

__all__ = ["MasteringChain", "MasteringResult"]

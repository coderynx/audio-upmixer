"""Backward-compatibility shim — re-exports from upmixer.mastering.eq_match.

Prefer importing from ``upmixer.mastering.eq_match`` directly.
"""
from upmixer.mastering.eq_match import (  # noqa: F401
    _CHANNEL_PROXIES,
    _N_BREAKPOINTS,
    _gaussian_smooth_log,
    EQMatcher,
    scale_breakpoints,
)

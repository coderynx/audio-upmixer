#!/usr/bin/env python3
"""Emit a deterministic StemRouter fixture for the browser parity check.

This is a routing-only check. It supplies the Python route scale to the WASM
configuration and uses ``master: {}``; it does not measure normalization or
mastering parity.
"""

import json
import os
import sys

import numpy as np

from upmixer.config import UpmixConfig
from upmixer.formats import FORMAT_MAP
from upmixer.separation.stem_router import StemRouter


SAMPLE_RATE = 48_000
FRAMES = int(os.environ.get("PARITY_FRAMES", "4096"))


def source(mono: bool = False) -> np.ndarray:
    # Quantize before routing so the Python and f32 WASM inputs are identical.
    left = np.asarray([((i * 37 + 13) % 1001 - 500) / 1000 for i in range(FRAMES)], dtype=np.float32)
    right = left if mono else np.asarray(
        [((i * 97 + 271) % 1001 - 500) / 1000 for i in range(FRAMES)], dtype=np.float32
    )
    return np.column_stack((left, right)).astype(np.float64)


def render(
    revision: int,
    rear: float,
    height: float,
    trim_db: float = 0.0,
    texture: float = 0.0,
    mono: bool = False,
    cutoff_hz: float = 1400.0,
) -> dict:
    audio = source(mono)
    cfg = UpmixConfig(
        output_format="7.1.4",
        output_type="multichannel",
        stem_routing={"Probe": {"FL": 1.0, "FR": 1.0}},
        stem_ambient_rear={"Probe": rear},
        stem_ambient_height={"Probe": height},
        stem_ambient_trim_db={"Probe": trim_db},
        stem_height_texture={"Probe": texture},
        stem_ambient_height_crossover_hz={"Probe": 2100.0},
        stem_ambient_height_cutoff_hz={"Probe": cutoff_hz},
    )
    router = StemRouter(cfg, FORMAT_MAP["7.1.4"], SAMPLE_RATE)
    scales = []
    original = router._route_scale

    def record_scale(*args, **kwargs):
        scale = original(*args, **kwargs)
        scales.append(scale)
        return scale

    router._route_scale = record_scale
    routed = router.route({"Probe": audio}, FRAMES)
    return {
        "route_scale": scales[0],
        "rear": rear,
        "height": height,
        "trim_db": trim_db,
        "texture": texture,
        "mono": mono,
        "cutoff_hz": cutoff_hz,
        "channels": [routed[channel.value].tolist() for channel in FORMAT_MAP["7.1.4"].channels],
    }


def main() -> None:
    if FRAMES < 1:
        raise ValueError("PARITY_FRAMES must be a positive integer")
    cases = {}
    for revision in (1, 2):
        for name, rear, height in (
            ("rear", 0.65, 0.0),
            ("height", 0.0, 0.65),
            ("combined", 0.65, 0.65),
        ):
            cases[f"rev{revision}-{name}"] = render(revision, rear, height)
    cases["rev2-trim"] = render(2, 0.65, 0.65, trim_db=6.0, cutoff_hz=1333.0)
    cases["rev2-texture"] = render(2, 0.0, 0.0, texture=0.25, cutoff_hz=1333.0)
    cases["rev2-combined-enhanced"] = render(
        2, 0.65, 0.65, trim_db=6.0, texture=0.25, cutoff_hz=1333.0
    )
    cases["rev2-mono-texture"] = render(
        2, 0.0, 0.0, texture=0.25, mono=True, cutoff_hz=1333.0
    )
    json.dump({"sample_rate": SAMPLE_RATE, "frames": FRAMES, "cases": cases}, sys.stdout)


if __name__ == "__main__":
    main()

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
from upmixer.binaural.decoder import load_decode_filter_set
from upmixer.binaural.geometry import speaker_azimuth_elevation
from upmixer.binaural.profiles import decode_filter_set_name
from upmixer.binaural.renderer import render_binaural
from upmixer.crosstalk.filters import load_xtc_filter_set
from upmixer.crosstalk.renderer import render_crosstalk
from upmixer.formats import ADM_DELIVERY_LAYOUTS, BINAURAL_BED_FORMATS, FORMAT_MAP
from upmixer.movement import MOVEMENT_TUNING_DEFAULTS, compile_movement_schedule, extract_feature_sidecar
from upmixer.separation.stem_router import StemRouter


SAMPLE_RATE = 48_000
FRAMES = int(os.environ.get("PARITY_FRAMES", "4096"))
MOVEMENT_FRAMES = int(os.environ.get("PARITY_MOVEMENT_FRAMES", str(max(FRAMES, 9_600))))


def source(mono: bool = False) -> np.ndarray:
    # Quantize before routing so the Python and f32 WASM inputs are identical.
    left = np.asarray([((i * 37 + 13) % 1001 - 500) / 1000 for i in range(FRAMES)], dtype=np.float32)
    right = left if mono else np.asarray(
        [((i * 97 + 271) % 1001 - 500) / 1000 for i in range(FRAMES)], dtype=np.float32
    )
    return np.column_stack((left, right)).astype(np.float64)


def movement_source() -> np.ndarray:
    """Prepared f32 fixture used for movement decisions.

    The preview proxy is deliberately absent here: the sidecar is extracted
    from this lossless prepared source and is the only input to the policy.
    """
    left = np.asarray(
        [0.18 * np.sin(2 * np.pi * 220 * i / SAMPLE_RATE)
         + 0.03 * (((i * 17 + 3) % 101) - 50) / 50 for i in range(MOVEMENT_FRAMES)],
        dtype=np.float32,
    )
    right = np.asarray(
        [0.16 * np.sin(2 * np.pi * 277 * i / SAMPLE_RATE)
         + 0.02 * (((i * 29 + 7) % 97) - 48) / 48 for i in range(MOVEMENT_FRAMES)],
        dtype=np.float32,
    )
    return np.column_stack((left, right))


def _movement_route(channels: list[str]) -> dict[str, float]:
    """A complete route keeps every movement send family live in the fixture."""
    return {
        channel: (0.1 if channel == "LFE" else 0.25 if channel == "C" else 0.55)
        for channel in channels
    }


def _movement_placement(
    azimuth: float,
    elevation: float,
    width: float,
    object_size: float,
    lfe: float,
) -> dict[str, float]:
    return {
        "azimuth_deg": azimuth,
        "elevation_deg": elevation,
        "width_deg": width,
        "object_size": object_size,
        "lfe": lfe,
        "diversity": 0.0,
        "center_level_db": 0.0,
    }


def _movement_speakers(channels: list[str], layout=None) -> tuple[list[str], list[str], list[dict]]:
    directions = {
        channel.value: (position.azimuth_rad, position.elevation_rad)
        for channel, position in (speaker_azimuth_elevation(layout).items() if layout else [])
    }
    shapes = []
    speakers = []
    for index, channel in enumerate(channels):
        if channel in ("FL", "BL", "SL", "TFL", "TBL"):
            shape = "left" if channel == "FL" else (
                "surround_left" if channel in ("BL", "SL") else "height_left"
            )
        elif channel in ("FR", "BR", "SR", "TFR", "TBR"):
            shape = "right" if channel == "FR" else (
                "surround_right" if channel in ("BR", "SR") else "height_right"
            )
        else:
            shape = "mono"
        shapes.append(shape)
        group_gain = (
            0.85 if channel == "C" else
            0.55 if channel in ("BL", "BR", "TFL", "TFR", "TBL", "TBR") else
            0.6 if channel in ("SL", "SR") else 1.0
        )
        speakers.append({
            "name": channel,
            "azimuth_rad": directions.get(channel, (index * 0.1, 0.0))[0],
            "elevation_rad": directions.get(channel, (index * 0.1, 0.0))[1],
            "group_gain": group_gain,
        })
    return shapes, speakers, [
        {"name": channel, "shape": shape}
        for channel, shape in zip(channels, shapes)
    ]


def _movement_engine_params(
    channels: list[str], request: dict, scales: list[float], lock: bool,
    output_mode: str = "native", decode_taps: np.ndarray | None = None,
    xtc_taps: np.ndarray | None = None, layout=None,
) -> dict:
    shapes, speakers, _ = _movement_speakers(channels, layout)
    stems = []
    for stem, scale in zip(request["stems"], scales):
        routing = _movement_route(channels)
        placement = dict(stem["placement"])
        object_mode = stem["object_mode"]
        if object_mode is not None:
            placement.update({
                "gain": 0.8 if stem["stem_key"] == "Piano" else 0.7,
                "channel_lock": stem["channel_lock"],
                "zone_exclusion": stem["zone_exclusion"],
            })
        stems.append({
            "routing": [[channel, gain] for channel, gain in routing.items()],
            "rebalance_db": 0.0,
            "enabled": True,
            "route_scale": scale,
            "object_mode": object_mode,
            "object_placement": placement if object_mode is not None else None,
            "movement": stem["settings"],
        })
    return {
        "speakers": speakers,
        "lfe_index": channels.index("LFE") if "LFE" in channels else None,
        "shapes": shapes,
        "sends": {
            "surround_bass_cutoff_hz": 250.0,
            "height_low_rolloff_hz": 150.0,
            "height_low_rolloff_gain": 0.15,
            "height_crossover_hz": 3000.0,
            "height_high_shelf_gain": 1.5,
            "height_directional_band_hz": 8000.0,
            "height_directional_band_gain": 1.0,
            "lfe_cutoff_hz": 120.0,
            "lfe_filter_order": 4,
            "lfe_gain": 0.31622776601683794,
        },
        "surround_downmix_coeff": 0.7071,
        "height_downmix_coeff": 0.7071,
        "spatial_downmix_lock": lock,
        "stems": stems,
        "master": {},
        "output_mode": output_mode,
        "soft_limit_threshold": 0.0,
        "bypass_mastering": False,
        "decode_taps": [] if decode_taps is None else decode_taps.reshape(-1).tolist(),
        "xtc_taps": [] if xtc_taps is None else xtc_taps.reshape(-1).tolist(),
    }


def movement_cases() -> dict:
    audio = movement_source()
    # The short Toms stem exercises zero padding after a ragged source ends;
    # all three keys remain in the canonical sidecar, so identity checks run
    # on exactly the set the engine receives.
    movement_audio = {
        "Other": audio,
        "Piano": audio[:, ::-1],
        "Toms": audio[: max(1, MOVEMENT_FRAMES - 911)],
    }
    sidecar = extract_feature_sidecar(movement_audio, SAMPLE_RATE)
    features = {entry["stem_key"]: entry["features"] for entry in sidecar["stems"]}
    settings = {
        key: {
            "enabled": True,
            "role": "featured",
            "depth": 0.4,
            "response": 1.0,
            "sensitivity": 0.5,
            "start_s": 0.0,
            "end_s": None,
        }
        for key in movement_audio
    }
    placements = {
        "Other": _movement_placement(48.0, 0.0, 0.0, 0.0, 0.1),
        "Piano": _movement_placement(-35.0, 10.0, 30.0, 0.25, 0.0),
        "Toms": _movement_placement(35.0, -5.0, 0.0, 0.2, 0.0),
    }
    cases = {}
    for layout in ADM_DELIVERY_LAYOUTS:
        channels = [channel.value for channel in FORMAT_MAP[layout].channels]
        route = _movement_route(channels)
        stems = [
            {
                "stem_key": stem_key,
                "stem_name": stem_key,
                "features": features[stem_key],
                "gain_db": 0.0 if stem_key == "Other" else (20.0 * np.log10(0.8 if stem_key == "Piano" else 0.7)),
                "enabled": True,
                "included": True,
                "placement": placements[stem_key],
                "home_gains": [route[channel] for channel in channels] if stem_key == "Other" else [],
                "home_right_gains": [],
                "settings": settings[stem_key],
                "object_mode": "linked-stereo" if stem_key == "Piano" else ("mono" if stem_key == "Toms" else None),
                "channel_lock": False,
                "zone_exclusion": [],
            }
            for stem_key in movement_audio
        ]
        request = {
            "sample_rate": SAMPLE_RATE,
            "duration_frames": MOVEMENT_FRAMES,
            "revision": 11,
            "channels": channels,
            "stems": stems,
            "tuning": MOVEMENT_TUNING_DEFAULTS,
        }
        schedule = compile_movement_schedule(request)
        layout_cases = {}
        routed_by_mode = {}
        for name, lock in (("normal", False), ("downmix-lock", True)):
            cfg = UpmixConfig(
                output_format=layout,
                output_type="multichannel",
                spatial_downmix_lock=lock,
                stem_routing={stem_key: route for stem_key in movement_audio},
                stem_enabled={stem_key: True for stem_key in movement_audio},
                stem_movement=settings,
                stem_movement_tuning=MOVEMENT_TUNING_DEFAULTS,
                stem_placement=placements,
                stem_object_mode={"Piano": "linked-stereo", "Toms": "mono"},
                stem_object_metadata={
                    "Piano": {"gain": 0.8, "importance": 10},
                    "Toms": {"gain": 0.7, "importance": 8},
                },
            )
            router = StemRouter(cfg, FORMAT_MAP[layout], SAMPLE_RATE)
            scales = []
            original_scale = router._route_scale

            def record_scale(*args, **kwargs):
                scale = original_scale(*args, **kwargs)
                scales.append(scale)
                return scale

            router._route_scale = record_scale
            routed = router.route(movement_audio, MOVEMENT_FRAMES, movement_features=sidecar)
            expected = [routed[channel].tolist() for channel in channels]
            routed_by_mode[name] = routed
            layout_cases[name] = {
                "params": _movement_engine_params(channels, request, scales, lock, layout=FORMAT_MAP[layout]),
                "expected": expected,
                "route_scales": scales,
            }
        collapse_cases = {}
        if layout in BINAURAL_BED_FORMATS:
            decode_taps = load_decode_filter_set(
                decode_filter_set_name("flat", FORMAT_MAP[layout]), SAMPLE_RATE
            ).taps
            xtc_taps = load_xtc_filter_set("stereo_xtc", SAMPLE_RATE).taps
            for name, lock in (("normal", False), ("downmix-lock", True)):
                bed = routed_by_mode[name]
                binaural = render_binaural(
                    bed, FORMAT_MAP[layout], SAMPLE_RATE, "flat",
                    lfe_already_processed=True,
                )
                transaural = render_crosstalk(
                    bed, FORMAT_MAP[layout], SAMPLE_RATE, "stereo",
                    lfe_already_processed=True,
                )
                collapse_cases[name] = {
                    "binaural": {
                        "params": _movement_engine_params(
                            channels, request, layout_cases[name]["route_scales"], lock,
                            output_mode="binaural", decode_taps=decode_taps,
                            layout=FORMAT_MAP[layout],
                        ),
                        "expected": [binaural[0].tolist(), binaural[1].tolist()],
                    },
                    "transaural": {
                        "params": _movement_engine_params(
                            channels, request, layout_cases[name]["route_scales"], lock,
                            output_mode="transaural", decode_taps=decode_taps,
                            xtc_taps=xtc_taps,
                            layout=FORMAT_MAP[layout],
                        ),
                        "expected": [transaural[0].tolist(), transaural[1].tolist()],
                    },
                }
        cases[layout] = {
            "request": request,
            "schedule": schedule,
            "audio_keys": list(movement_audio),
            "block_sizes": [1, 127, 128, 511, 1024],
            "cases": layout_cases,
            "collapse_cases": collapse_cases,
        }
    return {"audio": {
        key: {"left": value[:, 0].tolist(), "right": value[:, 1].tolist()}
        for key, value in movement_audio.items()
    }, "layouts": cases}


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
    if FRAMES < 1 or MOVEMENT_FRAMES < 1:
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
    json.dump({
        "sample_rate": SAMPLE_RATE,
        "frames": FRAMES,
        "cases": cases,
        "movement_cases": movement_cases(),
    }, sys.stdout)


if __name__ == "__main__":
    main()

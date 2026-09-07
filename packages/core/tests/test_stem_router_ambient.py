"""Focused coverage for revision-aware ambient and texture routing."""
from __future__ import annotations

import math

import numpy as np
import upmixer_dsp

from upmixer.config import UpmixConfig
from upmixer.formats import FORMAT_MAP, ChannelLabel, OutputFormat
from upmixer.separation.stem_router import StemRouter

from test_stem_router import _reverberant


def _router(**kwargs: object) -> StemRouter:
    config = UpmixConfig(output_format="7.1.4", **kwargs)
    return StemRouter(config, FORMAT_MAP["7.1.4"], 48000)


def test_uses_shared_ambient_route_and_height_cutoff(monkeypatch):
    calls: dict[str, object] = {}

    def ambient_route(
        left, right, sample_rate, rear, height, cutoff,
        rear_left=True, rear_right=True, height_left=True, height_right=True,
    ):
        calls.update({
            "sample_rate": sample_rate,
            "rear": rear,
            "height": height,
            "cutoff": cutoff,
            "rear_left": rear_left,
            "rear_right": rear_right,
            "height_left": height_left,
            "height_right": height_right,
        })
        direct_left = np.zeros_like(left)
        direct_right = np.zeros_like(right)
        raw_left = np.zeros_like(left)
        raw_right = np.zeros_like(right)
        rear_left = np.ones_like(left)
        rear_right = np.ones_like(right)
        height_left = np.ones_like(left)
        height_right = np.ones_like(right)
        return (
            direct_left, direct_right, raw_left, raw_right,
            rear_left, rear_right, height_left, height_right,
        )

    monkeypatch.setattr(upmixer_dsp, "ambient_route", ambient_route, raising=False)
    routing = {"Other": {ch: 0.0 for ch in (
        "SL", "SR", "BL", "BR", "TFL", "TFR", "TBL", "TBR",
    )}}
    routing["Other"].update({"FL": 1.0, "FR": 1.0})
    router = _router(
        stem_routing=routing,
        stem_ambient_rear={"Other": 0.2},
        stem_ambient_height={"Other": 0.3},
        stem_ambient_height_crossover_hz={"Other": 3999.0},
        stem_ambient_height_cutoff_hz={"Other": 1333.0},
    )
    monkeypatch.setattr(router, "_surround_send", lambda signal: signal)
    monkeypatch.setattr(router, "_height_send", lambda signal: signal)
    rendered = router.route({"Other": np.ones((32, 2))}, 32)

    assert calls == {
        "sample_rate": 48000,
        "rear": 0.2,
        "height": 0.3,
        "cutoff": 1333.0,
        "rear_left": True,
        "rear_right": True,
        "height_left": True,
        "height_right": True,
    }
    assert np.max(np.abs(rendered["FL"])) == 0.0
    assert np.max(np.abs(rendered["SL"])) > 0.0
    assert np.max(np.abs(rendered["TFL"])) > 0.0


def test_scales_ambient_per_source_side_and_gates_absent_side(monkeypatch):
    calls: dict[str, bool] = {}

    def ambient_route(
        left, right, sample_rate, rear, height, cutoff,
        rear_left=True, rear_right=True, height_left=True, height_right=True,
    ):
        calls.update({
            "rear_left": rear_left,
            "rear_right": rear_right,
            "height_left": height_left,
            "height_right": height_right,
        })
        direct_left = np.zeros_like(left)
        direct_right = np.zeros_like(right)
        ambient_left = np.ones_like(left)
        ambient_right = np.ones_like(right)
        return (
            direct_left, direct_right, ambient_left, ambient_right,
            ambient_left, ambient_right, ambient_left, ambient_right,
        )

    monkeypatch.setattr(upmixer_dsp, "ambient_route", ambient_route, raising=False)
    fmt = OutputFormat(
        "asymmetric",
        (
            ChannelLabel.FL, ChannelLabel.FR, ChannelLabel.C,
            ChannelLabel.SL, ChannelLabel.BL, ChannelLabel.SR,
            ChannelLabel.TFL,
        ),
    )
    routing = {"Other": {label.value: 1.0 for label in fmt.channels}}
    router = StemRouter(
        UpmixConfig(
            output_format="7.1.4",
            stem_routing=routing,
            stem_ambient_rear={"Other": 0.5},
            stem_ambient_height={"Other": 0.25},
        ),
        fmt,
        48000,
    )
    monkeypatch.setattr(router, "_surround_send", lambda signal: signal)
    monkeypatch.setattr(router, "_height_send", lambda signal: signal)
    monkeypatch.setattr(router, "_route_scale", lambda *args: 1.0)
    rendered = router.route({"Other": np.ones((8, 2))}, 8)

    assert calls == {
        "rear_left": True,
        "rear_right": True,
        "height_left": True,
        "height_right": False,
    }
    np.testing.assert_allclose(rendered["SL"], 0.5 / math.sqrt(2.0) * 0.6)
    np.testing.assert_allclose(rendered["BL"], 0.5 / math.sqrt(2.0) * 0.55)
    np.testing.assert_allclose(rendered["SR"], 0.5 * 0.6)
    np.testing.assert_allclose(rendered["TFL"], 0.25 * 0.55)


def test_revision_two_ambient_trim_boosts_wet_feeds_only(monkeypatch):
    def ambient_route(
        left, right, sample_rate, rear, height,
        cutoff, rear_left=True, rear_right=True, height_left=True, height_right=True,
    ):
        return (left, right, left, right, left, right, left, right)

    monkeypatch.setattr(upmixer_dsp, "ambient_route", ambient_route, raising=False)
    routing = {"Other": {label.value: 0.0 for label in FORMAT_MAP["7.1.4"].channels}}
    routing["Other"].update({"FL": 1.0, "FR": 1.0})
    common = {
        "stem_routing": routing,
        "stem_ambient_rear": {"Other": 0.5},
    }
    plain_router = _router(**common)
    trimmed_router = _router(**common, stem_ambient_trim_db={"Other": 6.0})
    for router in (plain_router, trimmed_router):
        monkeypatch.setattr(router, "_surround_send", lambda signal: signal)
        monkeypatch.setattr(router, "_height_send", lambda signal: signal)
        monkeypatch.setattr(router, "_route_scale", lambda *args: 1.0)

    audio = np.ones((16, 2))
    plain = plain_router.route({"Other": audio}, len(audio))
    trimmed = trimmed_router.route({"Other": audio}, len(audio))

    np.testing.assert_array_equal(trimmed["FL"], plain["FL"])
    np.testing.assert_allclose(
        trimmed["SL"], plain["SL"] * 10.0 ** (6.0 / 20.0), rtol=1e-12, atol=1e-12
    )


def test_revision_two_height_texture_uses_residual_without_subtraction(monkeypatch):
    routing = {"Other": {"FL": 1.0, "FR": 1.0}}
    plain_router = _router(stem_routing=routing)
    textured_router = _router(
        stem_routing=routing,
        stem_height_texture={"Other": 0.25},
        stem_ambient_height_cutoff_hz={"Other": 1333.0},
    )
    for router in (plain_router, textured_router):
        monkeypatch.setattr(router, "_height_send", lambda signal: signal)
        monkeypatch.setattr(router, "_height_texture_send", lambda signal, cutoff: signal)
        monkeypatch.setattr(router, "_route_scale", lambda *args: 1.0)

    audio = np.ones((16, 2))
    plain = plain_router.route({"Other": audio}, len(audio))
    textured = textured_router.route({"Other": audio}, len(audio))

    np.testing.assert_array_equal(textured["FL"], plain["FL"])
    np.testing.assert_array_equal(textured["FR"], plain["FR"])
    assert np.max(np.abs(textured["TFL"])) > 0.0
    np.testing.assert_array_equal(textured["TFL"], textured["TFR"])


def test_height_texture_send_uses_shared_filter_and_cutoff(monkeypatch):
    calls: dict[str, object] = {}

    def height_texture(signal, sample_rate, cutoff):
        calls.update({"sample_rate": sample_rate, "cutoff": cutoff})
        return signal * 2.0

    monkeypatch.setattr(upmixer_dsp, "height_texture", height_texture, raising=False)
    router = _router()
    monkeypatch.setattr(router, "_height_send", lambda signal: signal + 1.0)

    rendered = router._height_texture_send(np.ones(4), 1333.0)

    assert calls == {"sample_rate": 48000, "cutoff": 1333.0}
    np.testing.assert_array_equal(rendered, np.full(4, 3.0))


"""Tests for upmixer.manifest — load_manifest, apply_manifest, list_manifest_keys."""
from __future__ import annotations

import json
import tempfile
from pathlib import Path

import pytest

from upmixer.config import UpmixConfig
from upmixer.manifest import (
    _FIELD_MAP,
    _JOB_KEYS,
    _MASTERING_KEY_MAP,
    _MIXING_KEY_MAP,
    _expand_nested_sections,
    apply_manifest,
    list_manifest_keys,
    load_manifest,
)


# ---------------------------------------------------------------------------
# Helpers
# ---------------------------------------------------------------------------

def _write_json(directory: str, data: dict, name: str = "job.json") -> str:
    path = str(Path(directory) / name)
    Path(path).write_text(json.dumps(data), encoding="utf-8")
    return path


def _write_yaml(directory: str, data: str, name: str = "job.yaml") -> str:
    """Write raw YAML text (no pyyaml dependency for writing)."""
    path = str(Path(directory) / name)
    Path(path).write_text(data, encoding="utf-8")
    return path


# ---------------------------------------------------------------------------
# load_manifest — JSON
# ---------------------------------------------------------------------------

class TestLoadManifestJson:
    def test_loads_dict(self, tmp_path):
        data = {"input": "in.wav", "output": "out.wav", "format": "7.1.4"}
        path = _write_json(str(tmp_path), data)
        result = load_manifest(path)
        assert result == data

    def test_empty_file_returns_empty_dict(self, tmp_path):
        path = str(tmp_path / "empty.json")
        Path(path).write_text("{}", encoding="utf-8")
        assert load_manifest(path) == {}

    def test_file_not_found(self, tmp_path):
        with pytest.raises(FileNotFoundError):
            load_manifest(str(tmp_path / "missing.json"))

    def test_invalid_extension(self, tmp_path):
        path = str(tmp_path / "job.toml")
        Path(path).write_text("input = 'in.wav'", encoding="utf-8")
        with pytest.raises(ValueError, match="Unrecognised manifest extension"):
            load_manifest(path)

    def test_path_object_accepted(self, tmp_path):
        data = {"format": "5.1"}
        path = tmp_path / "job.json"
        path.write_text(json.dumps(data), encoding="utf-8")
        result = load_manifest(path)
        assert result["format"] == "5.1"


# ---------------------------------------------------------------------------
# load_manifest — YAML
# ---------------------------------------------------------------------------

class TestLoadManifestYaml:
    @pytest.fixture(autouse=True)
    def require_pyyaml(self):
        pytest.importorskip("yaml", reason="pyyaml not installed")

    def test_loads_yaml(self, tmp_path):
        yaml_text = "input: in.wav\noutput: out.wav\nformat: '7.1.4'\n"
        path = _write_yaml(str(tmp_path), yaml_text)
        result = load_manifest(path)
        assert result["input"] == "in.wav"
        assert result["format"] == "7.1.4"

    def test_yml_extension(self, tmp_path):
        yaml_text = "format: '5.1.2'\n"
        path = _write_yaml(str(tmp_path), yaml_text, name="job.yml")
        result = load_manifest(path)
        assert result["format"] == "5.1.2"

    def test_empty_yaml_returns_empty_dict(self, tmp_path):
        path = _write_yaml(str(tmp_path), "")
        assert load_manifest(path) == {}


# ---------------------------------------------------------------------------
# apply_manifest — field mapping
# ---------------------------------------------------------------------------

class TestApplyManifestFields:
    def test_format_maps_to_output_format(self, tmp_path):
        cfg = UpmixConfig()
        job = apply_manifest(cfg, {"format": "7.1.4"})
        assert cfg.output_format == "7.1.4"
        assert job == {}

    def test_output_sample_rate(self, tmp_path):
        cfg = UpmixConfig()
        apply_manifest(cfg, {"output_sample_rate": 48000})
        assert cfg.output_sample_rate == 48000

    def test_lfe_cutoff(self):
        cfg = UpmixConfig()
        apply_manifest(cfg, {"lfe_cutoff": 100.0})
        assert cfg.lfe_cutoff_hz == pytest.approx(100.0)

    def test_loudness_target(self):
        cfg = UpmixConfig()
        apply_manifest(cfg, {"loudness_target": -23.0})
        assert cfg.loudness_target_lkfs == pytest.approx(-23.0)

    def test_loudness_normalize_false(self):
        cfg = UpmixConfig()
        apply_manifest(cfg, {"loudness_normalize": False})
        assert cfg.loudness_normalize is False

    def test_center_gain(self):
        cfg = UpmixConfig()
        apply_manifest(cfg, {"center_gain": 0.9})
        assert cfg.center_gain == pytest.approx(0.9)

    def test_surround_gain(self):
        cfg = UpmixConfig()
        apply_manifest(cfg, {"surround_gain": 0.75})
        assert cfg.surround_gain == pytest.approx(0.75)

    def test_height_gain(self):
        cfg = UpmixConfig()
        apply_manifest(cfg, {"height_gain": 0.4})
        assert cfg.height_gain == pytest.approx(0.4)

    def test_preview_true(self):
        cfg = UpmixConfig()
        apply_manifest(cfg, {"preview": True})
        assert cfg.preview is True

    def test_preview_duration(self):
        cfg = UpmixConfig()
        apply_manifest(cfg, {"preview_duration": 45.0})
        assert cfg.preview_duration_s == pytest.approx(45.0)

    def test_null_value_ignored(self):
        """None / null values must not override config defaults."""
        cfg = UpmixConfig()
        original_lfe = cfg.lfe_cutoff_hz
        apply_manifest(cfg, {"lfe_cutoff": None})
        assert cfg.lfe_cutoff_hz == original_lfe

    def test_multiple_fields_at_once(self):
        cfg = UpmixConfig()
        apply_manifest(cfg, {
            "format": "5.1.4",
            "center_gain": 0.7,
            "surround_gain": 0.5,
            "loudness_target": -16.0,
        })
        assert cfg.output_format == "5.1.4"
        assert cfg.center_gain == pytest.approx(0.7)
        assert cfg.surround_gain == pytest.approx(0.5)
        assert cfg.loudness_target_lkfs == pytest.approx(-16.0)


# ---------------------------------------------------------------------------
# apply_manifest — job params
# ---------------------------------------------------------------------------

class TestApplyManifestJobParams:
    def test_input_returned_in_job_params(self):
        cfg = UpmixConfig()
        job = apply_manifest(cfg, {"input": "in.wav"})
        assert job["input"] == "in.wav"

    def test_output_returned_in_job_params(self):
        cfg = UpmixConfig()
        job = apply_manifest(cfg, {"output": "out.wav"})
        assert job["output"] == "out.wav"

    def test_mode_returned_in_job_params(self):
        cfg = UpmixConfig()
        job = apply_manifest(cfg, {"mode": "stem"})
        assert job["mode"] == "stem"

    def test_stem_model_returned(self):
        cfg = UpmixConfig()
        job = apply_manifest(cfg, {"stem_model": "htdemucs_ft.yaml"})
        assert job["stem_model"] == "htdemucs_ft.yaml"

    def test_stem_model_dir_returned(self):
        cfg = UpmixConfig()
        job = apply_manifest(cfg, {"stem_model_dir": "/tmp/models"})
        assert job["stem_model_dir"] == "/tmp/models"

    def test_absent_job_keys_not_in_dict(self):
        cfg = UpmixConfig()
        job = apply_manifest(cfg, {"format": "7.1"})
        assert "input" not in job
        assert "output" not in job

    def test_profile_not_in_job_params(self):
        """'profile' is consumed by apply_manifest itself, not returned."""
        cfg = UpmixConfig()
        job = apply_manifest(cfg, {"profile": "atmos-music"})
        assert "profile" not in job

    def test_null_job_key_not_in_dict(self):
        cfg = UpmixConfig()
        job = apply_manifest(cfg, {"input": None})
        assert "input" not in job


# ---------------------------------------------------------------------------
# apply_manifest — profile application
# ---------------------------------------------------------------------------

class TestApplyManifestProfile:
    def test_atmos_music_profile_sets_loudness(self):
        cfg = UpmixConfig()
        apply_manifest(cfg, {"profile": "atmos-music"})
        assert cfg.loudness_target_lkfs == pytest.approx(-18.0)
        assert cfg.loudness_max_tp == pytest.approx(-1.0)

    def test_atmos_music_profile_sets_lfe_cutoff(self):
        cfg = UpmixConfig()
        apply_manifest(cfg, {"profile": "atmos-music"})
        assert cfg.lfe_cutoff_hz == pytest.approx(120.0)

    def test_atmos_music_profile_sets_sample_rate(self):
        cfg = UpmixConfig()
        apply_manifest(cfg, {"profile": "atmos-music"})
        assert cfg.output_sample_rate == 48_000

    def test_field_overrides_profile(self):
        """Manifest field listed after profile must win over profile default."""
        cfg = UpmixConfig()
        apply_manifest(cfg, {"profile": "atmos-music", "loudness_target": -16.0})
        assert cfg.loudness_target_lkfs == pytest.approx(-16.0)

    def test_unknown_profile_raises(self):
        cfg = UpmixConfig()
        with pytest.raises(ValueError, match="Unknown profile"):
            apply_manifest(cfg, {"profile": "does-not-exist"})

    def test_atmos_bluray_profile(self):
        cfg = UpmixConfig()
        apply_manifest(cfg, {"profile": "atmos-bluray"})
        assert cfg.loudness_target_lkfs == pytest.approx(-27.0)
        assert cfg.loudness_max_tp == pytest.approx(-2.0)


# ---------------------------------------------------------------------------
# apply_manifest — unknown keys
# ---------------------------------------------------------------------------

class TestApplyManifestUnknownKeys:
    def test_unknown_key_logs_warning(self, caplog):
        import logging
        cfg = UpmixConfig()
        with caplog.at_level(logging.WARNING, logger="upmixer"):
            apply_manifest(cfg, {"definitely_not_a_key": 999})
        assert any("definitely_not_a_key" in r.message for r in caplog.records)

    def test_unknown_key_silenced_with_flag(self, caplog):
        import logging
        cfg = UpmixConfig()
        with caplog.at_level(logging.WARNING, logger="upmixer"):
            apply_manifest(cfg, {"unknown_key": 1}, allow_unknown_keys=True)
        assert not any("unknown_key" in r.message for r in caplog.records)

    def test_bad_type_coercion_raises(self):
        cfg = UpmixConfig()
        with pytest.raises(ValueError, match="center_gain"):
            apply_manifest(cfg, {"center_gain": "not_a_float"})


# ---------------------------------------------------------------------------
# list_manifest_keys
# ---------------------------------------------------------------------------

class TestListManifestKeys:
    def test_returns_dict(self):
        keys = list_manifest_keys()
        assert isinstance(keys, dict)

    def test_format_present(self):
        keys = list_manifest_keys()
        assert "format" in keys

    def test_job_keys_present(self):
        keys = list_manifest_keys()
        for jk in _JOB_KEYS:
            assert jk in keys, f"Job key '{jk}' missing from list_manifest_keys()"

    def test_all_field_map_keys_present(self):
        keys = list_manifest_keys()
        for mk in _FIELD_MAP:
            assert mk in keys, f"Field map key '{mk}' missing from list_manifest_keys()"

    def test_values_are_strings(self):
        for key, desc in list_manifest_keys().items():
            assert isinstance(desc, str), f"Value for key '{key}' is not str"

    def test_mastering_eq_keys_present(self):
        keys = list_manifest_keys()
        assert "mastering_eq_profile" in keys
        assert "mastering_eq_strength" in keys

    def test_mastering_comp_keys_present(self):
        keys = list_manifest_keys()
        for k in [
            "mastering_comp_profile",
            "mastering_comp_threshold_db",
            "mastering_comp_ratio",
            "mastering_comp_attack_ms",
            "mastering_comp_release_ms",
            "mastering_comp_knee_db",
            "mastering_comp_makeup_db",
        ]:
            assert k in keys, f"Missing key '{k}'"


# ---------------------------------------------------------------------------
# Nested mastering: section expansion
# ---------------------------------------------------------------------------

class TestExpandNestedSections:
    def test_no_mastering_key_returns_equivalent(self):
        """Without a mastering: key, output is equivalent to input."""
        data = {"format": "7.1.4", "input": "in.wav"}
        expanded = _expand_nested_sections(data)
        assert expanded == data

    def test_mastering_section_expands_eq_profile(self):
        data = {"mastering": {"eq_profile": "spatial-air"}}
        expanded = _expand_nested_sections(data)
        assert "mastering_eq_profile" in expanded
        assert expanded["mastering_eq_profile"] == "spatial-air"
        assert "mastering" not in expanded

    def test_mastering_section_expands_comp_profile(self):
        data = {"mastering": {"comp_profile": "glue"}}
        expanded = _expand_nested_sections(data)
        assert expanded["mastering_comp_profile"] == "glue"

    def test_mastering_section_loudness_normalize(self):
        data = {"mastering": {"loudness_normalize": False}}
        expanded = _expand_nested_sections(data)
        assert expanded["loudness_normalize"] is False

    def test_mastering_section_loudness_target(self):
        data = {"mastering": {"loudness_target": -23.0}}
        expanded = _expand_nested_sections(data)
        assert expanded["loudness_target"] == pytest.approx(-23.0)

    def test_flat_key_wins_over_nested(self):
        """Top-level flat key takes priority over nested mastering: value."""
        data = {
            "mastering_eq_profile": "spatial-warm",   # flat key
            "mastering": {"eq_profile": "spatial-air"},  # nested key — should lose
        }
        expanded = _expand_nested_sections(data)
        assert expanded["mastering_eq_profile"] == "spatial-warm"

    def test_non_dict_mastering_value_ignored(self):
        """If mastering: is not a dict (e.g. null), no expansion occurs."""
        data = {"mastering": None, "format": "5.1"}
        expanded = _expand_nested_sections(data)
        assert expanded.get("mastering") is None  # unchanged

    def test_full_mastering_section_apply_manifest(self):
        """apply_manifest correctly processes a full mastering: section."""
        cfg = UpmixConfig()
        manifest = {
            "mastering": {
                "eq_profile": "spatial-present",
                "eq_strength": 0.7,
                "comp_profile": "warm",
                "loudness_normalize": False,
                "loudness_target": -16.0,
            }
        }
        apply_manifest(cfg, manifest)
        assert cfg.mastering_eq_profile == "spatial-present"
        assert cfg.mastering_eq_strength == pytest.approx(0.7)
        assert cfg.mastering_comp_profile == "warm"
        assert cfg.loudness_normalize is False
        assert cfg.loudness_target_lkfs == pytest.approx(-16.0)

    def test_mastering_section_comp_threshold(self):
        cfg = UpmixConfig()
        apply_manifest(cfg, {"mastering": {"comp_profile": "glue", "comp_threshold": -20.0}})
        assert cfg.mastering_comp_threshold_db == pytest.approx(-20.0)


# ---------------------------------------------------------------------------
# Profile-linked loudness_normalize
# ---------------------------------------------------------------------------

class TestProfileLinkedLoudness:
    def test_atmos_music_profile_enables_loudness(self):
        cfg = UpmixConfig(loudness_normalize=False)  # start disabled
        apply_manifest(cfg, {"profile": "atmos-music"})
        assert cfg.loudness_normalize is True  # profile forces it on

    def test_atmos_bluray_profile_enables_loudness(self):
        cfg = UpmixConfig(loudness_normalize=False)
        apply_manifest(cfg, {"profile": "atmos-bluray"})
        assert cfg.loudness_normalize is True


# ---------------------------------------------------------------------------
# Nested mixing: section expansion
# ---------------------------------------------------------------------------

class TestExpandMixingSection:
    def test_no_mixing_key_unchanged(self):
        data = {"format": "7.1.4", "input": "in.wav"}
        expanded = _expand_nested_sections(data)
        assert expanded == data

    def test_mixing_section_expands_stem_rebalance(self):
        data = {"mixing": {"stem_rebalance": {"Vocals": 2.0}}}
        expanded = _expand_nested_sections(data)
        assert "stem_rebalance" in expanded
        assert expanded["stem_rebalance"] == {"Vocals": 2.0}
        assert "mixing" not in expanded

    def test_mixing_section_expands_stem_eq(self):
        data = {"mixing": {"stem_eq": {"Vocals": "vocal-presence"}}}
        expanded = _expand_nested_sections(data)
        assert "stem_eq_profiles" in expanded
        assert expanded["stem_eq_profiles"] == {"Vocals": "vocal-presence"}

    def test_flat_key_wins_over_mixing_nested(self):
        data = {
            "stem_rebalance": {"Vocals": 1.0},       # flat key
            "mixing": {"stem_rebalance": {"Vocals": 3.0}},  # nested — should lose
        }
        expanded = _expand_nested_sections(data)
        assert expanded["stem_rebalance"] == {"Vocals": 1.0}

    def test_mixing_and_mastering_both_expanded(self):
        data = {
            "mixing":    {"stem_eq": {"Bass": "bass-warmth"}},
            "mastering": {"eq_profile": "spatial-air"},
        }
        expanded = _expand_nested_sections(data)
        assert "stem_eq_profiles"    in expanded
        assert "mastering_eq_profile" in expanded
        assert "mixing"    not in expanded
        assert "mastering" not in expanded

    def test_mixing_section_non_dict_ignored(self):
        """If mixing: is not a dict, no expansion."""
        data = {"mixing": None, "format": "5.1"}
        expanded = _expand_nested_sections(data)
        assert expanded.get("mixing") is None

    def test_apply_manifest_mixing_section(self):
        cfg = UpmixConfig()
        manifest = {
            "mixing": {
                "stem_rebalance": {"Vocals": 2.0, "Drums": -1.0},
                "stem_eq": {"Bass": "bass-warmth"},
            }
        }
        apply_manifest(cfg, manifest, allow_unknown_keys=False)
        assert cfg.stem_rebalance == {"Vocals": 2.0, "Drums": -1.0}
        assert cfg.stem_eq_profiles == {"Bass": "bass-warmth"}


# ---------------------------------------------------------------------------
# list_manifest_keys — mixing + bass + reference EQ keys
# ---------------------------------------------------------------------------

class TestListManifestKeysMixingBass:
    def test_stem_rebalance_present(self):
        keys = list_manifest_keys()
        assert "stem_rebalance" in keys

    def test_stem_eq_profiles_present(self):
        keys = list_manifest_keys()
        assert "stem_eq_profiles" in keys

    def test_mastering_bass_keys_present(self):
        keys = list_manifest_keys()
        for k in [
            "mastering_bass_profile",
            "mastering_bass_sub_gain_db",
            "mastering_bass_mid_gain_db",
            "mastering_bass_mono_cutoff_hz",
            "mastering_bass_excite",
            "mastering_bass_lfe_gain_db",
        ]:
            assert k in keys, f"Missing key '{k}'"

    def test_mastering_eq_reference_present(self):
        keys = list_manifest_keys()
        assert "mastering_eq_reference" in keys

    def test_mixing_key_map_keys_match_field_map(self):
        """All values in _MIXING_KEY_MAP that point to config fields are in _FIELD_MAP."""
        for manifest_key, flat_key in _MIXING_KEY_MAP.items():
            assert flat_key in _FIELD_MAP, (
                f"_MIXING_KEY_MAP value '{flat_key}' not in _FIELD_MAP"
            )

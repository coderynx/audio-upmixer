//! Layout-adaptive stem-routing presets.

use super::panner::{has_height, StemPlacement};

const fn placement(
    azimuth_deg: f64,
    elevation_deg: f64,
    width_deg: f64,
    object_size: f64,
    lfe: f64,
) -> StemPlacement {
    StemPlacement::new(azimuth_deg, elevation_deg, width_deg, object_size, lfe)
}

/// Preset names, in the order they are offered.
pub const PRESET_NAMES: [&str; 6] = ["balanced", "intimate", "stage", "wide", "immersive", "live"];
pub const PRESET_STEMS: [&str; 16] = [
    "Lead Vocals",
    "Vocals",
    "Backing Vocals",
    "Bass",
    "Kick",
    "Snare",
    "Toms",
    "Drums",
    "Hi-Hat",
    "Ride",
    "Crash",
    "Guitar",
    "Piano",
    "Other",
    "Instrumental",
    "Crowd",
];
const REFERENCE_CHANNELS: [&str; 12] = [
    "FL", "FR", "C", "LFE", "SL", "SR", "BL", "BR", "TFL", "TFR", "TBL", "TBR",
];

/// Everything a preset decides for one stem before layout realization.
#[derive(Clone, Copy, Debug, PartialEq)]
pub struct PresetTreatment {
    pub placement: StemPlacement,
    pub ambient_rear: f64,
    pub ambient_height: f64,
    pub ambient_trim_db: f64,
    pub height_texture: f64,
    pub ambient_height_cutoff_hz: f64,
    pub ambient_height_crossover_hz: f64,
}

#[derive(Clone, Copy)]
struct Profile {
    height: f64,
    width: f64,
    rear_send: f64,
    height_send: f64,
    enhancement_scale: f64,
    vertical_scale: f64,
}

fn profile(name: &str) -> Option<Profile> {
    Some(match name {
        "intimate" => Profile {
            height: 0.0,
            width: 42.0,
            rear_send: 0.45,
            height_send: 0.30,
            enhancement_scale: 0.35,
            vertical_scale: 0.35,
        },
        "balanced" => Profile {
            height: 0.0,
            width: 56.0,
            rear_send: 0.60,
            height_send: 0.45,
            enhancement_scale: 0.55,
            vertical_scale: 0.55,
        },
        "stage" => Profile {
            height: 0.0,
            width: 52.0,
            rear_send: 0.65,
            height_send: 0.45,
            enhancement_scale: 0.70,
            vertical_scale: 0.70,
        },
        "wide" => Profile {
            height: 0.0,
            width: 46.0,
            rear_send: 0.70,
            height_send: 0.50,
            enhancement_scale: 0.80,
            vertical_scale: 0.80,
        },
        "immersive" => Profile {
            height: 28.0,
            width: 50.0,
            rear_send: 0.75,
            height_send: 0.70,
            enhancement_scale: 1.0,
            vertical_scale: 1.0,
        },
        "live" => Profile {
            height: 22.0,
            width: 56.0,
            rear_send: 0.85,
            height_send: 0.60,
            enhancement_scale: 0.90,
            vertical_scale: 0.90,
        },
        _ => return None,
    })
}

#[derive(Clone, Copy)]
struct StemEnhancement {
    texture: f64,
    trim_db: f64,
    cutoff_hz: f64,
}

fn enhancement(stem: &str) -> StemEnhancement {
    let (texture, trim_db, cutoff_hz) = match stem {
        "Lead Vocals" | "Vocals" => (0.02, 3.0, 1800.0),
        "Backing Vocals" => (0.12, 4.5, 1000.0),
        "Bass" | "Kick" => (0.0, 2.0, 2500.0),
        "Snare" | "Toms" => (0.06, 3.5, 1400.0),
        "Drums" => (0.10, 4.0, 1400.0),
        "Hi-Hat" | "Ride" | "Crash" => (0.22, 4.0, 900.0),
        "Guitar" | "Piano" => (0.14, 4.0, 1200.0),
        "Other" | "Instrumental" => (0.12, 4.5, 1200.0),
        "Crowd" => (0.20, 5.0, 700.0),
        _ => unreachable!("only preset stems are passed here"),
    };
    StemEnhancement {
        texture,
        trim_db,
        cutoff_hz,
    }
}

fn is_anchor(stem: &str) -> bool {
    matches!(stem, "Lead Vocals" | "Vocals" | "Bass" | "Kick" | "Snare")
}

fn lfe(stem: &str) -> f64 {
    match stem {
        "Bass" => 0.72,
        "Kick" => 0.82,
        "Toms" => 0.18,
        "Drums" => 0.28,
        "Instrumental" => 0.38,
        _ => 0.0,
    }
}

fn vertical_elevation(stem: &str, profile: Profile, has_height: bool) -> f64 {
    if !has_height {
        return 0.0;
    }
    let base = match stem {
        "Lead Vocals" | "Kick" => 0.0,
        "Bass" => 2.0,
        "Vocals" => 3.0,
        "Snare" => 5.0,
        "Backing Vocals" | "Drums" | "Piano" => 8.0,
        "Toms" | "Instrumental" => 10.0,
        "Guitar" => 12.0,
        "Other" => 14.0,
        "Crowd" => 16.0,
        "Hi-Hat" | "Crash" => 18.0,
        "Ride" => 20.0,
        _ => unreachable!("only preset stems are passed here"),
    };
    base * profile.vertical_scale
}

fn anchor_placement(stem: &str, elevation_deg: f64) -> StemPlacement {
    match stem {
        "Lead Vocals" => placement(0.0, elevation_deg, 60.0, 0.10, 0.0).with_bed_controls(0.0, 1.5),
        "Vocals" => placement(0.0, elevation_deg, 32.0, 0.12, 0.0).with_bed_controls(0.0, 0.6),
        "Bass" => placement(0.0, elevation_deg, 50.0, 0.06, lfe(stem)).with_bed_controls(0.0, 0.5),
        "Kick" => placement(0.0, elevation_deg, 38.0, 0.05, lfe(stem)).with_bed_controls(0.0, 1.0),
        "Snare" => placement(0.0, elevation_deg, 42.0, 0.08, 0.0).with_bed_controls(0.0, 0.8),
        _ => unreachable!("only anchors are passed here"),
    }
}

fn secondary_index(stem: &str, stems: &[&str]) -> Option<usize> {
    PRESET_STEMS
        .iter()
        .filter(|candidate| !is_anchor(candidate) && stems.contains(candidate))
        .position(|candidate| *candidate == stem)
}

fn secondary_placement(
    stem: &str,
    rear: bool,
    elevation_deg: f64,
    profile: Profile,
) -> StemPlacement {
    // Keep the direct image centered left/right; preserve its front/rear and height position.
    placement(
        if rear { 180.0 } else { 0.0 },
        elevation_deg,
        profile.width,
        0.20,
        lfe(stem),
    )
    .with_bed_controls(
        if rear { 0.18 } else { 0.08 },
        if rear { -1.0 } else { -0.2 },
    )
}

fn treatment(profile: Profile, stem: &str, stems: &[&str], channels: &[&str]) -> PresetTreatment {
    let count = stems.iter().filter(|stem| !is_anchor(stem)).count();
    let rich = count >= 3;
    let height_available = rich && has_height(channels);
    let enhancement = enhancement(stem);
    // Sparse mixes still contain ambience; only direct placement needs the rich-mix gate.
    let height_count = channels
        .iter()
        .filter(|channel| channel.starts_with('T'))
        .count();
    let surround_available = channels
        .iter()
        .any(|channel| matches!(*channel, "SL" | "SR" | "BL" | "BR"));
    let mix_scale = if rich { 1.0 } else { 0.85 };
    let stem_scale = match stem {
        "Bass" | "Kick" => 0.55,
        "Lead Vocals" | "Vocals" | "Snare" => 0.85,
        _ => 1.0,
    };
    let send_scale = mix_scale * stem_scale;
    let index = secondary_index(stem, stems).unwrap_or(0);
    let rear = rich
        && index % 4 >= 2
        && channels
            .iter()
            .any(|channel| matches!(*channel, "BL" | "BR"));
    let height_zone = height_available && index % 4 >= 2;
    let elevation_deg = vertical_elevation(stem, profile, height_available).max(if height_zone {
        profile.height
    } else {
        0.0
    });
    PresetTreatment {
        placement: if is_anchor(stem) {
            anchor_placement(stem, vertical_elevation(stem, profile, height_available))
        } else {
            secondary_placement(stem, rear, elevation_deg, profile)
        },
        ambient_rear: if surround_available {
            profile.rear_send * send_scale
        } else {
            0.0
        },
        ambient_height: if height_count > 0 {
            profile.height_send * send_scale * if height_count >= 4 { 1.0 } else { 0.85 }
        } else {
            0.0
        },
        ambient_trim_db: if surround_available || height_count > 0 {
            enhancement.trim_db * profile.enhancement_scale * mix_scale
        } else {
            0.0
        },
        // Texture also sends direct residual, so keep anchors tight even with strong ambience.
        height_texture: if height_count > 0 {
            enhancement.texture * profile.enhancement_scale * mix_scale
        } else {
            0.0
        },
        ambient_height_cutoff_hz: enhancement.cutoff_hz * if rich { 1.0 } else { 1.2 },
        ambient_height_crossover_hz: if matches!(
            stem,
            "Hi-Hat" | "Ride" | "Crash" | "Guitar" | "Piano" | "Other" | "Instrumental" | "Crowd"
        ) {
            2000.0
        } else {
            4000.0
        },
    }
}

/// The treatment for every known stem present in one selected track and speaker layout.
pub fn preset_treatments_for_layout(
    preset: &str,
    stems: &[&str],
    channels: &[&str],
) -> Vec<(&'static str, PresetTreatment)> {
    let Some(profile) = profile(preset) else {
        return Vec::new();
    };
    PRESET_STEMS
        .iter()
        .filter(|stem| stems.contains(stem))
        .map(|stem| (*stem, treatment(profile, stem, stems, channels)))
        .collect()
}

/// The complete treatment a preset gives one stem in the full reference stem layout.
pub fn preset_treatment(preset: &str, stem: &str) -> Option<PresetTreatment> {
    preset_treatments_for_layout(preset, &PRESET_STEMS, &REFERENCE_CHANNELS)
        .into_iter()
        .find(|(name, _)| *name == stem)
        .map(|(_, treatment)| treatment)
}

pub fn preset_placement(preset: &str, stem: &str) -> Option<StemPlacement> {
    preset_treatment(preset, stem).map(|treatment| treatment.placement)
}

/// Every stem a preset names, in canonical order.
pub fn preset_stems(preset: &str) -> &'static [&'static str] {
    if profile(preset).is_some() {
        &PRESET_STEMS
    } else {
        &[]
    }
}

pub fn preset_ambient(preset: &str, stem: &str) -> Option<(f64, f64)> {
    let treatment = preset_treatment(preset, stem)?;
    Some((treatment.ambient_rear, treatment.ambient_height))
}

pub fn preset_ambient_height_crossover(preset: &str, stem: &str) -> Option<f64> {
    preset_treatment(preset, stem).map(|treatment| treatment.ambient_height_crossover_hz)
}

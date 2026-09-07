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
    pub ambient_height_crossover_hz: f64,
}

#[derive(Clone, Copy)]
struct Profile {
    height: f64,
    width: f64,
    rear_send: f64,
    height_send: f64,
}

fn profile(name: &str) -> Option<Profile> {
    Some(match name {
        "intimate" => Profile {
            height: 0.0,
            width: 42.0,
            rear_send: 0.06,
            height_send: 0.0,
        },
        "balanced" => Profile {
            height: 0.0,
            width: 56.0,
            rear_send: 0.14,
            height_send: 0.0,
        },
        "stage" => Profile {
            height: 0.0,
            width: 52.0,
            rear_send: 0.18,
            height_send: 0.0,
        },
        "wide" => Profile {
            height: 0.0,
            width: 46.0,
            rear_send: 0.22,
            height_send: 0.0,
        },
        "immersive" => Profile {
            height: 28.0,
            width: 50.0,
            rear_send: 0.30,
            height_send: 0.24,
        },
        "live" => Profile {
            height: 22.0,
            width: 56.0,
            rear_send: 0.34,
            height_send: 0.18,
        },
        _ => return None,
    })
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

fn anchor_placement(stem: &str) -> StemPlacement {
    match stem {
        "Lead Vocals" => placement(0.0, 0.0, 60.0, 0.10, 0.0).with_bed_controls(0.0, 1.5),
        "Vocals" => placement(0.0, 0.0, 32.0, 0.12, 0.0).with_bed_controls(0.0, 0.6),
        "Bass" => placement(0.0, 0.0, 50.0, 0.06, lfe(stem)).with_bed_controls(0.0, 0.5),
        "Kick" => placement(0.0, 0.0, 38.0, 0.05, lfe(stem)).with_bed_controls(0.0, 1.0),
        "Snare" => placement(0.0, 0.0, 42.0, 0.08, 0.0).with_bed_controls(0.0, 0.8),
        _ => unreachable!("only anchors are passed here"),
    }
}

fn secondary_index(stem: &str, stems: &[&str]) -> Option<usize> {
    PRESET_STEMS
        .iter()
        .filter(|candidate| !is_anchor(candidate) && stems.contains(candidate))
        .position(|candidate| *candidate == stem)
}

fn secondary_placement(stem: &str, rear: bool, elevated: bool, profile: Profile) -> StemPlacement {
    // Keep the direct image centered left/right; preserve its front/rear and height position.
    placement(
        if rear { 180.0 } else { 0.0 },
        if elevated { profile.height } else { 0.0 },
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
    if is_anchor(stem) {
        return PresetTreatment {
            placement: anchor_placement(stem),
            ambient_rear: 0.0,
            ambient_height: 0.0,
            ambient_height_crossover_hz: 4000.0,
        };
    }
    let count = stems.iter().filter(|stem| !is_anchor(stem)).count();
    let index = secondary_index(stem, stems).expect("known selected secondary stem");
    let rear = count >= 3
        && index % 4 >= 2
        && channels
            .iter()
            .any(|channel| matches!(*channel, "BL" | "BR"));
    let elevated = count >= 3 && index % 4 >= 2 && profile.height > 0.0 && has_height(channels);
    PresetTreatment {
        placement: secondary_placement(stem, rear, elevated, profile),
        ambient_rear: if rear {
            profile.rear_send
        } else {
            profile.rear_send * 0.4
        },
        ambient_height: if elevated { profile.height_send } else { 0.0 },
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

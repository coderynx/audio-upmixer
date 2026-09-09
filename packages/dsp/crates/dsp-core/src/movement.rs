//! Canonical stem movement analysis, policy, and event evaluation.
//!
//! This module is deliberately independent of audio IO.  Hosts hand it the
//! prepared float32 stem pair and a JSON-safe description of the current mix;
//! every renderer then evaluates the resulting immutable schedule at absolute
//! programme time.

use serde::{Deserialize, Serialize};

use crate::spatial::panner::{direction, object_positions, PannerLayout, StemPlacement};

pub const FEATURE_VERSION: u32 = 1;
pub const FEATURE_WINDOW_US: i64 = 10_000;
pub const EVENT_GRID_US: i64 = 20_000;
pub const INTERPOLATION_US: i64 = 5_208;

const DB_FLOOR: f64 = -240.0;

type ObjectRouteCache = std::collections::HashMap<[u64; 3], Vec<f64>>;

#[derive(Clone, Debug, PartialEq, Serialize, Deserialize)]
pub struct CanonicalFeatures {
    pub version: u32,
    pub sample_rate: u32,
    pub frame_count: usize,
    pub window_frames: usize,
    pub energies: Vec<f64>,
}

impl CanonicalFeatures {
    pub fn validate(&self) -> Result<(), String> {
        if self.version != FEATURE_VERSION {
            return Err(format!(
                "unsupported movement feature version {}",
                self.version
            ));
        }
        if self.sample_rate == 0 || self.window_frames == 0 {
            return Err("movement feature rate and window size must be positive".into());
        }
        let expected_window = ((self.sample_rate as f64 * FEATURE_WINDOW_US as f64 / 1_000_000.0)
            .round() as usize)
            .max(1);
        if self.window_frames != expected_window {
            return Err("movement feature window is not 10 ms at its sample rate".into());
        }
        let expected = self.frame_count.div_ceil(self.window_frames);
        if self.energies.len() != expected {
            return Err(format!(
                "movement feature window count {} does not match expected {expected}",
                self.energies.len()
            ));
        }
        if self
            .energies
            .iter()
            .any(|energy| !energy.is_finite() || *energy < 0.0)
        {
            return Err("movement feature energies must be finite and non-negative".into());
        }
        Ok(())
    }

    pub fn levels_db(&self) -> Vec<f64> {
        self.energies
            .iter()
            .map(|energy| (10.0 * energy.max(1e-24).log10()).max(DB_FLOOR))
            .collect()
    }

    pub fn p95_level_db(&self) -> f64 {
        percentile(&self.levels_db(), 0.95)
    }

    pub fn duration_us(&self) -> i64 {
        ((self.frame_count as u128 * 1_000_000) / self.sample_rate as u128) as i64
    }

    pub fn envelope_db(&self) -> Vec<f64> {
        envelope(&self.levels_db(), 150.0, 500.0)
    }

    pub fn onsets_db(&self) -> Vec<f64> {
        let levels = self.levels_db();
        (0..levels.len())
            .map(|index| {
                let first = index.saturating_sub(20);
                let baseline = percentile(&levels[first..index.max(first + 1)], 0.5);
                (levels[index] - baseline).max(0.0)
            })
            .collect()
    }
}

/// Extract the only feature that crosses the preparation boundary: raw
/// window energy.  The final window uses its actual sample count.
pub fn extract_features(
    left: &[f32],
    right: Option<&[f32]>,
    sample_rate: u32,
) -> Result<CanonicalFeatures, String> {
    if sample_rate == 0 {
        return Err("movement feature sample rate must be positive".into());
    }
    if let Some(channel) = right {
        if channel.len() != left.len() {
            return Err("stereo movement PCM channels must have equal lengths".into());
        }
    }
    let frames = left.len();
    let window_frames =
        ((sample_rate as f64 * FEATURE_WINDOW_US as f64 / 1_000_000.0).round() as usize).max(1);
    let mut energies = Vec::with_capacity(frames.div_ceil(window_frames));
    for start in (0..frames).step_by(window_frames) {
        let end = (start + window_frames).min(frames);
        let count = (end - start) as f64;
        let mut energy = 0.0;
        for index in start..end {
            let l = left[index] as f64;
            if !l.is_finite() {
                return Err("movement PCM must contain finite samples".into());
            }
            energy += match right {
                Some(channel) => {
                    let r = channel[index] as f64;
                    if !r.is_finite() {
                        return Err("movement PCM must contain finite samples".into());
                    }
                    (l * l + r * r) * 0.5
                }
                None => l * l,
            };
        }
        energies.push(energy / count);
    }
    let features = CanonicalFeatures {
        version: FEATURE_VERSION,
        sample_rate,
        frame_count: frames,
        window_frames,
        energies,
    };
    features.validate()?;
    Ok(features)
}

#[derive(Clone, Debug, PartialEq, Serialize, Deserialize)]
pub struct MovementStemFeatures {
    pub stem_key: String,
    pub features: CanonicalFeatures,
}

#[derive(Clone, Debug, PartialEq, Serialize, Deserialize)]
pub struct MovementFeatureSidecar {
    pub version: u32,
    pub sample_rate: u32,
    pub frame_count: usize,
    pub stems: Vec<MovementStemFeatures>,
}

impl MovementFeatureSidecar {
    pub fn validate(&self, expected_stems: &[&str]) -> Result<(), String> {
        if self.version != FEATURE_VERSION {
            return Err(format!(
                "unsupported movement sidecar version {}",
                self.version
            ));
        }
        if self.sample_rate == 0 {
            return Err("movement sidecar sample rate must be positive".into());
        }
        if self
            .stems
            .iter()
            .any(|stem| stem.stem_key.trim().is_empty())
        {
            return Err("movement sidecar stem identities must be non-empty".into());
        }
        let mut keys: Vec<&str> = self
            .stems
            .iter()
            .map(|stem| stem.stem_key.as_str())
            .collect();
        keys.sort_unstable();
        if keys.windows(2).any(|pair| pair[0] == pair[1]) {
            return Err("movement sidecar contains duplicate stem identities".into());
        }
        let mut expected = expected_stems.to_vec();
        expected.sort_unstable();
        if keys != expected {
            return Err("movement sidecar stem identities do not match the prepared stems".into());
        }
        let max_frame_count = self
            .stems
            .iter()
            .map(|stem| stem.features.frame_count)
            .max()
            .unwrap_or(0);
        if self.frame_count != max_frame_count {
            return Err("movement sidecar frame_count must equal its longest stem".into());
        }
        for stem in &self.stems {
            stem.features.validate()?;
            if stem.features.sample_rate != self.sample_rate {
                return Err("movement sidecar stem rates disagree".into());
            }
            if stem.features.frame_count > self.frame_count {
                return Err("movement sidecar stem exceeds programme duration".into());
            }
            if stem.features.duration_us() > self.duration_us() + 1 {
                return Err("movement sidecar stem duration exceeds programme duration".into());
            }
        }
        Ok(())
    }

    fn duration_us(&self) -> i64 {
        ((self.frame_count as u128 * 1_000_000) / self.sample_rate as u128) as i64
    }
}

#[derive(Clone, Copy, Debug, Default, PartialEq, Eq, Serialize, Deserialize)]
#[serde(rename_all = "lowercase")]
pub enum MovementRole {
    #[default]
    Auto,
    Supporting,
    Featured,
}

#[derive(Clone, Debug, PartialEq, Serialize, Deserialize)]
pub struct MovementSettings {
    #[serde(default)]
    pub enabled: bool,
    #[serde(default)]
    pub role: MovementRole,
    #[serde(default = "default_depth")]
    pub depth: f64,
    #[serde(default = "default_response")]
    pub response: f64,
    #[serde(default = "default_sensitivity")]
    pub sensitivity: f64,
    #[serde(default)]
    pub start_s: f64,
    #[serde(default)]
    pub end_s: Option<f64>,
}

impl Default for MovementSettings {
    fn default() -> Self {
        Self {
            enabled: false,
            role: MovementRole::Auto,
            depth: 0.0,
            response: 1.0,
            sensitivity: 0.5,
            start_s: 0.0,
            end_s: None,
        }
    }
}

impl MovementSettings {
    pub fn validate(&self) -> Result<(), String> {
        if !(0.0..=1.0).contains(&self.depth) || !self.depth.is_finite() {
            return Err("movement depth must be finite and in 0..1".into());
        }
        if !(0.5..=2.0).contains(&self.response) || !self.response.is_finite() {
            return Err("movement response must be finite and in 0.5..2".into());
        }
        if !(0.0..=1.0).contains(&self.sensitivity) || !self.sensitivity.is_finite() {
            return Err("movement sensitivity must be finite and in 0..1".into());
        }
        if !self.start_s.is_finite() || self.start_s < 0.0 {
            return Err("movement start_s must be finite and non-negative".into());
        }
        if let Some(end) = self.end_s {
            if !end.is_finite() || end < 0.0 || self.start_s >= end {
                return Err("movement interval must satisfy finite 0 <= start_s < end_s".into());
            }
        }
        Ok(())
    }
}

fn default_depth() -> f64 {
    0.0
}
fn default_response() -> f64 {
    1.0
}
fn default_sensitivity() -> f64 {
    0.5
}

/// Acoustic values supplied by core/profile configuration. The compiler does
/// not choose acoustic defaults; every caller must pass the complete tuning
/// object so preview and export cannot silently diverge.
#[derive(Clone, Debug, PartialEq, Serialize, Deserialize)]
pub struct MovementTuning {
    pub activity_floor_db: f64,
    pub activity_enter_db: f64,
    pub activity_leave_db: f64,
    pub activity_dwell_ms: f64,
    pub activity_exit_dwell_ms: f64,
    pub focus_prominence_db: f64,
    pub focus_share: f64,
    pub focus_qualification_ms: f64,
    pub focus_attack_ms: f64,
    pub focus_release_ms: f64,
    pub vocal_hold_ms: f64,
    pub winner_hold_ms: f64,
    pub challenger_db: f64,
    pub supporting_span_db: f64,
    pub percussion_rise_db: f64,
    pub percussion_gate_db: f64,
    pub percussion_retrigger_ms: f64,
    pub percussion_attack_ms: f64,
    pub percussion_return_ms: f64,
    pub toms_return_ms: f64,
    pub crash_return_ms: f64,
}

impl MovementTuning {
    pub fn validate(&self) -> Result<(), String> {
        let finite = [
            self.activity_floor_db,
            self.activity_enter_db,
            self.activity_leave_db,
            self.activity_dwell_ms,
            self.activity_exit_dwell_ms,
            self.focus_prominence_db,
            self.focus_share,
            self.focus_qualification_ms,
            self.focus_attack_ms,
            self.focus_release_ms,
            self.vocal_hold_ms,
            self.winner_hold_ms,
            self.challenger_db,
            self.supporting_span_db,
            self.percussion_rise_db,
            self.percussion_gate_db,
            self.percussion_retrigger_ms,
            self.percussion_attack_ms,
            self.percussion_return_ms,
            self.toms_return_ms,
            self.crash_return_ms,
        ];
        if finite.iter().any(|value| !value.is_finite()) {
            return Err("movement tuning values must be finite".into());
        }
        if !(-120.0..=0.0).contains(&self.activity_floor_db)
            || !(0.0..=60.0).contains(&self.activity_enter_db)
            || !(0.0..=60.0).contains(&self.activity_leave_db)
            || !(0.0..=60.0).contains(&self.focus_prominence_db)
            || !(0.0..=60.0).contains(&self.percussion_rise_db)
            || !(0.0..=60.0).contains(&self.percussion_gate_db)
        {
            return Err("movement tuning thresholds are outside their valid ranges".into());
        }
        if !(0.0..=1.0).contains(&self.focus_share)
            || self.focus_qualification_ms < 0.0
            || self.focus_attack_ms <= 0.0
            || self.focus_release_ms <= 0.0
            || self.vocal_hold_ms < 0.0
            || self.winner_hold_ms < 0.0
            || self.challenger_db < 0.0
            || self.supporting_span_db <= 0.0
            || self.percussion_retrigger_ms < 0.0
            || self.percussion_attack_ms <= 0.0
            || self.percussion_return_ms <= 0.0
            || self.toms_return_ms <= 0.0
            || self.crash_return_ms <= 0.0
        {
            return Err("movement tuning contains an invalid range".into());
        }
        if self.activity_enter_db < self.activity_leave_db
            || self.activity_dwell_ms < 0.0
            || self.activity_exit_dwell_ms < 0.0
        {
            return Err("movement activity thresholds or dwell values are invalid".into());
        }
        Ok(())
    }
}

#[derive(Clone, Copy, Debug, PartialEq, Eq, Serialize, Deserialize)]
#[serde(rename_all = "kebab-case")]
pub enum MovementObjectMode {
    LinkedStereo,
    Mono,
}

#[derive(Clone, Debug, PartialEq, Serialize, Deserialize)]
pub struct MovementStemInput {
    pub stem_key: String,
    #[serde(default)]
    pub stem_name: String,
    pub features: CanonicalFeatures,
    #[serde(default)]
    pub gain_db: f64,
    #[serde(default = "enabled")]
    pub enabled: bool,
    #[serde(default = "included")]
    pub included: bool,
    pub placement: StemPlacement,
    #[serde(default)]
    pub home_gains: Vec<f64>,
    #[serde(default)]
    pub home_right_gains: Vec<f64>,
    #[serde(default)]
    pub settings: MovementSettings,
    #[serde(default)]
    pub object_mode: Option<MovementObjectMode>,
    #[serde(default)]
    pub channel_lock: bool,
    #[serde(default)]
    pub zone_exclusion: Vec<String>,
}

fn enabled() -> bool {
    true
}
fn included() -> bool {
    true
}

#[derive(Clone, Debug, PartialEq, Serialize, Deserialize)]
pub struct MovementCompileRequest {
    pub sample_rate: u32,
    pub duration_frames: usize,
    #[serde(default)]
    pub revision: u64,
    pub channels: Vec<String>,
    pub stems: Vec<MovementStemInput>,
    pub tuning: MovementTuning,
}

#[derive(Clone, Debug, PartialEq, Serialize, Deserialize)]
pub struct MovementEvent {
    pub time_us: i64,
    pub position: [f64; 3],
    pub gains: Vec<f64>,
    #[serde(default)]
    pub right_position: Option<[f64; 3]>,
    #[serde(default)]
    pub right_gains: Option<Vec<f64>>,
    pub interpolation_us: i64,
}

#[derive(Clone, Debug, PartialEq, Serialize, Deserialize)]
pub struct MovementStemSchedule {
    pub stem_key: String,
    pub stem_index: usize,
    pub events: Vec<MovementEvent>,
}

#[derive(Clone, Debug, PartialEq, Serialize, Deserialize)]
pub struct MovementSchedule {
    pub version: u32,
    pub revision: u64,
    pub sample_rate: u32,
    pub duration_frames: usize,
    pub grid_us: i64,
    pub interpolation_us: i64,
    pub stems: Vec<MovementStemSchedule>,
}

impl MovementSchedule {
    pub fn validate(&self, channels: usize) -> Result<(), String> {
        if self.version != FEATURE_VERSION
            || self.grid_us != EVENT_GRID_US
            || self.interpolation_us != INTERPOLATION_US
            || self.sample_rate == 0
            || self.duration_frames == 0
            || channels == 0
        {
            return Err("invalid movement schedule header".into());
        }
        let duration = self.duration_us();
        let mut stem_indices = Vec::with_capacity(self.stems.len());
        let mut stem_keys = Vec::with_capacity(self.stems.len());
        for stem in &self.stems {
            if stem.stem_key.trim().is_empty()
                || !stem_indices.iter().all(|index| *index != stem.stem_index)
                || !stem_keys.iter().all(|key| key != &stem.stem_key)
            {
                return Err(
                    "movement schedule stem identities must be unique and non-empty".into(),
                );
            }
            stem_indices.push(stem.stem_index);
            stem_keys.push(stem.stem_key.clone());
            if stem.events.is_empty() || stem.events[0].time_us != 0 {
                return Err("movement schedule must begin at programme time zero".into());
            }
            for (event_index, event) in stem.events.iter().enumerate() {
                if event.time_us < 0
                    || event.time_us > duration
                    || event.time_us % EVENT_GRID_US != 0
                    || event.interpolation_us != INTERPOLATION_US
                    || event.gains.len() != channels
                    || event
                        .right_gains
                        .as_ref()
                        .is_some_and(|gains| gains.len() != channels)
                    || event
                        .position
                        .iter()
                        .any(|v| !v.is_finite() || !(-1.0..=1.0).contains(v))
                    || event.gains.iter().any(|v| !v.is_finite() || *v < 0.0)
                    || event.right_position.is_some_and(|position| {
                        position
                            .iter()
                            .any(|v| !v.is_finite() || !(-1.0..=1.0).contains(v))
                    })
                    || event
                        .right_gains
                        .as_ref()
                        .is_some_and(|gains| gains.iter().any(|v| !v.is_finite() || *v < 0.0))
                {
                    return Err(format!("invalid movement event {event_index}"));
                }
                if event_index > 0 && event.time_us <= stem.events[event_index - 1].time_us {
                    return Err("movement events must be strictly increasing".into());
                }
            }
        }
        Ok(())
    }

    pub fn duration_us(&self) -> i64 {
        ((self.duration_frames as u128 * 1_000_000) / self.sample_rate as u128) as i64
    }

    pub fn stem(&self, index: usize) -> Option<&MovementStemSchedule> {
        self.stems.iter().find(|stem| stem.stem_index == index)
    }
}

impl MovementStemSchedule {
    /// Borrow held gains when an entire render block lies outside a transition.
    pub fn constant_gains_for_range(
        &self,
        start_us: f64,
        end_us: f64,
        right: bool,
    ) -> Option<&[f64]> {
        let index = self
            .events
            .partition_point(|event| event.time_us as f64 <= start_us)
            .saturating_sub(1);
        let event = self.events.get(index)?;
        if (index > 0 && start_us < event.time_us as f64 + INTERPOLATION_US as f64)
            || self
                .events
                .get(index + 1)
                .is_some_and(|next| end_us >= next.time_us as f64)
        {
            return None;
        }
        Some(if right {
            event.right_gains.as_deref().unwrap_or(&event.gains)
        } else {
            &event.gains
        })
    }

    /// Evaluate one endpoint with the shared 5208 µs gain transition.
    pub fn sample_into(&self, time_us: i64, right: bool, out: &mut [f64]) {
        self.sample_into_at(time_us as f64, right, out);
    }

    /// Floating-point programme time keeps sample-rate conversion from
    /// accumulating rounded-microsecond error. Event lookup is logarithmic so
    /// a long render does not rescan the whole schedule for every sample.
    pub fn sample_into_at(&self, time_us: f64, right: bool, out: &mut [f64]) {
        let Some(first) = self.events.first() else {
            out.fill(0.0);
            return;
        };
        if self.events.len() == 1 {
            let values = if right {
                self.events[0].right_gains.as_ref().unwrap_or(&first.gains)
            } else {
                &first.gains
            };
            out.copy_from_slice(values);
            return;
        }
        let current_index = self
            .events
            .partition_point(|event| event.time_us as f64 <= time_us)
            .saturating_sub(1);
        let current = &self.events[current_index];
        let previous = current_index
            .checked_sub(1)
            .map(|index| &self.events[index])
            .unwrap_or(current);
        let current_values = if right {
            current.right_gains.as_ref().unwrap_or(&current.gains)
        } else {
            &current.gains
        };
        if current.time_us == first.time_us
            || previous.time_us == current.time_us
            || time_us >= current.time_us as f64 + INTERPOLATION_US as f64
        {
            out.copy_from_slice(current_values);
            return;
        }
        let previous_values = if right {
            previous.right_gains.as_ref().unwrap_or(&previous.gains)
        } else {
            &previous.gains
        };
        let amount = ((time_us - current.time_us as f64) / INTERPOLATION_US as f64).clamp(0.0, 1.0);
        for ((out, before), target) in out.iter_mut().zip(previous_values).zip(current_values) {
            *out = before + (target - before) * amount;
        }
    }

    pub fn position_at(&self, time_us: i64, right: bool) -> [f64; 3] {
        self.position_at_f64(time_us as f64, right)
    }

    pub fn position_at_f64(&self, time_us: f64, right: bool) -> [f64; 3] {
        let Some(first) = self.events.first() else {
            return [0.0; 3];
        };
        let current_index = self
            .events
            .partition_point(|event| event.time_us as f64 <= time_us)
            .saturating_sub(1);
        let current = &self.events[current_index];
        let previous = current_index
            .checked_sub(1)
            .map(|index| &self.events[index])
            .unwrap_or(current);
        let target = if right {
            current.right_position.unwrap_or(current.position)
        } else {
            current.position
        };
        if current.time_us == first.time_us || previous.time_us == current.time_us {
            return target;
        }
        let before = if right {
            previous.right_position.unwrap_or(previous.position)
        } else {
            previous.position
        };
        let amount = ((time_us - current.time_us as f64) / INTERPOLATION_US as f64).clamp(0.0, 1.0);
        [
            before[0] + (target[0] - before[0]) * amount,
            before[1] + (target[1] - before[1]) * amount,
            before[2] + (target[2] - before[2]) * amount,
        ]
    }
}

#[derive(Clone)]
struct Analysis {
    levels: Vec<f64>,
    active: Vec<bool>,
    envelope: Vec<f64>,
    prominence: Vec<f64>,
    hits: Vec<Option<i8>>,
}

pub fn compile_movement(request: &MovementCompileRequest) -> Result<MovementSchedule, String> {
    Ok(PreparedMovement::new(request.clone())?.schedule)
}

/// Control-thread state: placement edits cannot change activity or focus winners.
pub struct PreparedMovement {
    request: MovementCompileRequest,
    analyses: Vec<Analysis>,
    winners: Vec<Option<usize>>,
    pub schedule: MovementSchedule,
}

#[derive(Deserialize)]
pub struct MovementPlacementUpdate {
    pub stem_key: String,
    pub placement: StemPlacement,
    pub home_gains: Vec<f64>,
    pub home_right_gains: Vec<f64>,
}

impl PreparedMovement {
    pub fn new(request: MovementCompileRequest) -> Result<Self, String> {
        if request.sample_rate == 0 || request.channels.is_empty() || request.duration_frames == 0 {
            return Err("movement compilation needs a sample rate and channels".into());
        }
        request.tuning.validate()?;
        let mut keys = request
            .stems
            .iter()
            .map(|stem| stem.stem_key.trim())
            .collect::<Vec<_>>();
        if keys.iter().any(|key| key.is_empty()) {
            return Err("movement stems need non-empty identities".into());
        }
        keys.sort_unstable();
        if keys.windows(2).any(|pair| pair[0] == pair[1]) {
            return Err("movement stems contain duplicate identities".into());
        }
        let names: Vec<&str> = request.channels.iter().map(String::as_str).collect();
        let layout = PannerLayout::new(&names);
        let duration_us =
            ((request.duration_frames as u128 * 1_000_000) / request.sample_rate as u128) as i64;

        let mut analyses = Vec::with_capacity(request.stems.len());
        for stem in &request.stems {
            stem.features.validate()?;
            stem.settings.validate()?;
            if !stem.gain_db.is_finite() {
                return Err(format!("movement gain is not finite for {}", stem.stem_key));
            }
            validate_placement(&stem.placement)?;
            validate_gains(&stem.home_gains, request.channels.len(), "home_gains")?;
            validate_gains(
                &stem.home_right_gains,
                request.channels.len(),
                "home_right_gains",
            )?;
            if stem.features.sample_rate != request.sample_rate {
                return Err(format!("movement rate mismatch for {}", stem.stem_key));
            }
            if stem.features.frame_count > request.duration_frames {
                return Err(format!(
                    "movement features exceed programme duration for {}",
                    stem.stem_key
                ));
            }
            analyses.push(analyse(stem, &request.tuning));
        }
        let winners = winners(&request, &analyses, &request.tuning);
        let mut schedules = Vec::with_capacity(request.stems.len());
        for (index, stem) in request.stems.iter().enumerate() {
            if !stem.included {
                continue;
            }
            schedules.push(compile_stem(
                index,
                stem,
                &analyses[index],
                &winners,
                &layout,
                &names,
                request.duration_frames,
                duration_us,
                request.sample_rate,
                &request.tuning,
            )?);
        }
        let schedule = MovementSchedule {
            version: FEATURE_VERSION,
            revision: request.revision,
            sample_rate: request.sample_rate,
            duration_frames: request.duration_frames,
            grid_us: EVENT_GRID_US,
            interpolation_us: INTERPOLATION_US,
            stems: schedules,
        };
        schedule.validate(request.channels.len())?;
        Ok(Self {
            request,
            analyses,
            winners,
            schedule,
        })
    }

    pub fn update_placements(
        &mut self,
        revision: u64,
        updates: Vec<MovementPlacementUpdate>,
    ) -> Result<&MovementSchedule, String> {
        let names: Vec<&str> = self.request.channels.iter().map(String::as_str).collect();
        let layout = PannerLayout::new(&names);
        let mut replacements = Vec::new();
        for update in updates {
            let index = self
                .request
                .stems
                .iter()
                .position(|stem| stem.stem_key == update.stem_key)
                .ok_or("unknown movement stem")?;
            if replacements
                .iter()
                .any(|(previous, _, _)| *previous == index)
            {
                return Err("duplicate movement placement update".into());
            }
            validate_placement(&update.placement)?;
            validate_gains(&update.home_gains, names.len(), "home_gains")?;
            validate_gains(&update.home_right_gains, names.len(), "home_right_gains")?;
            let mut stem = self.request.stems[index].clone();
            stem.placement = update.placement;
            stem.home_gains = update.home_gains;
            stem.home_right_gains = update.home_right_gains;
            let schedule = if stem.included {
                Some(compile_stem(
                    index,
                    &stem,
                    &self.analyses[index],
                    &self.winners,
                    &layout,
                    &names,
                    self.request.duration_frames,
                    self.schedule.duration_us(),
                    self.request.sample_rate,
                    &self.request.tuning,
                )?)
            } else {
                None
            };
            replacements.push((index, stem, schedule));
        }
        // Commit only after every update passed validation and compilation.
        for (index, stem, schedule) in replacements {
            self.request.stems[index] = stem;
            if let Some(schedule) = schedule {
                *self
                    .schedule
                    .stems
                    .iter_mut()
                    .find(|stem| stem.stem_index == index)
                    .unwrap() = schedule;
            }
        }
        self.schedule.revision = revision;
        Ok(&self.schedule)
    }
}

fn validate_gains(gains: &[f64], channels: usize, label: &str) -> Result<(), String> {
    if !gains.is_empty() && gains.len() != channels {
        return Err(format!("movement {label} must match the channel count"));
    }
    if gains.iter().any(|gain| !gain.is_finite() || *gain < 0.0) {
        return Err(format!("movement {label} must be finite and non-negative"));
    }
    Ok(())
}

fn validate_placement(placement: &StemPlacement) -> Result<(), String> {
    if !placement.azimuth_deg.is_finite()
        || !(-180.0..=180.0).contains(&placement.azimuth_deg)
        || !placement.elevation_deg.is_finite()
        || !(-90.0..=90.0).contains(&placement.elevation_deg)
        || !placement.width_deg.is_finite()
        || placement.width_deg.abs() > 360.0
        || placement
            .left_right
            .is_some_and(|value| !value.is_finite() || !(-1.0..=1.0).contains(&value))
        || placement
            .back_front
            .is_some_and(|value| !value.is_finite() || !(-1.0..=1.0).contains(&value))
        || !placement.object_size.is_finite()
        || !(0.0..=1.0).contains(&placement.object_size)
        || !placement.lfe.is_finite()
        || !(0.0..=1.0).contains(&placement.lfe)
        || !placement.diversity.is_finite()
        || !(0.0..=1.0).contains(&placement.diversity)
        || !(-82.0..=6.0).contains(&placement.center_level_db)
    {
        return Err("movement placement is outside its finite range".into());
    }
    Ok(())
}

fn analyse(stem: &MovementStemInput, tuning: &MovementTuning) -> Analysis {
    let mut levels = stem.features.levels_db();
    for level in &mut levels {
        *level += stem.gain_db;
    }
    let p95 = percentile(&levels, 0.95);
    let floor = tuning.activity_floor_db.max(p95 - 40.0);
    let shift = (stem.settings.sensitivity - 0.5) * 6.0;
    let enter = floor + tuning.activity_enter_db - shift;
    let leave = floor + tuning.activity_leave_db - shift;
    let enter_dwell = (tuning.activity_dwell_ms / 10.0).ceil().max(1.0) as usize;
    let exit_dwell = (tuning.activity_exit_dwell_ms / 10.0).ceil().max(1.0) as usize;
    let mut active = vec![false; levels.len()];
    let mut active_now = false;
    let mut above = 0;
    let mut below = 0;
    for (index, level) in levels.iter().enumerate() {
        if *level >= enter {
            above += 1;
            below = 0;
            if !active_now && above >= enter_dwell {
                active_now = true;
            }
        } else {
            above = 0;
            if *level < leave {
                below += 1;
                if below >= exit_dwell {
                    active_now = false;
                }
            } else {
                // Between the thresholds is a hysteresis band. It neither
                // adds to the exit dwell nor clears existing activity.
                below = 0;
            }
        }
        active[index] = active_now;
    }

    let median = rolling_median(&levels, 5);
    let short_envelope = envelope(&median, 150.0, 500.0);
    let mut baseline = Vec::with_capacity(levels.len());
    for index in 0..levels.len() {
        let first = index.saturating_sub(300);
        baseline.push(percentile(&short_envelope[first..=index], 0.5));
    }
    let prominence = short_envelope
        .iter()
        .zip(&baseline)
        .map(|(level, base)| *level - *base)
        .collect::<Vec<_>>();
    let lookback = 20;
    let rise = tuning.percussion_rise_db - (stem.settings.sensitivity - 0.5) * 6.0;
    let gate = floor + tuning.percussion_gate_db - shift;
    let mut hits = vec![None; levels.len()];
    let mut last_hit: Option<usize> = None;
    let mut side = 1_i8;
    for index in 0..levels.len() {
        let first = index.saturating_sub(lookback);
        let previous = if first < index {
            let mean =
                stem.features.energies[first..index].iter().sum::<f64>() / (index - first) as f64;
            10.0 * mean.max(1e-24).log10() + stem.gain_db
        } else {
            DB_FLOOR
        };
        let refractory = (tuning.percussion_retrigger_ms / 10.0).ceil() as usize;
        if first < index
            && levels[index] >= gate
            && levels[index] - previous >= rise
            && last_hit.is_none_or(|last| index.saturating_sub(last) >= refractory)
        {
            hits[index] = Some(side);
            side = -side;
            last_hit = Some(index);
        }
    }
    Analysis {
        levels,
        active,
        envelope: short_envelope,
        prominence,
        hits,
    }
}

fn winners(
    request: &MovementCompileRequest,
    analyses: &[Analysis],
    tuning: &MovementTuning,
) -> Vec<Option<usize>> {
    let frames = analyses.iter().map(|a| a.levels.len()).max().unwrap_or(0);
    let mut result = vec![None; frames];
    let required = (tuning.focus_qualification_ms / 10.0).ceil().max(1.0) as usize;
    let mut qualified_runs = vec![vec![0usize; frames]; request.stems.len()];
    let prominence_enter =
        |stem: &MovementStemInput| tuning.focus_prominence_db - stem.settings.sensitivity * 5.0;
    let share_enter =
        |stem: &MovementStemInput| tuning.focus_share - stem.settings.sensitivity * 0.5;
    for frame in 0..frames {
        let total_energy = request
            .stems
            .iter()
            .enumerate()
            .filter(|(index, stem)| {
                stem.included
                    && stem.enabled
                    && stem.settings.enabled
                    && stem.settings.depth > 0.0
                    && stem.settings.role == MovementRole::Auto
                    && is_melodic(&stem.stem_name, &stem.stem_key)
                    && analyses[*index].active.get(frame).copied().unwrap_or(false)
            })
            .map(|(index, _)| {
                10.0_f64.powf(
                    analyses[index]
                        .levels
                        .get(frame)
                        .copied()
                        .unwrap_or(DB_FLOOR)
                        / 10.0,
                )
            })
            .sum::<f64>();
        for (index, stem) in request.stems.iter().enumerate() {
            let active = analyses[index].active.get(frame).copied().unwrap_or(false);
            let eligible = stem.included
                && stem.enabled
                && stem.settings.enabled
                && stem.settings.depth > 0.0
                && stem.settings.role == MovementRole::Auto
                && is_melodic(&stem.stem_name, &stem.stem_key)
                && active;
            if !eligible {
                continue;
            }
            let level = analyses[index]
                .levels
                .get(frame)
                .copied()
                .unwrap_or(DB_FLOOR);
            let prominence = analyses[index]
                .prominence
                .get(frame)
                .copied()
                .unwrap_or(DB_FLOOR);
            let share = if total_energy > 0.0 {
                10.0_f64.powf(level / 10.0) / total_energy
            } else {
                0.0
            };
            let enter = prominence >= prominence_enter(stem) || share >= share_enter(stem);
            let retain =
                prominence >= prominence_enter(stem) - 2.0 || share >= share_enter(stem) - 0.15;
            let previous = frame
                .checked_sub(1)
                .map_or(0, |previous| qualified_runs[index][previous]);
            qualified_runs[index][frame] = if enter || (previous > 0 && retain) {
                previous + 1
            } else {
                0
            };
        }
    }

    let mut current = None;
    let mut held_until = 0;
    let mut vocal_until = 0;
    let mut vocal_winner: Option<usize> = None;
    for frame in 0..frames {
        let mut vocal_candidates = request
            .stems
            .iter()
            .enumerate()
            .filter(|(index, stem)| {
                stem.included
                    && stem.enabled
                    && analyses[*index].active.get(frame).copied().unwrap_or(false)
                    && is_vocal(&stem.stem_name, &stem.stem_key)
            })
            .map(|(index, _)| index)
            .collect::<Vec<_>>();
        vocal_candidates
            .sort_by(|a, b| request.stems[*a].stem_key.cmp(&request.stems[*b].stem_key));
        let vocal_active = vocal_winner
            .is_some_and(|index| analyses[index].active.get(frame).copied().unwrap_or(false));
        if let Some(index) = vocal_candidates.first().copied() {
            if vocal_winner.is_none() || !vocal_active {
                vocal_winner = Some(index);
            }
            // The hold is measured from the last active vocal frame. This
            // refresh is intentional while activity continues; a later gap
            // therefore gets the full priority window.
            vocal_until = frame + (tuning.vocal_hold_ms / 10.0).ceil() as usize;
        } else if !vocal_active && frame > vocal_until {
            vocal_winner = None;
        }
        let vocal = vocal_winner.filter(|_| vocal_active || frame <= vocal_until);
        let preferred = if let Some(vocal) = vocal {
            Some(vocal)
        } else {
            let mut candidates = request
                .stems
                .iter()
                .enumerate()
                .filter(|(index, stem)| {
                    stem.included
                        && stem.enabled
                        && stem.settings.enabled
                        && stem.settings.depth > 0.0
                        && stem.settings.role == MovementRole::Auto
                        && analyses[*index].active.get(frame).copied().unwrap_or(false)
                        && is_melodic(&stem.stem_name, &stem.stem_key)
                        && qualified_runs[*index].get(frame).copied().unwrap_or(0) >= required
                })
                .map(|(index, _)| index)
                .collect::<Vec<_>>();
            candidates.sort_by(|a, b| {
                let level_a = analyses[*a].levels.get(frame).copied().unwrap_or(DB_FLOOR);
                let level_b = analyses[*b].levels.get(frame).copied().unwrap_or(DB_FLOOR);
                level_b
                    .total_cmp(&level_a)
                    .then_with(|| request.stems[*a].stem_key.cmp(&request.stems[*b].stem_key))
            });
            candidates.first().copied()
        };

        if let Some(candidate) = preferred {
            let previous = current;
            if previous.is_none() {
                current = Some(candidate);
            } else if previous == Some(candidate) {
                current = previous;
            } else if vocal.is_some() {
                // Vocal priority is the immediate handover rule. The
                // instrumental winner hold protects only instrument-to-
                // instrument changes.
                current = Some(candidate);
            } else if frame >= held_until {
                let old_level = previous
                    .and_then(|index| analyses[index].levels.get(frame).copied())
                    .unwrap_or(DB_FLOOR);
                let new_level = analyses[candidate]
                    .levels
                    .get(frame)
                    .copied()
                    .unwrap_or(DB_FLOOR);
                if new_level >= old_level + tuning.challenger_db {
                    current = Some(candidate);
                }
            }
            if current != previous {
                held_until = frame + (tuning.winner_hold_ms / 10.0).ceil() as usize;
            }
        } else if frame >= held_until && vocal.is_none() {
            current = None;
        }
        result[frame] = current;
    }
    result
}

fn compile_stem(
    index: usize,
    stem: &MovementStemInput,
    analysis: &Analysis,
    winners: &[Option<usize>],
    layout: &PannerLayout,
    channels: &[&str],
    duration_frames: usize,
    duration_us: i64,
    sample_rate: u32,
    tuning: &MovementTuning,
) -> Result<MovementStemSchedule, String> {
    let mut route_cache = ObjectRouteCache::new();
    let home_positions = object_positions(&stem.placement);
    let home = if stem.object_mode.is_some() {
        object_route_at(layout, stem, home_positions[0], &mut route_cache)
    } else if stem.home_gains.is_empty() {
        layout.placement_route(&stem.placement)
    } else {
        stem.home_gains.clone()
    };
    let home_right = if stem.object_mode.is_some() {
        object_route_at(layout, stem, home_positions[1], &mut route_cache)
    } else if stem.home_right_gains.is_empty() {
        Vec::new()
    } else {
        stem.home_right_gains.clone()
    };
    let start = quantize_time(stem.settings.start_s * 1_000_000.0).clamp(0, duration_us);
    let end = stem
        .settings
        .end_s
        .map(|value| quantize_time(value * 1_000_000.0))
        .unwrap_or(duration_us)
        .clamp(start, duration_us);
    let ordinary_stereo = is_ordinary_stereo(channels);
    let movable = is_movable(&stem.stem_name, &stem.stem_key);
    let mut events = Vec::new();
    let mut last: Option<MovementEvent> = None;
    let mut last_target = None;
    let mut last_hit: Option<(i64, i8)> = None;
    let mut focus_level = 0.0;
    let mut lift_level = 0.0;
    let mut previous_time = 0_i64;
    let supporting_p95 = stem.features.p95_level_db() + stem.gain_db;
    // A finite interval owns its return. Reserve the release for the kind of
    // movement this stem can produce before the stop, compressing attack and
    // release together only when the interval is shorter than requested.
    let finite_interval = stem.settings.end_s.is_some();
    let requested_attack_us = (tuning.focus_attack_ms / stem.settings.response).max(1.0) * 1_000.0;
    let requested_focus_release_us =
        (tuning.focus_release_ms / stem.settings.response).max(1.0) * 1_000.0;
    let is_featured = stem.settings.role == MovementRole::Featured;
    let is_lift = is_supporting_lift(&stem.stem_name, &stem.stem_key);
    let is_percussive = is_percussion(&stem.stem_name, &stem.stem_key);
    let role_return_ms = if !is_featured && is_lift {
        if is_crowd(&stem.stem_name, &stem.stem_key) {
            4_000.0
        } else {
            2_000.0
        }
    } else if !is_featured && is_percussive {
        return_time_ms(&stem.stem_name, &stem.stem_key, tuning)
    } else {
        tuning.focus_release_ms
    };
    let requested_percussion_attack_us =
        (tuning.percussion_attack_ms / stem.settings.response).max(1.0) * 1_000.0;
    let requested_role_return_us = (role_return_ms / stem.settings.response).max(1.0) * 1_000.0;
    // Percussion's outward attack is part of the time it needs to get home.
    // Other roles describe their complete return envelope directly.
    let requested_return_window_us = if !is_featured && is_percussive {
        requested_percussion_attack_us + requested_role_return_us
    } else {
        requested_role_return_us
    };
    let interval_us = end.saturating_sub(start) as f64;
    let (attack_us, focus_release_us, role_return_us) = if finite_interval {
        let reserved_return = requested_return_window_us.min((interval_us * 0.5).max(0.0));
        (
            requested_attack_us.min((interval_us - reserved_return).max(0.0)),
            requested_focus_release_us.min((interval_us * 0.5).max(0.0)),
            reserved_return,
        )
    } else {
        (
            requested_attack_us,
            requested_focus_release_us,
            requested_return_window_us,
        )
    };
    let percussion_attack_us = if finite_interval {
        requested_percussion_attack_us.min((role_return_us * 0.5).max(0.0))
    } else {
        requested_percussion_attack_us
    };
    let percussion_recovery_us = if finite_interval {
        (role_return_us - percussion_attack_us).max(1.0)
    } else {
        requested_role_return_us
    };
    let (requested_lift_attack_us, requested_lift_release_us) =
        if is_crowd(&stem.stem_name, &stem.stem_key) {
            (
                (2_000.0 / stem.settings.response).max(1.0) * 1_000.0,
                (4_000.0 / stem.settings.response).max(1.0) * 1_000.0,
            )
        } else {
            (
                (1_000.0 / stem.settings.response).max(1.0) * 1_000.0,
                (2_000.0 / stem.settings.response).max(1.0) * 1_000.0,
            )
        };
    let lift_attack_us = if finite_interval {
        requested_lift_attack_us.min((interval_us - role_return_us).max(0.0))
    } else {
        requested_lift_attack_us
    };
    let lift_release_us = if finite_interval {
        role_return_us
    } else {
        requested_lift_release_us
    };
    let return_start = stem
        .settings
        .end_s
        .map(|_| end.saturating_sub(role_return_us.round() as i64));
    for time_us in (0..=duration_us).step_by(EVENT_GRID_US as usize) {
        let in_interval = time_us >= start && time_us < end;
        let frame = (time_us.max(0) / FEATURE_WINDOW_US) as usize;
        let active = analysis.active.get(frame).copied().unwrap_or(false);
        let stopping = return_start.is_some_and(|deadline| time_us >= deadline);
        let movement_on = !stopping
            && !ordinary_stereo
            && stem.enabled
            && stem.included
            && stem.settings.enabled
            && stem.settings.depth > 0.0
            && in_interval;
        // Once an interval enters its return window, existing percussion and
        // supporting lift state still decays. New hits and active lift targets
        // are gated by `movement_on` below, so the stop cannot introduce a new
        // excursion while the old one is returning home.
        let motion_window = !ordinary_stereo
            && stem.enabled
            && stem.included
            && stem.settings.enabled
            && stem.settings.depth > 0.0
            && in_interval;
        let winner = winners.get(frame).copied().flatten() == Some(index);
        let manual_featured = stem.settings.role == MovementRole::Featured && active;
        let focus_target = movement_on && movable && (manual_featured || winner);
        let dt_ms = (time_us - previous_time).max(0) as f64 / 1_000.0;
        let focus_duration = if focus_target {
            (attack_us / 1_000.0).max(1.0)
        } else {
            (focus_release_us / 1_000.0).max(1.0)
        };
        focus_level = move_toward(focus_level, f64::from(focus_target), dt_ms / focus_duration);
        if return_start.is_some_and(|deadline| time_us >= deadline)
            && stem.settings.end_s.is_some()
            && time_us >= end.saturating_sub(INTERPOLATION_US)
        {
            // A finite interval owns its complete return. The final changed
            // event has a full interpolation window before the stop, and no
            // movement state is allowed to leak past that boundary.
            focus_level = 0.0;
            lift_level = 0.0;
            last_hit = None;
        }

        let mut placement = stem.placement;
        let mut target_placement = stem.placement;
        let mut movement_fraction = 0.0;
        let mut moved = false;
        if focus_level > 0.0 && movable {
            let amount = stem.settings.depth * focus_level;
            placement.azimuth_deg *= 1.0 - amount;
            placement.elevation_deg *= 1.0 - amount;
            target_placement.azimuth_deg = 0.0;
            target_placement.elevation_deg = 0.0;
            if let (Some(left_right), Some(back_front)) =
                (stem.placement.left_right, stem.placement.back_front)
            {
                placement.left_right = Some(left_right * (1.0 - amount));
                placement.back_front = Some(back_front + (1.0 - back_front) * amount);
                target_placement.left_right = Some(0.0);
                target_placement.back_front = Some(1.0);
            }
            movement_fraction = amount;
            moved = amount > 1e-12;
        } else if motion_window && !matches!(stem.settings.role, MovementRole::Featured) {
            if movement_on {
                if let Some((hit_frame, side)) = (frame.saturating_sub(1)..=frame)
                    .rev()
                    .filter_map(|candidate| {
                        analysis
                            .hits
                            .get(candidate)
                            .copied()
                            .flatten()
                            .map(|side| (candidate, side))
                    })
                    .next()
                {
                    last_hit = Some((hit_frame as i64 * FEATURE_WINDOW_US, side));
                }
            }
            if let Some((hit_time, side)) = last_hit {
                let age_ms = (time_us - hit_time).max(0) as f64 / 1_000.0;
                let attack = (percussion_attack_us / 1_000.0).max(1.0);
                let recovery = (percussion_recovery_us / 1_000.0).max(attack + 1.0);
                let hit_strength = if age_ms <= attack {
                    age_ms / attack
                } else {
                    (1.0 - (age_ms - attack) / recovery).max(0.0)
                };
                if hit_strength > 0.0 && is_percussion(&stem.stem_name, &stem.stem_key) {
                    let amount = (stem.settings.depth * hit_strength).clamp(0.0, 1.0);
                    let excursion = side as f64 * excursion_deg(&stem.stem_name, &stem.stem_key);
                    placement.azimuth_deg += excursion * amount;
                    target_placement.azimuth_deg += excursion;
                    rotate_panner_anchor(&mut placement, excursion * amount);
                    rotate_panner_anchor(&mut target_placement, excursion);
                    movement_fraction = amount;
                    moved = true;
                }
            }

            let lift_target =
                if movement_on && is_supporting_lift(&stem.stem_name, &stem.stem_key) && active {
                    let level = analysis.envelope.get(frame).copied().unwrap_or(DB_FLOOR);
                    ((level - (supporting_p95 - tuning.supporting_span_db))
                        / tuning.supporting_span_db)
                        .clamp(0.0, 1.0)
                } else {
                    0.0
                };
            let attack = lift_attack_us / 1_000.0;
            let release = lift_release_us / 1_000.0;
            let duration = if lift_target > lift_level {
                attack
            } else {
                release
            };
            lift_level = move_toward(lift_level, lift_target, dt_ms / duration.max(0.001));
            if lift_level > 0.0 && is_supporting_lift(&stem.stem_name, &stem.stem_key) {
                let available = available_elevation(layout, channels, &stem.placement);
                let lift_scale = if is_crowd(&stem.stem_name, &stem.stem_key) {
                    0.5
                } else {
                    1.0
                };
                let amount = (stem.settings.depth * lift_level).clamp(0.0, 1.0);
                placement.elevation_deg += available * lift_scale * amount;
                target_placement.elevation_deg += available * lift_scale;
                movement_fraction = amount;
                moved = moved || available > 0.0;
            }
        } else {
            // An interval stop and an ordinary stereo bypass still let the
            // state return smoothly to Supporting for subsequent events.
            lift_level = move_toward(lift_level, 0.0, dt_ms / 2_000.0);
        }
        let can_emit = time_us == 0
            || (time_us >= start
                && time_us <= end
                && end.saturating_sub(time_us) >= INTERPOLATION_US);
        let target_inputs = (placement, target_placement, movement_fraction, moved);
        // Policy still advances every grid tick; held geometry needs no rerouting.
        if !can_emit || last_target.as_ref() == Some(&target_inputs) {
            previous_time = time_us;
            continue;
        }
        last_target = Some(target_inputs);
        let (position, gains, right_position, right_gains) = target(
            layout,
            stem,
            channels,
            &home,
            &home_right,
            placement,
            target_placement,
            movement_fraction,
            moved,
            &mut route_cache,
        );
        let event = MovementEvent {
            time_us,
            position,
            gains,
            right_position,
            right_gains,
            interpolation_us: INTERPOLATION_US,
        };
        if last
            .as_ref()
            .is_none_or(|previous| !same_event_target(previous, &event))
        {
            last = Some(event.clone());
            events.push(event);
        }
        previous_time = time_us;
    }
    if events.is_empty() {
        let (position, gains, right_position, right_gains) = target(
            layout,
            stem,
            channels,
            &home,
            &home_right,
            stem.placement,
            stem.placement,
            0.0,
            false,
            &mut route_cache,
        );
        events.push(MovementEvent {
            time_us: 0,
            position,
            gains,
            right_position,
            right_gains,
            interpolation_us: INTERPOLATION_US,
        });
    }
    let _ = duration_frames;
    let _ = sample_rate;
    Ok(MovementStemSchedule {
        stem_key: stem.stem_key.clone(),
        stem_index: index,
        events,
    })
}

fn move_toward(current: f64, target: f64, amount: f64) -> f64 {
    let amount = amount.max(0.0);
    if target >= current {
        (current + amount).min(target)
    } else {
        (current - amount).max(target)
    }
}

fn rotate_panner_anchor(placement: &mut StemPlacement, azimuth_delta_deg: f64) {
    let (Some(left_right), Some(back_front)) = (placement.left_right, placement.back_front) else {
        return;
    };
    let radius = left_right.hypot(back_front);
    let angle = left_right.atan2(back_front) - azimuth_delta_deg.to_radians();
    placement.left_right = Some(radius * angle.sin());
    placement.back_front = Some(radius * angle.cos());
}

fn target(
    layout: &PannerLayout,
    stem: &MovementStemInput,
    channels: &[&str],
    home: &[f64],
    home_right: &[f64],
    placement: StemPlacement,
    target_placement: StemPlacement,
    movement_fraction: f64,
    moved: bool,
    route_cache: &mut ObjectRouteCache,
) -> ([f64; 3], Vec<f64>, Option<[f64; 3]>, Option<Vec<f64>>) {
    if stem.object_mode.is_none() {
        let mut gains = if movement_fraction > 1e-12 {
            let mut target = layout.placement_route(&target_placement);
            retain_bed_main_norm(&mut target, home, channels);
            home.iter()
                .zip(target)
                .map(|(home, target)| home + (target - home) * movement_fraction.clamp(0.0, 1.0))
                .collect()
        } else {
            home.to_vec()
        };
        if movement_fraction > 1e-12 {
            retain_bed_main_norm(&mut gains, home, channels);
        }
        if let Some(lfe) = channels.iter().position(|name| *name == "LFE") {
            if let Some(value) = home.get(lfe) {
                gains[lfe] = *value;
            }
        }
        return (
            placement_position(placement.azimuth_deg, placement.elevation_deg),
            gains,
            None,
            None,
        );
    }

    let linked = matches!(stem.object_mode, Some(MovementObjectMode::LinkedStereo));
    let positions = object_positions(&StemPlacement {
        width_deg: if linked { placement.width_deg } else { 0.0 },
        ..placement
    });
    let mut gains = if moved {
        object_route_at(layout, stem, positions[0], route_cache)
    } else {
        home.to_vec()
    };
    if let Some(lfe) = channels.iter().position(|name| *name == "LFE") {
        gains[lfe] = 0.0;
    }
    if !linked {
        return (positions[0], gains, None, None);
    }
    let mut right = if moved {
        object_route_at(layout, stem, positions[1], route_cache)
    } else {
        home_right.to_vec()
    };
    if let Some(lfe) = channels.iter().position(|name| *name == "LFE") {
        right[lfe] = 0.0;
    }
    (positions[0], gains, Some(positions[1]), Some(right))
}

fn object_route_at(
    layout: &PannerLayout,
    stem: &MovementStemInput,
    position: [f64; 3],
    cache: &mut ObjectRouteCache,
) -> Vec<f64> {
    // Layout and object metadata are fixed for this stem compilation. Reuse
    // exact endpoints across repeated hits without rounding movement positions.
    cache
        .entry(position.map(f64::to_bits))
        .or_insert_with(|| {
            layout.cartesian_object_route(
                position,
                stem.placement.object_size,
                stem.channel_lock,
                &stem
                    .zone_exclusion
                    .iter()
                    .map(String::as_str)
                    .collect::<Vec<_>>(),
            )
        })
        .clone()
}

fn same_event_target(left: &MovementEvent, right: &MovementEvent) -> bool {
    left.position == right.position
        && left.gains == right.gains
        && left.right_position == right.right_position
        && left.right_gains == right.right_gains
}

fn placement_position(azimuth_deg: f64, elevation_deg: f64) -> [f64; 3] {
    let [x, y, z] = direction(azimuth_deg, elevation_deg);
    [x, -z, y]
}

fn retain_bed_main_norm(gains: &mut [f64], home: &[f64], channels: &[&str]) {
    let norm = |values: &[f64]| {
        values
            .iter()
            .enumerate()
            .filter(|(index, _)| channels.get(*index).is_none_or(|name| *name != "LFE"))
            .map(|(_, value)| value * value)
            .sum::<f64>()
            .sqrt()
    };
    let home_norm = norm(home);
    let target_norm = norm(gains);
    if home_norm > 0.0 && target_norm > 0.0 {
        let scale = home_norm / target_norm;
        for (index, gain) in gains.iter_mut().enumerate() {
            if channels.get(index).is_none_or(|name| *name != "LFE") {
                *gain *= scale;
            }
        }
    }
}

fn available_elevation(
    layout: &PannerLayout,
    _channels: &[&str],
    placement: &StemPlacement,
) -> f64 {
    let max_elevation = layout.max_elevation_deg();
    if max_elevation <= 0.0 {
        0.0
    } else {
        (max_elevation - placement.elevation_deg).max(0.0)
    }
}

fn quantize_time(time_us: f64) -> i64 {
    ((time_us / EVENT_GRID_US as f64).round() as i64) * EVENT_GRID_US
}

fn return_time_ms(name: &str, key: &str, tuning: &MovementTuning) -> f64 {
    if is_toms(name, key) {
        tuning.toms_return_ms
    } else if is_crash(name, key) {
        tuning.crash_return_ms
    } else {
        tuning.percussion_return_ms
    }
}

fn excursion_deg(name: &str, key: &str) -> f64 {
    if is_toms(name, key) {
        25.0
    } else if is_crash(name, key) {
        10.0
    } else {
        8.0
    }
}

fn stem_text(name: &str, key: &str) -> String {
    let text = if name.trim().is_empty() {
        key.split('@').next().unwrap_or(key)
    } else {
        name
    };
    text.trim().to_ascii_lowercase()
}

fn is_vocal(name: &str, key: &str) -> bool {
    matches!(
        stem_text(name, key).as_str(),
        "vocals" | "lead vocals" | "lead vocal"
    )
}

fn is_melodic(name: &str, key: &str) -> bool {
    matches!(stem_text(name, key).as_str(), "guitar" | "piano")
}

fn is_toms(name: &str, key: &str) -> bool {
    matches!(stem_text(name, key).as_str(), "tom" | "toms")
}

fn is_crash(name: &str, key: &str) -> bool {
    stem_text(name, key) == "crash"
}

fn is_crowd(name: &str, key: &str) -> bool {
    stem_text(name, key) == "crowd"
}

fn is_supporting_lift(name: &str, key: &str) -> bool {
    matches!(
        stem_text(name, key).as_str(),
        "backing vocal" | "backing vocals" | "crowd"
    )
}

fn is_percussion(name: &str, key: &str) -> bool {
    matches!(
        stem_text(name, key).as_str(),
        "tom" | "toms" | "hi-hat" | "hihat" | "ride" | "crash"
    )
}

fn is_anchor(name: &str, key: &str) -> bool {
    matches!(
        stem_text(name, key).as_str(),
        "vocals"
            | "lead vocals"
            | "lead vocal"
            | "bass"
            | "kick"
            | "snare"
            | "drums"
            | "combined drums"
    )
}

fn is_movable(name: &str, key: &str) -> bool {
    !is_anchor(name, key)
}

fn is_ordinary_stereo(channels: &[&str]) -> bool {
    channels.len() == 2 && channels[0] == "FL" && channels[1] == "FR"
}

fn percentile(values: &[f64], fraction: f64) -> f64 {
    if values.is_empty() {
        return DB_FLOOR;
    }
    let mut scratch = values.to_vec();
    let index = ((scratch.len() - 1) as f64 * fraction.clamp(0.0, 1.0)).round() as usize;
    *scratch.select_nth_unstable_by(index, f64::total_cmp).1
}

fn rolling_median(values: &[f64], width: usize) -> Vec<f64> {
    (0..values.len())
        .map(|index| {
            percentile(
                &values
                    [index.saturating_sub(width / 2)..=(index + width / 2).min(values.len() - 1)],
                0.5,
            )
        })
        .collect()
}

fn envelope(values: &[f64], attack_ms: f64, release_ms: f64) -> Vec<f64> {
    let mut result = Vec::with_capacity(values.len());
    let mut state = DB_FLOOR;
    for value in values {
        let coefficient = if *value > state {
            1.0 - (-10.0 / attack_ms.max(1.0)).exp()
        } else {
            1.0 - (-10.0 / release_ms.max(1.0)).exp()
        };
        state += (*value - state) * coefficient;
        result.push(state);
    }
    result
}

#[cfg(test)]
mod tests {
    use super::*;

    fn tuning() -> MovementTuning {
        MovementTuning {
            activity_floor_db: -65.0,
            activity_enter_db: 6.0,
            activity_leave_db: 3.0,
            activity_dwell_ms: 80.0,
            activity_exit_dwell_ms: 250.0,
            focus_prominence_db: 6.0,
            focus_share: 0.85,
            focus_qualification_ms: 750.0,
            focus_attack_ms: 750.0,
            focus_release_ms: 1_500.0,
            vocal_hold_ms: 1_200.0,
            winner_hold_ms: 1_000.0,
            challenger_db: 3.0,
            supporting_span_db: 24.0,
            percussion_rise_db: 6.0,
            percussion_gate_db: 6.0,
            percussion_retrigger_ms: 120.0,
            percussion_attack_ms: 40.0,
            percussion_return_ms: 300.0,
            toms_return_ms: 500.0,
            crash_return_ms: 1_000.0,
        }
    }

    #[test]
    fn percentile_selection_matches_full_sort() {
        for len in [0, 1, 2, 5, 301, 1000] {
            let values: Vec<_> = (0..len).map(|i| ((i * 37) % 101) as f64 - 50.0).collect();
            let mut sorted = values.clone();
            sorted.sort_by(f64::total_cmp);
            for fraction in [-0.2_f64, 0.0, 0.5, 0.95, 1.0, 1.2] {
                let expected = if sorted.is_empty() {
                    DB_FLOOR
                } else {
                    sorted[((len - 1) as f64 * fraction.clamp(0.0, 1.0)).round() as usize]
                };
                assert_eq!(percentile(&values, fraction), expected);
            }
        }
    }

    #[test]
    fn extracts_ragged_stereo_without_cancellation() {
        let left = vec![1.0f32; 481];
        let right = vec![-1.0f32; 481];
        let features = extract_features(&left, Some(&right), 48_000).unwrap();
        assert_eq!(features.window_frames, 480);
        assert_eq!(features.energies.len(), 2);
        assert!((features.energies[0] - 1.0).abs() < 1e-12);
        assert!((features.energies[1] - 1.0).abs() < 1e-12);
    }

    #[test]
    fn schedule_is_grid_aligned_and_interpolates_shared_gains() {
        let features = extract_features(&vec![0.5f32; 48_000], None, 48_000).unwrap();
        let request = MovementCompileRequest {
            sample_rate: 48_000,
            duration_frames: 48_000,
            revision: 7,
            channels: vec!["FL".into(), "FR".into(), "C".into()],
            stems: vec![MovementStemInput {
                stem_key: "Guitar".into(),
                stem_name: "Guitar".into(),
                features,
                gain_db: 0.0,
                enabled: true,
                included: true,
                placement: StemPlacement::new(60.0, 0.0, 0.0, 0.0, 0.0),
                home_gains: vec![],
                home_right_gains: vec![],
                settings: MovementSettings {
                    enabled: true,
                    depth: 0.5,
                    ..MovementSettings::default()
                },
                object_mode: None,
                channel_lock: false,
                zone_exclusion: vec![],
            }],
            tuning: tuning(),
        };
        let layout = PannerLayout::new(&["FL", "FR", "C"]);
        let mut cache = ObjectRouteCache::new();
        let stem = &request.stems[0];
        for azimuth in [60.0, -60.0, 60.0] {
            let position = placement_position(azimuth, 0.0);
            let cached = object_route_at(&layout, stem, position, &mut cache);
            let direct =
                layout.cartesian_object_route(position, stem.placement.object_size, false, &[]);
            assert_eq!(cached, direct);
        }
        assert_eq!(cache.len(), 2);
        let schedule = compile_movement(&request).unwrap();
        assert_eq!(schedule.revision, 7);
        assert!(schedule.stems[0]
            .events
            .iter()
            .all(|event| event.time_us % EVENT_GRID_US == 0));
        let mut at = vec![0.0; 3];
        schedule.stems[0].sample_into(10_000, false, &mut at);
        assert!(at.iter().all(|value| value.is_finite()));
    }
}

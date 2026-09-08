//! Streaming stem-to-speaker-bed routing.
//!
//! Mirrors `separation/stem_router.py::StemRouter.route` block by block: each
//! shaped send carries the filter state and decorrelator history its offline
//! counterpart would have accumulated, so the two agree sample for sample.
use super::params::{SendParams, SendShape};
use super::state::OnePole;
use crate::kernels::biquad::SosFilter;
use crate::kernels::butter::{butter_sos, BandType};
use crate::routing::ambient::AmbientSplit;
use crate::routing::ambient_expander::FixedAmbientExpander;
use crate::routing::decorrelate::{
    velvet_pair_seeded, VelvetFir, VelvetLine, VELVET_SEED, VELVET_SEED_HEIGHT,
};
use crate::routing::sends::{directional_band_sos, height_texture_sos};
use crate::stem_dynamic_eq::{StemDynamicEq, StemDynamicEqParams};
use crate::stem_dynamics::{StemDynamics, StemDynamicsParams};
use crate::stem_eq::{StemEq, StemEqParams};

const AMBIENT_GAIN_RAMP_MS: f64 = 8.0;
// The preview accepts blocks larger than the Web Audio quantum (the focused
// stream suite uses 4096), so keep the ambient output scratch at that size
// before the first callback. `process_into` reuses it without growing it.
const AMBIENT_BUFFER_CAPACITY: usize = 4096;

mod routing_ambient;
pub use routing_ambient::LfeBus;

/// One shaped send: a filter chain and, for a send fed from the dry stem,
/// one side of a decorrelator pair. The ambient sends carry no decorrelator:
/// their two sides are already independent signals — that is what the split
/// selected them for, so a velvet pair would only smear them.
struct Send {
    filters: Vec<SosFilter>,
    velvet: Option<VelvetLine>,
    /// Elevation-EQ gains, when this send is a height send: low rolloff,
    /// high shelf, directional band.
    elevation: Option<(f64, f64, f64)>,
}

impl Send {
    fn reset(&mut self) {
        for f in &mut self.filters {
            f.reset();
        }
        if let Some(velvet) = &mut self.velvet {
            velvet.reset();
        }
    }

    /// Re-derive a surround send's highpass in place.
    fn retune_surround(&mut self, sample_rate: u32, p: &SendParams) {
        let nyq = sample_rate as f64 / 2.0;
        let hp = butter_sos(2, p.surround_bass_cutoff_hz / nyq, BandType::High);
        self.filters[0].retune_flat(&hp);
    }

    /// Re-derive a height send's low-rolloff/crossover/band in place.
    fn retune_height(&mut self, sample_rate: u32, p: &SendParams) {
        let nyq = sample_rate as f64 / 2.0;
        let lp = butter_sos(1, p.height_low_rolloff_hz / nyq, BandType::Low);
        let hp = butter_sos(2, p.height_crossover_hz / nyq, BandType::High);
        let band = directional_band_sos(
            p.height_directional_band_hz,
            sample_rate,
            p.height_directional_band_gain,
        );
        self.filters[0].retune_flat(&lp);
        self.filters[1].retune_flat(&hp);
        self.filters[2].retune_flat(&[band]);
        self.elevation = Some((
            p.height_low_rolloff_gain,
            p.height_high_shelf_gain,
            p.height_directional_band_gain,
        ));
    }

    fn process_in_place(&mut self, buffer: &mut Vec<f64>) {
        match self.elevation {
            None => {
                for x in buffer.iter_mut() {
                    *x = self.filters[0].tick(*x);
                }
            }
            Some((low_gain, high_gain, band_gain)) => {
                for x in buffer.iter_mut() {
                    let low = self.filters[0].tick(*x);
                    let bass_shaped = *x - low * (1.0 - low_gain);
                    let high = self.filters[1].tick(bass_shaped);
                    let shelved = bass_shaped + high * (high_gain - 1.0);
                    *x = if band_gain == 1.0 {
                        shelved
                    } else {
                        self.filters[2].tick(shelved)
                    };
                }
            }
        }
        if let Some(velvet) = &mut self.velvet {
            velvet.process(buffer);
        }
    }
}

/// EQ'd stem samples covering `[base, base + left.len())`, grown forward as
/// blocks are asked for and trimmed from behind once they are consumed.
#[derive(Default)]
struct Ahead {
    left: Vec<f64>,
    right: Vec<f64>,
    base: usize,
}

impl Ahead {
    fn clear(&mut self, at: usize) {
        self.left.clear();
        self.right.clear();
        self.base = at;
    }

    fn end(&self) -> usize {
        self.base + self.left.len()
    }

    /// Fill forward to `target`, taking raw stem PCM through the stem EQ.
    fn fill(
        &mut self,
        stem_left: &[f32],
        stem_right: &[f32],
        eq: &mut Option<StemEq>,
        dynamic_eq: &mut Option<StemDynamicEq>,
        dynamics: &mut Option<StemDynamics>,
        target: usize,
    ) {
        if target <= self.end() {
            return;
        }
        let (from, count) = (self.end(), target - self.end());
        let take = |source: &[f32]| -> Vec<f64> {
            (from..from + count)
                .map(|i| *source.get(i).unwrap_or(&0.0) as f64)
                .collect()
        };
        let (mut left, mut right) = (take(stem_left), take(stem_right));
        if let Some(eq) = eq {
            eq.process_stereo(&mut left, &mut right);
        }
        if let Some(dynamic_eq) = dynamic_eq {
            dynamic_eq.process_stereo(&mut left, &mut right);
        }
        if let Some(dynamics) = dynamics {
            let mut channels = [left, right];
            dynamics.process(&mut channels);
            left = std::mem::take(&mut channels[0]);
            right = std::mem::take(&mut channels[1]);
        }
        self.left.extend_from_slice(&left);
        self.right.extend_from_slice(&right);
    }

    /// Drop samples before `keep_from`, in whole chunks so the copy is rare.
    fn trim(&mut self, keep_from: usize) {
        let drop = keep_from.saturating_sub(self.base);
        if drop < self.left.len() / 2 {
            return;
        }
        self.left.drain(..drop);
        self.right.drain(..drop);
        self.base += drop;
    }
}

/// First of the two ambient-surround signals; the right side follows it.
pub const AMBIENT_SURROUND: usize = 7;
/// First of the two ambient-height signals; the right side follows it.
pub const AMBIENT_HEIGHT: usize = 9;
/// The dry pair as it stood before the ambient half was taken out of it —
/// the stem's own level, which is what the route normalization matches the
/// routed sum to. Reading the post-split pair instead would make a stem get
/// quieter as its sends come up, since the sends are inside the routed sum.
pub const STEM_INPUT: usize = 11;
/// Direct-residual height texture, kept separate from expanded ambient sends.
pub const AMBIENT_TEXTURE: usize = 13;
/// First per-destination fixed-expansion signal slot.
pub const AMBIENT_EXPANDED: usize = 15;
pub const AMBIENT_EXPANDED_COUNT: usize = 8;
/// Signals a stem's routing can draw on: [`shape_index`]'s seven, the four
/// ambient sends, the input pair, direct height texture, and expanded sends.
pub const SIGNALS: usize = AMBIENT_EXPANDED + AMBIENT_EXPANDED_COUNT;

const AMBIENT_DESTINATIONS: [&str; AMBIENT_EXPANDED_COUNT] =
    ["SL", "SR", "BL", "BR", "TFL", "TFR", "TBL", "TBR"];

pub fn ambient_expanded_slot(destination: &str) -> Option<usize> {
    AMBIENT_DESTINATIONS
        .iter()
        .position(|candidate| *candidate == destination)
        .map(|index| AMBIENT_EXPANDED + index)
}

/// Per-stem shaping state: the four sends plus the optional stem EQ, and —
/// when the stem asks for it — the primary/ambient split and the four extra
/// sends its ambient half plays through.
pub struct StemRouteState {
    sample_rate_hz: f64,
    pub eq: Option<StemEq>,
    dynamic_eq: Option<StemDynamicEq>,
    dynamics: Option<StemDynamics>,
    surround: [Send; 2],
    height: [Send; 2],
    split: Option<AmbientSplit>,
    /// Stem samples past the stem EQ, filled ahead of the block being
    /// rendered. The split's mask needs a window that ends after the block
    /// does, and it has to be the same signal the export splits — which is
    /// the EQ'd stem, since the offline path EQs before it routes.
    ahead: Ahead,
    ambient_surround: [Send; 2],
    ambient_height: [Send; 2],
    ambient_texture: [Send; 2],
    ambient_expander: FixedAmbientExpander,
    ambient_expanded: Vec<Vec<f64>>,
    ambient_output_slots: [Option<usize>; AMBIENT_EXPANDED_COUNT],
    ambient_tail_remaining: usize,
    ambient_texture_cutoff: [SosFilter; 2],
    ambient_rear_gain: [OnePole; 2],
    ambient_height_gain: [OnePole; 2],
    ambient_trim_gain: OnePole,
    height_texture_gain: [OnePole; 2],
    height_texture_cutoff: OnePole,
    ambient_trim_target: f64,
    height_texture_cutoff_target: f64,
    ambient_gains_initialized: bool,
    height_texture_initialized: bool,
    /// Distinguishes a fresh route from a live texture enable. A fresh
    /// configured route starts at its target; a live edit must ramp from zero.
    texture_started: bool,
    texture_active: bool,
    /// The dry pair of the block being shaped, after the ambient half has
    /// been taken out of it.
    scratch: [Vec<f64>; 2],
    texture_scratch: [Vec<f64>; 2],
    /// Last block's signals, indexed by [`shape_index`]: the dry pair, their
    /// mono sum, the four shaped sends, then the four ambient sends.
    shaped: [Vec<f64>; SIGNALS],
}

impl StemRouteState {
    pub fn new(
        sample_rate: u32,
        p: &SendParams,
        eq: Option<StemEqParams>,
        dynamic_eq: Option<StemDynamicEqParams>,
        dynamics: Option<StemDynamicsParams>,
    ) -> Self {
        Self::new_for_layout(
            sample_rate,
            p,
            eq,
            dynamic_eq,
            dynamics,
            &AMBIENT_DESTINATIONS,
        )
    }

    pub fn new_for_layout(
        sample_rate: u32,
        p: &SendParams,
        eq: Option<StemEqParams>,
        dynamic_eq: Option<StemDynamicEqParams>,
        dynamics: Option<StemDynamicsParams>,
        destinations: &[&str],
    ) -> Self {
        let nyq = sample_rate as f64 / 2.0;
        let surround_hp = butter_sos(2, p.surround_bass_cutoff_hz / nyq, BandType::High);
        let height_lp = butter_sos(1, p.height_low_rolloff_hz / nyq, BandType::Low);
        let height_hp = butter_sos(2, p.height_crossover_hz / nyq, BandType::High);
        let height_band = directional_band_sos(
            p.height_directional_band_hz,
            sample_rate,
            p.height_directional_band_gain,
        );
        let texture_cutoff = height_texture_sos(
            sample_rate,
            crate::routing::ambient::AMBIENT_HEIGHT_CROSSOVER_HZ,
        );

        let (surround_l, surround_r) = velvet_pair_seeded(sample_rate, VELVET_SEED);
        let (height_l, height_r) = velvet_pair_seeded(sample_rate, VELVET_SEED_HEIGHT);

        let surround_send = |fir: Option<&VelvetFir>| Send {
            filters: vec![SosFilter::from_flat(&surround_hp)],
            velvet: fir.map(VelvetLine::new),
            elevation: None,
        };
        let height_send = |fir: Option<&VelvetFir>| Send {
            filters: vec![
                SosFilter::from_flat(&height_lp),
                SosFilter::from_flat(&height_hp),
                SosFilter::from_flat(&[height_band]),
            ],
            velvet: fir.map(VelvetLine::new),
            elevation: Some((
                p.height_low_rolloff_gain,
                p.height_high_shelf_gain,
                p.height_directional_band_gain,
            )),
        };

        let ambient_expander = FixedAmbientExpander::new(sample_rate, destinations);
        let ambient_expanded = (0..ambient_expander.destinations().len())
            .map(|_| Vec::with_capacity(AMBIENT_BUFFER_CAPACITY))
            .collect();
        let mut ambient_output_slots = [None; AMBIENT_EXPANDED_COUNT];
        for (output, destination) in ambient_expander.destinations().enumerate() {
            if let Some(slot) = ambient_expanded_slot(destination) {
                ambient_output_slots[slot - AMBIENT_EXPANDED] = Some(output);
            }
        }
        Self {
            sample_rate_hz: sample_rate as f64,
            eq: eq.map(|params| StemEq::new(sample_rate, params)),
            dynamic_eq: dynamic_eq.map(|params| StemDynamicEq::new(sample_rate, params)),
            dynamics: dynamics.map(|params| StemDynamics::new(sample_rate, params)),
            surround: [
                surround_send(Some(&surround_l)),
                surround_send(Some(&surround_r)),
            ],
            height: [height_send(Some(&height_l)), height_send(Some(&height_r))],
            split: None,
            ahead: Ahead::default(),
            ambient_surround: [surround_send(None), surround_send(None)],
            ambient_height: [height_send(None), height_send(None)],
            ambient_texture: [height_send(None), height_send(None)],
            ambient_expander,
            ambient_expanded,
            ambient_output_slots,
            ambient_tail_remaining: 0,
            ambient_texture_cutoff: [
                SosFilter::from_flat(&texture_cutoff),
                SosFilter::from_flat(&texture_cutoff),
            ],
            ambient_rear_gain: std::array::from_fn(|_| {
                OnePole::new(AMBIENT_GAIN_RAMP_MS, sample_rate as f64)
            }),
            ambient_height_gain: std::array::from_fn(|_| {
                OnePole::new(AMBIENT_GAIN_RAMP_MS, sample_rate as f64)
            }),
            ambient_trim_gain: OnePole::new_at(AMBIENT_GAIN_RAMP_MS, sample_rate as f64, 1.0),
            height_texture_gain: std::array::from_fn(|_| {
                OnePole::new(AMBIENT_GAIN_RAMP_MS, sample_rate as f64)
            }),
            height_texture_cutoff: OnePole::new_at(
                AMBIENT_GAIN_RAMP_MS,
                sample_rate as f64,
                crate::routing::ambient::AMBIENT_HEIGHT_CROSSOVER_HZ,
            ),
            ambient_trim_target: 1.0,
            height_texture_cutoff_target: crate::routing::ambient::AMBIENT_HEIGHT_CROSSOVER_HZ,
            ambient_gains_initialized: false,
            height_texture_initialized: false,
            texture_started: false,
            texture_active: false,
            scratch: Default::default(),
            texture_scratch: Default::default(),
            shaped: Default::default(),
        }
    }

    pub fn has_ambient(&self) -> bool {
        self.split.is_some() || self.texture_active
    }

    /// Shape one block of a stem into every signal a speaker can draw on.
    ///
    /// The stem EQ runs ahead of the block rather than in step with it, so
    /// the ambient split can read the window its mask needs without the
    /// output being delayed by one. `rear`/`height` are the amounts moved out
    /// of the dry pair and into the ambient sends.
    #[allow(clippy::too_many_arguments)]
    pub fn process_block(
        &mut self,
        stem_left: &[f32],
        stem_right: &[f32],
        start: usize,
        count: usize,
        rear: [f64; 2],
        height: [f64; 2],
        texture: [f64; 2],
        surround: bool,
        height_send: bool,
    ) {
        let look_ahead = self.split.as_ref().map_or(0, |s| s.look_ahead());
        if start < self.ahead.base || start > self.ahead.end() {
            // A seek, or the first block: the EQ's history and the split's
            // frame grid both restart where the transport landed.
            self.ahead.clear(start);
            if let Some(eq) = &mut self.eq {
                eq.reset();
            }
            if let Some(split) = &mut self.split {
                split.reset();
            }
            // A direct transport jump can bypass the engine-level rewind. Drop
            // the fixed expander's FIR history too, otherwise the first block
            // after the jump contains the previous playhead's tail.
            self.ambient_expander.seek(start);
            for output in &mut self.ambient_expanded {
                output.clear();
            }
        }
        self.ahead.fill(
            stem_left,
            stem_right,
            &mut self.eq,
            &mut self.dynamic_eq,
            &mut self.dynamics,
            start + count + look_ahead,
        );

        let offset = start - self.ahead.base;
        self.scratch[0].clear();
        self.scratch[0].extend_from_slice(&self.ahead.left[offset..offset + count]);
        self.scratch[1].clear();
        self.scratch[1].extend_from_slice(&self.ahead.right[offset..offset + count]);

        for i in 0..2 {
            self.shaped[STEM_INPUT + i].clear();
            self.shaped[STEM_INPUT + i].extend_from_slice(&self.scratch[i]);
        }
        let ambient_target =
            rear.iter().any(|value| *value > 0.0) || height.iter().any(|value| *value > 0.0);
        let ambient_fading = self
            .ambient_rear_gain
            .iter()
            .any(|gain| !gain.is_settled(0.0))
            || self
                .ambient_height_gain
                .iter()
                .any(|gain| !gain.is_settled(0.0));
        let texture_target = texture.iter().any(|value| *value > 0.0);
        let texture_fading = self
            .height_texture_gain
            .iter()
            .any(|gain| !gain.is_settled(0.0));
        let ambient_source_active = ambient_target || ambient_fading;
        if ambient_source_active {
            self.ambient_tail_remaining = 0;
        } else if self.split.is_some() && self.ambient_tail_remaining == 0 {
            // The send ramp has reached zero, but the fixed FIR still has a
            // real tail. Keep feeding it zeros until that tail has drained
            // instead of truncating it at the end of the gain ramp.
            self.ambient_tail_remaining = self.ambient_expander.span();
        }
        if self.split.is_some() && ambient_source_active {
            self.split_ambient(start, count, rear, height);
            self.expand_ambient(count);
        } else if self.split.is_some() && self.ambient_tail_remaining > 0 {
            for slot in AMBIENT_SURROUND..AMBIENT_SURROUND + 4 {
                self.shaped[slot].clear();
                self.shaped[slot].resize(count, 0.0);
            }
            self.expand_ambient(count);
            self.ambient_tail_remaining = self.ambient_tail_remaining.saturating_sub(count);
        } else if !ambient_source_active {
            for slot in AMBIENT_SURROUND..AMBIENT_SURROUND + 4 {
                self.shaped[slot].clear();
                self.shaped[slot].resize(count, 0.0);
            }
            self.clear_expanded(count);
        }
        for slot in AMBIENT_TEXTURE..AMBIENT_TEXTURE + 2 {
            self.shaped[slot].clear();
            self.shaped[slot].resize(count, 0.0);
        }
        self.shape_sends(count, surround, height_send);
        if self.texture_active && (texture_target || texture_fading) {
            self.apply_height_texture(count, texture);
        }
        if self.split.is_some() && !ambient_source_active && self.ambient_tail_remaining == 0 {
            // The last zero block was processed above, so leave its drained
            // output visible until the next block and drop the split now.
            self.split = None;
            self.ambient_expander.reset();
        }
        if !ambient_source_active && !texture_target && !texture_fading {
            self.texture_active = false;
        }
        self.texture_started = true;
        self.ahead.trim(start + count);
    }

    pub fn reset(&mut self) {
        if let Some(eq) = &mut self.eq {
            eq.reset();
        }
        if let Some(dynamics) = &mut self.dynamics {
            dynamics.reset();
        }
        if let Some(dynamic_eq) = &mut self.dynamic_eq {
            dynamic_eq.reset();
        }
        for s in self
            .surround
            .iter_mut()
            .chain(self.height.iter_mut())
            .chain(self.ambient_surround.iter_mut())
            .chain(self.ambient_height.iter_mut())
            .chain(self.ambient_texture.iter_mut())
        {
            s.reset();
        }
        for filter in &mut self.ambient_texture_cutoff {
            filter.reset();
        }
        if let Some(split) = &mut self.split {
            split.reset();
        }
        self.ambient_expander.reset();
        for output in &mut self.ambient_expanded {
            output.clear();
        }
        self.ambient_tail_remaining = 0;
        for side in 0..2 {
            self.ambient_rear_gain[side].reset();
            self.ambient_height_gain[side].reset();
            self.height_texture_gain[side].reset();
        }
        self.ambient_trim_gain.set(1.0);
        self.height_texture_cutoff
            .set(crate::routing::ambient::AMBIENT_HEIGHT_CROSSOVER_HZ);
        self.ambient_gains_initialized = false;
        self.height_texture_initialized = false;
        // Keep the configured texture route across a seek. `rewind` resets
        // filter state but does not reapply the parameter block.
        self.texture_started = false;
        self.ahead.clear(0);
    }

    fn clear_expanded(&mut self, count: usize) {
        for output in &mut self.ambient_expanded {
            output.clear();
            output.resize(count, 0.0);
        }
    }

    fn expand_ambient(&mut self, count: usize) {
        let inputs = [
            &self.shaped[AMBIENT_SURROUND][..count],
            &self.shaped[AMBIENT_SURROUND + 1][..count],
            &self.shaped[AMBIENT_HEIGHT][..count],
            &self.shaped[AMBIENT_HEIGHT + 1][..count],
        ];
        self.ambient_expander
            .process_into(inputs, &mut self.ambient_expanded);
    }

    /// Adopt new send shaping and/or a new stem EQ in place, keeping every
    /// filter's carried state and the EQ convolvers' history — a live mix
    /// edit re-derives coefficients rather than restarting this stem's
    /// routing cold. `sends_changed`/`eq_changed` let the caller skip the
    /// (still cheap, but non-zero) work when that half didn't move.
    pub fn retune(
        &mut self,
        sample_rate: u32,
        sends: &SendParams,
        eq: Option<StemEqParams>,
        dynamic_eq: Option<StemDynamicEqParams>,
        dynamics: Option<StemDynamicsParams>,
        sends_changed: bool,
        eq_changed: bool,
        dynamic_eq_changed: bool,
        dynamics_changed: bool,
    ) {
        if sends_changed {
            for s in self.surround.iter_mut() {
                s.retune_surround(sample_rate, sends);
            }
            for s in self.height.iter_mut() {
                s.retune_height(sample_rate, sends);
            }
            for s in self.ambient_texture.iter_mut() {
                s.retune_height(sample_rate, sends);
            }
        }
        if eq_changed {
            match (self.eq.as_mut(), eq) {
                (Some(current), Some(next)) => current.retune(sample_rate, next),
                (None, Some(next)) => self.eq = Some(StemEq::new(sample_rate, next)),
                (_, None) => self.eq = None,
            }
        }
        if dynamics_changed {
            match (self.dynamics.as_mut(), dynamics) {
                (Some(current), Some(next)) => current.retune(sample_rate, next),
                (None, Some(next)) => self.dynamics = Some(StemDynamics::new(sample_rate, next)),
                (_, None) => self.dynamics = None,
            }
        }
        if dynamic_eq_changed {
            match (self.dynamic_eq.as_mut(), dynamic_eq) {
                (Some(current), Some(next)) => current.retune(sample_rate, next),
                (None, Some(next)) => self.dynamic_eq = Some(StemDynamicEq::new(sample_rate, next)),
                (_, None) => self.dynamic_eq = None,
            }
        }
    }

    /// Shape a whole block into the seven signals a speaker can draw on,
    /// readable through [`Self::signal`].
    ///
    /// A send no speaker draws from is skipped and reads back as the dry
    /// signal, which is what `StemRouter.route`'s `needs_surround` /
    /// `needs_height` guards produce offline. Its filters then start cold if
    /// a later mix edit routes the stem there — the same cold start the
    /// offline path takes on every render.
    pub fn process(&mut self, left: &[f64], right: &[f64], surround: bool, height: bool) {
        self.scratch[0].clear();
        self.scratch[0].extend_from_slice(left);
        self.scratch[1].clear();
        self.scratch[1].extend_from_slice(right);
        self.shape_sends(left.len().min(right.len()), surround, height);
    }

    /// The dry pair in `scratch` into the seven dry signals.
    fn shape_sends(&mut self, count: usize, surround: bool, height: bool) {
        for i in 0..2 {
            self.shaped[i].clear();
            self.shaped[i].extend_from_slice(&self.scratch[i][..count]);
        }
        self.shaped[2].clear();
        self.shaped[2].extend(
            self.scratch[0][..count]
                .iter()
                .zip(&self.scratch[1][..count])
                .map(|(l, r)| (l + r) * 0.5),
        );
        for i in 0..2 {
            if surround {
                let (send, out) = (&mut self.surround[i], &mut self.shaped[3 + i]);
                out.clear();
                out.extend_from_slice(&self.scratch[i][..count]);
                send.process_in_place(out);
            } else {
                self.shaped[3 + i].clear();
                self.shaped[3 + i].extend_from_slice(&self.scratch[i][..count]);
            }
            if height {
                let (send, out) = (&mut self.height[i], &mut self.shaped[5 + i]);
                out.clear();
                out.extend_from_slice(&self.scratch[i][..count]);
                send.process_in_place(out);
            } else {
                self.shaped[5 + i].clear();
                self.shaped[5 + i].extend_from_slice(&self.scratch[i][..count]);
            }
        }
    }

    /// One signal of the block [`Self::process`] just shaped, by
    /// [`shape_index`].
    #[inline]
    pub fn signal(&self, index: usize) -> &[f64] {
        if let Some(output) = index
            .checked_sub(AMBIENT_EXPANDED)
            .and_then(|slot| self.ambient_output_slots.get(slot).copied().flatten())
        {
            return &self.ambient_expanded[output];
        }
        &self.shaped[index]
    }

    pub fn dynamic_eq_gain_reduction_db(&self) -> f64 {
        self.dynamic_eq
            .as_ref()
            .map_or(0.0, StemDynamicEq::gain_reduction_db)
    }

    pub fn dynamics_gain_reduction_db(&self) -> f64 {
        self.dynamics
            .as_ref()
            .map_or(0.0, StemDynamics::gain_reduction_db)
    }
}

/// Index into the dry signals a speaker can draw on: the three dry shapes,
/// then the four shaped sends. Ambient sends use their dedicated slots.
pub fn shape_index(shape: SendShape) -> usize {
    match shape {
        SendShape::Left => 0,
        SendShape::Right => 1,
        SendShape::Mono => 2,
        SendShape::SurroundLeft => 3,
        SendShape::SurroundRight => 4,
        SendShape::HeightLeft => 5,
        SendShape::HeightRight => 6,
    }
}

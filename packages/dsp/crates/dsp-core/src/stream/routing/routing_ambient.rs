use crate::kernels::biquad::SosFilter;
use crate::kernels::butter::linkwitz_riley_lowpass_sos;
use crate::routing::ambient::AmbientSplit;

use super::{StemRouteState, AMBIENT_HEIGHT, AMBIENT_SURROUND, AMBIENT_TEXTURE, STEM_INPUT};
use crate::routing::sends::height_texture_sos;
use crate::stream::params::SendParams;

impl StemRouteState {
    /// Build or drop the ambient half.
    pub fn set_ambient(
        &mut self,
        sample_rate: u32,
        p: &SendParams,
        wanted: bool,
        source_wanted: bool,
        height_crossover_hz: f64,
        height_cutoff_hz: f64,
        ambient_trim_db: f64,
        _height_texture: f64,
    ) {
        self.ambient_trim_target = 10.0_f64.powf(valid_trim_db(ambient_trim_db) / 20.0);
        self.height_texture_cutoff_target = valid_texture_cutoff(height_cutoff_hz);
        if valid_texture_amount(_height_texture) > 0.0 {
            self.texture_active = true;
        }
        if !wanted {
            if self.split.is_none() && !self.texture_active {
                self.ambient_gains_initialized = false;
                self.height_texture_initialized = false;
            }
            return;
        }
        if source_wanted && self.split.is_none() {
            self.split = Some(AmbientSplit::with_height_crossover_and_cutoff(
                sample_rate,
                height_crossover_hz,
                height_cutoff_hz,
            ));
            self.ambient_gains_initialized = false;
        } else if let Some(split) = &mut self.split {
            split.set_height_crossover(height_crossover_hz);
            split.set_height_cutoff(height_cutoff_hz);
        }
        for s in self.ambient_surround.iter_mut() {
            s.retune_surround(sample_rate, p);
        }
        for s in self.ambient_height.iter_mut() {
            s.retune_height(sample_rate, p);
        }
        for s in self.ambient_texture.iter_mut() {
            s.retune_height(sample_rate, p);
        }
    }

    /// Take the block's ambient half out of the scratch pair and leave it,
    /// shaped, in the four ambient signals.
    pub(super) fn split_ambient(
        &mut self,
        start: usize,
        count: usize,
        rear: [f64; 2],
        height: [f64; 2],
    ) {
        if !self.ambient_gains_initialized {
            for side in 0..2 {
                self.ambient_rear_gain[side].set(rear[side]);
                self.ambient_height_gain[side].set(height[side]);
            }
            self.ambient_trim_gain.set(self.ambient_trim_target);
            self.ambient_gains_initialized = true;
        }
        let start_rear = std::array::from_fn(|side| self.ambient_rear_gain[side].current());
        let start_height = std::array::from_fn(|side| self.ambient_height_gain[side].current());
        let Some(split) = &mut self.split else { return };
        let block = split.advance_with_side_amounts(
            self.ahead.base,
            &self.ahead.left,
            &self.ahead.right,
            start,
            count,
            start_rear,
            start_height,
        );
        let sources = [
            (AMBIENT_SURROUND, block.rear[0]),
            (AMBIENT_SURROUND + 1, block.rear[1]),
            (AMBIENT_HEIGHT, block.height[0]),
            (AMBIENT_HEIGHT + 1, block.height[1]),
        ];
        for (slot, source) in sources {
            self.shaped[slot].clear();
            self.shaped[slot].extend_from_slice(source);
        }
        let mut rear_gain = self.ambient_rear_gain;
        let mut height_gain = self.ambient_height_gain;
        for sample in 0..count {
            let rear_left = rear_gain[0].tick(rear[0]);
            let rear_right = rear_gain[1].tick(rear[1]);
            let height_left = height_gain[0].tick(height[0]);
            let height_right = height_gain[1].tick(height[1]);
            self.scratch[0][sample] = self.shaped[STEM_INPUT][sample]
                - rear_left * self.shaped[AMBIENT_SURROUND][sample]
                - height_left.min(1.0 - rear_left) * self.shaped[AMBIENT_HEIGHT][sample];
            self.scratch[1][sample] = self.shaped[STEM_INPUT + 1][sample]
                - rear_right * self.shaped[AMBIENT_SURROUND + 1][sample]
                - height_right.min(1.0 - rear_right) * self.shaped[AMBIENT_HEIGHT + 1][sample];
        }
        for i in 0..2 {
            self.ambient_surround[i].process_in_place(&mut self.shaped[AMBIENT_SURROUND + i]);
            self.ambient_height[i].process_in_place(&mut self.shaped[AMBIENT_HEIGHT + i]);
        }
        let mut trim_gain = self.ambient_trim_gain;
        for sample in 0..count {
            let wet_trim = trim_gain.tick(self.ambient_trim_target);
            for i in 0..2 {
                self.shaped[AMBIENT_SURROUND + i][sample] *=
                    self.ambient_rear_gain[i].tick(rear[i]) * wet_trim;
                self.shaped[AMBIENT_HEIGHT + i][sample] *=
                    self.ambient_height_gain[i].tick(height[i]) * wet_trim;
            }
        }
        self.ambient_trim_gain = trim_gain;
    }

    /// Add direct-residual detail to the height feed. This deliberately uses
    /// only the existing height voicing plus a stateful high-pass; it never
    /// enters the ambient subtraction.
    pub(super) fn apply_height_texture(&mut self, count: usize, texture: [f64; 2]) {
        let first = !self.height_texture_initialized;
        if first {
            if !self.texture_started {
                for side in 0..2 {
                    self.height_texture_gain[side].set(texture[side]);
                }
                self.height_texture_cutoff
                    .set(self.height_texture_cutoff_target);
            }
            self.height_texture_initialized = true;
        }
        let cutoff = if first && !self.texture_started {
            self.height_texture_cutoff.current()
        } else {
            self.height_texture_cutoff
                .advance(self.height_texture_cutoff_target, count)
        };
        let cutoff_sos = height_texture_sos(self.sample_rate_hz as u32, cutoff);
        for side in 0..2 {
            self.texture_scratch[side].clear();
            self.texture_scratch[side].extend_from_slice(&self.scratch[side][..count]);
            self.ambient_texture_cutoff[side].retune_flat(&cutoff_sos);
            self.ambient_texture_cutoff[side].process(&mut self.texture_scratch[side]);
            self.ambient_texture[side].process_in_place(&mut self.texture_scratch[side]);
            let output = &mut self.shaped[AMBIENT_TEXTURE + side];
            if output.len() != count {
                output.resize(count, 0.0);
            }
            for sample in 0..count {
                let gain = self.height_texture_gain[side].tick(texture[side]);
                output[sample] += gain * self.texture_scratch[side][sample];
            }
        }
    }
}

fn valid_trim_db(value: f64) -> f64 {
    if value.is_finite() {
        value.clamp(0.0, 6.0)
    } else {
        0.0
    }
}

fn valid_texture_amount(value: f64) -> f64 {
    if value.is_finite() {
        value.clamp(0.0, 0.25)
    } else {
        0.0
    }
}

fn valid_texture_cutoff(value: f64) -> f64 {
    if value.is_finite() {
        value.clamp(
            crate::routing::ambient::AMBIENT_HEIGHT_CROSSOVER_MIN_HZ,
            crate::routing::ambient::AMBIENT_HEIGHT_CROSSOVER_MAX_HZ,
        )
    } else {
        crate::routing::ambient::AMBIENT_HEIGHT_CROSSOVER_HZ
    }
}

/// The LFE bus: stems sum in dry, then the whole bus is filtered once.
pub struct LfeBus {
    filter: SosFilter,
    gain: f64,
}

impl LfeBus {
    pub fn new(sample_rate: u32, p: &SendParams) -> Self {
        let nyq = sample_rate as f64 / 2.0;
        Self {
            filter: SosFilter::from_flat(&linkwitz_riley_lowpass_sos(
                p.lfe_filter_order,
                p.lfe_cutoff_hz / nyq,
            )),
            gain: p.lfe_gain,
        }
    }

    pub fn reset(&mut self) {
        self.filter.reset();
    }

    /// Re-derive the lowpass and gain in place, keeping the filter state.
    pub fn retune(&mut self, sample_rate: u32, p: &SendParams) {
        let nyq = sample_rate as f64 / 2.0;
        let sos = linkwitz_riley_lowpass_sos(p.lfe_filter_order, p.lfe_cutoff_hz / nyq);
        self.filter.retune_flat(&sos);
        self.gain = p.lfe_gain;
    }

    #[inline]
    pub fn tick(&mut self, summed: f64) -> f64 {
        self.filter.tick(summed) * self.gain
    }
}

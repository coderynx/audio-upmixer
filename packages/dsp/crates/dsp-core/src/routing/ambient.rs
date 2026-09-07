//! Per-stem direct/ambient split for rear and height sends.
//!
//! The split uses a band-regularized complex multichannel Wiener estimate.
//! It follows Paulus and Torcoli's primary/ambient decomposition, extending
//! its real stereo covariance to preserve phase-coherent direct sources.

use rustfft::num_complex::Complex64;

use crate::kernels::fft::RealFft;
use crate::kernels::stft::hann_periodic;

mod ambient_bands;
use ambient_bands::{erb_bands, Band, BandInterpolation};

/// Analysis length. 21 ms at 48 kHz keeps the look-ahead small while
/// resolving a reverb tail from the note that caused it.
pub const AMBIENT_FFT_SIZE: usize = 1024;

/// Four overlaps give constant overlap-add with the square-root Hann pair.
pub const AMBIENT_OVERLAP: usize = 4;

/// Covariance frames in the primary/ambient estimate.
pub const AMBIENT_COVARIANCE_FRAMES: usize = 5;

/// Wiener-matrix frames averaged after covariance estimation.
pub const AMBIENT_MATRIX_FRAMES: usize = 3;

/// Smallest analysis band. Narrower low-frequency bands are too noisy.
pub const AMBIENT_BAND_MIN_BINS: usize = 3;

/// Default frequency where the ambient half is shared equally by rear and
/// height sends.
pub const AMBIENT_HEIGHT_CROSSOVER_HZ: f64 = 2000.0;
pub const AMBIENT_HEIGHT_CROSSOVER_MIN_HZ: f64 = 500.0;
pub const AMBIENT_HEIGHT_CROSSOVER_MAX_HZ: f64 = 4000.0;

const EPS: f64 = 1e-20;
const RELATIVE_ENERGY_FLOOR: f64 = 1e-6;

/// One block of ambient signal, already split into its two destinations.
pub struct AmbientBlock<'a> {
    /// Direct residual after bounded ambient removal.
    pub direct: [&'a [f64]; 2],
    pub rear: [&'a [f64]; 2],
    pub height: [&'a [f64]; 2],
}

#[derive(Clone, Copy, Default)]
struct Covariance {
    ll: f64,
    lr: Complex64,
    rr: f64,
}

impl Covariance {
    fn add(self, other: Self) -> Self {
        Self {
            ll: self.ll + other.ll,
            lr: self.lr + other.lr,
            rr: self.rr + other.rr,
        }
    }

    fn subtract(self, other: Self) -> Self {
        Self {
            ll: self.ll - other.ll,
            lr: self.lr - other.lr,
            rr: self.rr - other.rr,
        }
    }

    fn scale(self, gain: f64) -> Self {
        Self {
            ll: self.ll * gain,
            lr: self.lr * gain,
            rr: self.rr * gain,
        }
    }
}

/// Hermitian ambient matrix: `[ll, lr; conj(lr), rr]`.
#[derive(Clone, Copy, Default)]
struct AmbientMatrix {
    ll: f64,
    lr: Complex64,
    rr: f64,
}

impl AmbientMatrix {
    fn add(self, other: Self) -> Self {
        Self {
            ll: self.ll + other.ll,
            lr: self.lr + other.lr,
            rr: self.rr + other.rr,
        }
    }

    fn subtract(self, other: Self) -> Self {
        Self {
            ll: self.ll - other.ll,
            lr: self.lr - other.lr,
            rr: self.rr - other.rr,
        }
    }

    fn scale(self, gain: f64) -> Self {
        Self {
            ll: self.ll * gain,
            lr: self.lr * gain,
            rr: self.rr * gain,
        }
    }

    fn lerp(self, other: Self, amount: f64) -> Self {
        self.scale(1.0 - amount).add(other.scale(amount))
    }
}

pub struct AmbientSplit {
    fft: RealFft,
    n: usize,
    hop: usize,
    bin_hz: f64,
    window: Vec<f64>,
    cola: Vec<f64>,
    bands: Vec<Band>,
    interpolation: Vec<BandInterpolation>,
    covariance_history: Vec<Vec<Covariance>>,
    covariance_sum: Vec<Covariance>,
    covariance_cursor: usize,
    covariance_count: usize,
    matrix_history: Vec<Vec<AmbientMatrix>>,
    matrix_sum: Vec<AmbientMatrix>,
    matrix_cursor: usize,
    matrix_count: usize,
    masked: [[Vec<Complex64>; 2]; 2],
    next_frame: usize,
    ola: [[Vec<f64>; 2]; 2],
    ring: usize,
    height_crossover_hz: f64,
    target_height_crossover_hz: f64,
    height_cutoff_hz: f64,
    target_height_cutoff_hz: f64,
    rear_amount: [f64; 2],
    height_amount: [f64; 2],
    direct: [Vec<f64>; 2],
    rear: [Vec<f64>; 2],
    height: [Vec<f64>; 2],
    cursor: Option<usize>,
    frame: Vec<f64>,
    spectrum: [Vec<Complex64>; 2],
}

impl AmbientSplit {
    pub fn new(sample_rate: u32) -> Self {
        Self::with_height_crossover(sample_rate, AMBIENT_HEIGHT_CROSSOVER_HZ)
    }

    pub fn with_height_crossover(sample_rate: u32, height_crossover_hz: f64) -> Self {
        let n = AMBIENT_FFT_SIZE;
        let hop = n / AMBIENT_OVERLAP;
        let window: Vec<f64> = hann_periodic(n).into_iter().map(f64::sqrt).collect();
        let mut cola = vec![0.0; hop];
        for (i, w) in window.iter().enumerate() {
            cola[i % hop] += w * w;
        }
        for value in &mut cola {
            *value = 1.0 / *value;
        }
        let ring = (n + 4 * hop).next_power_of_two();
        let bins = n / 2 + 1;
        let (bands, interpolation) = erb_bands(bins, sample_rate, n);
        let band_count = bands.len();
        let height_crossover_hz = valid_height_crossover(height_crossover_hz);
        Self {
            fft: RealFft::new(n),
            n,
            hop,
            bin_hz: sample_rate as f64 / n as f64,
            window,
            cola,
            bands,
            interpolation,
            covariance_history: vec![
                vec![Covariance::default(); band_count];
                AMBIENT_COVARIANCE_FRAMES
            ],
            covariance_sum: vec![Covariance::default(); band_count],
            covariance_cursor: 0,
            covariance_count: 0,
            matrix_history: vec![vec![AmbientMatrix::default(); band_count]; AMBIENT_MATRIX_FRAMES],
            matrix_sum: vec![AmbientMatrix::default(); band_count],
            matrix_cursor: 0,
            matrix_count: 0,
            masked: std::array::from_fn(|_| {
                std::array::from_fn(|_| vec![Complex64::new(0.0, 0.0); bins])
            }),
            next_frame: 0,
            ola: std::array::from_fn(|_| std::array::from_fn(|_| vec![0.0; ring])),
            ring,
            height_crossover_hz,
            target_height_crossover_hz: height_crossover_hz,
            height_cutoff_hz: AMBIENT_HEIGHT_CROSSOVER_HZ,
            target_height_cutoff_hz: AMBIENT_HEIGHT_CROSSOVER_HZ,
            rear_amount: [0.0; 2],
            height_amount: [0.0; 2],
            direct: [Vec::new(), Vec::new()],
            rear: [Vec::new(), Vec::new()],
            height: [Vec::new(), Vec::new()],
            cursor: None,
            frame: vec![0.0; n],
            spectrum: [
                vec![Complex64::new(0.0, 0.0); bins],
                vec![Complex64::new(0.0, 0.0); bins],
            ],
        }
    }

    pub fn with_height_crossover_and_cutoff(
        sample_rate: u32,
        height_crossover_hz: f64,
        height_cutoff_hz: f64,
    ) -> Self {
        let mut split = Self::with_height_crossover(sample_rate, height_crossover_hz);
        split.height_cutoff_hz = valid_height_crossover(height_cutoff_hz);
        split.target_height_cutoff_hz = split.height_cutoff_hz;
        split
    }

    /// Samples the split reads past the end of the block it is asked for.
    pub fn look_ahead(&self) -> usize {
        self.n
    }

    /// Move a live crossover edit at the analysis-frame cadence.
    pub fn set_height_crossover(&mut self, height_crossover_hz: f64) {
        self.target_height_crossover_hz = valid_height_crossover(height_crossover_hz);
    }

    /// Move the revision-2 height voicing cutoff at the analysis-frame rate.
    pub fn set_height_cutoff(&mut self, height_cutoff_hz: f64) {
        self.target_height_cutoff_hz = valid_height_crossover(height_cutoff_hz);
    }

    pub fn reset(&mut self) {
        for history in &mut self.covariance_history {
            history.fill(Covariance::default());
        }
        self.covariance_sum.fill(Covariance::default());
        self.covariance_cursor = 0;
        self.covariance_count = 0;
        for history in &mut self.matrix_history {
            history.fill(AmbientMatrix::default());
        }
        self.matrix_sum.fill(AmbientMatrix::default());
        self.matrix_cursor = 0;
        self.matrix_count = 0;
        self.next_frame = 0;
        self.cursor = None;
        for destination in &mut self.ola {
            for channel in destination {
                channel.fill(0.0);
            }
        }
    }

    /// Ambient rear and height pairs for `[start, start + len)` of a stem.
    ///
    /// `left` and `right` cover `[base, base + left.len())` and must reach
    /// [`Self::look_ahead`] past the block.
    pub fn advance(
        &mut self,
        base: usize,
        left: &[f64],
        right: &[f64],
        start: usize,
        len: usize,
    ) -> AmbientBlock<'_> {
        self.advance_with_amounts(base, left, right, start, len, 0.0, 0.0)
    }

    /// Ambient pairs plus a direct residual whose removal is bounded
    /// independently at every STFT bin. It uses `R = r + min(h, 1-r) * H`;
    /// this conservative overlap
    /// keeps `R` in `[0, 1]` while reusing the rear and height transforms.
    /// `rear_amount` and `height_amount` are layout-gated by the caller.
    pub fn advance_with_amounts(
        &mut self,
        base: usize,
        left: &[f64],
        right: &[f64],
        start: usize,
        len: usize,
        rear_amount: f64,
        height_amount: f64,
    ) -> AmbientBlock<'_> {
        self.advance_with_side_amounts(
            base,
            left,
            right,
            start,
            len,
            [rear_amount; 2],
            [height_amount; 2],
        )
    }

    /// The same route with independent left/right destination amounts. A
    /// layout with no destination for one side passes zero for that side, so
    /// its direct anchor is never attenuated by an unrendered send.
    pub fn advance_with_side_amounts(
        &mut self,
        base: usize,
        left: &[f64],
        right: &[f64],
        start: usize,
        len: usize,
        rear_amount: [f64; 2],
        height_amount: [f64; 2],
    ) -> AmbientBlock<'_> {
        self.rear_amount = rear_amount.map(valid_amount);
        self.height_amount = height_amount.map(valid_amount);
        let height_removal = [
            self.height_amount[0].min(1.0 - self.rear_amount[0]),
            self.height_amount[1].min(1.0 - self.rear_amount[1]),
        ];
        if self.cursor != Some(start) {
            self.reset();
            self.cursor = Some(start);
            self.next_frame = start / self.hop;
        }

        for channel in 0..2 {
            self.direct[channel].resize(len, 0.0);
            self.rear[channel].resize(len, 0.0);
            self.height[channel].resize(len, 0.0);
        }

        let mut done = 0;
        while done < len {
            let position = start + done;
            let frame = position / self.hop;
            while self.next_frame <= frame {
                self.process_frame(base, left, right, self.next_frame);
                self.next_frame += 1;
            }
            let until = ((frame + 1) * self.hop).min(start + len);
            for offset in position..until {
                let slot = offset & (self.ring - 1);
                let cola_inv = self.cola[offset & (self.hop - 1)];
                let index = offset - start;
                let source_index = offset.checked_sub(base);
                let source_left = source_index
                    .and_then(|source_index| left.get(source_index))
                    .copied()
                    .unwrap_or(0.0);
                let source_right = source_index
                    .and_then(|source_index| right.get(source_index))
                    .copied()
                    .unwrap_or(0.0);
                for channel in 0..2 {
                    let source_sample = if channel == 0 {
                        source_left
                    } else {
                        source_right
                    };
                    let removal = self.rear_amount[channel] * self.ola[0][channel][slot]
                        + height_removal[channel] * self.ola[1][channel][slot];
                    self.direct[channel][index] = source_sample - removal * cola_inv;
                    self.rear[channel][index] = self.ola[0][channel][slot] * cola_inv;
                    self.height[channel][index] = self.ola[1][channel][slot] * cola_inv;
                    self.ola[0][channel][slot] = 0.0;
                    self.ola[1][channel][slot] = 0.0;
                }
            }
            done += until - position;
        }
        self.cursor = Some(start + len);

        AmbientBlock {
            direct: [&self.direct[0], &self.direct[1]],
            rear: [&self.rear[0], &self.rear[1]],
            height: [&self.height[0], &self.height[1]],
        }
    }

    fn process_frame(&mut self, base: usize, left: &[f64], right: &[f64], index: usize) {
        self.height_crossover_hz +=
            (self.target_height_crossover_hz - self.height_crossover_hz) * 0.5;
        self.height_cutoff_hz +=
            (self.target_height_cutoff_hz - self.height_cutoff_hz) * 0.5;
        let frame_start = index * self.hop;
        self.windowed(left, base, frame_start, 0);
        self.windowed(right, base, frame_start, 1);

        let mut total_power = 0.0;
        for bin in 0..self.spectrum[0].len() {
            total_power += self.spectrum[0][bin].norm_sqr() + self.spectrum[1][bin].norm_sqr();
        }
        let gate = (total_power / self.spectrum[0].len() as f64 * RELATIVE_ENERGY_FLOOR).max(EPS);
        let covariance_divisor = (self.covariance_count + 1).min(AMBIENT_COVARIANCE_FRAMES) as f64;
        let matrix_divisor = (self.matrix_count + 1).min(AMBIENT_MATRIX_FRAMES) as f64;

        for band_index in 0..self.bands.len() {
            let band = &self.bands[band_index];
            let mut covariance = Covariance::default();
            for bin in band.start..band.end {
                let l = self.spectrum[0][bin];
                let r = self.spectrum[1][bin];
                covariance.ll += l.norm_sqr();
                covariance.lr += l * r.conj();
                covariance.rr += r.norm_sqr();
            }
            covariance = covariance.scale(1.0 / (band.end - band.start) as f64);

            let old_covariance = self.covariance_history[self.covariance_cursor][band_index];
            self.covariance_history[self.covariance_cursor][band_index] = covariance;
            self.covariance_sum[band_index] = self.covariance_sum[band_index]
                .add(covariance)
                .subtract(old_covariance);
            let estimate = self.covariance_sum[band_index].scale(1.0 / covariance_divisor);
            let target = ambient_matrix(estimate, gate);

            let old_matrix = self.matrix_history[self.matrix_cursor][band_index];
            self.matrix_history[self.matrix_cursor][band_index] = target;
            self.matrix_sum[band_index] =
                self.matrix_sum[band_index].add(target).subtract(old_matrix);
        }
        self.covariance_cursor = (self.covariance_cursor + 1) % AMBIENT_COVARIANCE_FRAMES;
        self.covariance_count = (self.covariance_count + 1).min(AMBIENT_COVARIANCE_FRAMES);
        self.matrix_cursor = (self.matrix_cursor + 1) % AMBIENT_MATRIX_FRAMES;
        self.matrix_count = (self.matrix_count + 1).min(AMBIENT_MATRIX_FRAMES);

        for bin in 0..self.spectrum[0].len() {
            let interpolation = self.interpolation[bin];
            let lower = self.matrix_sum[interpolation.lower].scale(1.0 / matrix_divisor);
            let upper = self.matrix_sum[interpolation.upper].scale(1.0 / matrix_divisor);
            let matrix = lower.lerp(upper, interpolation.amount);
            let left = self.spectrum[0][bin];
            let right = self.spectrum[1][bin];
            let ambient_left = left * matrix.ll + right * matrix.lr;
            let ambient_right = left * matrix.lr.conj() + right * matrix.rr;
            let height = height_mask(bin as f64 * self.bin_hz, self.height_cutoff_hz);
            self.masked[0][0][bin] = ambient_left;
            self.masked[0][1][bin] = ambient_right;
            self.masked[1][0][bin] = ambient_left * height;
            self.masked[1][1][bin] = ambient_right * height;
        }
        self.overlap_add(frame_start);
    }

    /// Windowed transform of one channel at `base`, left in `spectrum[side]`.
    fn windowed(&mut self, signal: &[f64], base: usize, frame_start: usize, side: usize) {
        self.frame.fill(0.0);
        if let Some(start) = frame_start.checked_sub(base) {
            let available = signal.len().saturating_sub(start).min(self.n);
            self.frame[..available].copy_from_slice(&signal[start..start + available]);
            for (sample, window) in self.frame[..available].iter_mut().zip(&self.window) {
                *sample *= *window;
            }
        }
        self.fft
            .rfft_into(&mut self.frame, &mut self.spectrum[side]);
    }

    fn overlap_add(&mut self, base: usize) {
        for destination in 0..2 {
            for channel in 0..2 {
                self.fft
                    .irfft_into(&mut self.masked[destination][channel], &mut self.frame);
                for i in 0..self.n {
                    let slot = (base + i) & (self.ring - 1);
                    self.ola[destination][channel][slot] += self.frame[i] * self.window[i];
                }
            }
        }
    }
}

fn valid_height_crossover(value: f64) -> f64 {
    if value.is_finite()
        && (AMBIENT_HEIGHT_CROSSOVER_MIN_HZ..=AMBIENT_HEIGHT_CROSSOVER_MAX_HZ).contains(&value)
    {
        value
    } else {
        AMBIENT_HEIGHT_CROSSOVER_HZ
    }
}

fn valid_amount(value: f64) -> f64 {
    if value.is_finite() {
        value.clamp(0.0, 1.0)
    } else {
        0.0
    }
}

/// Height's real-valued share of one ambient STFT bin.
pub fn height_mask(frequency_hz: f64, crossover_hz: f64) -> f64 {
    let ratio = (frequency_hz.max(0.0) / valid_height_crossover(crossover_hz)).max(0.0);
    if ratio <= 1.0 {
        let power = ratio.powi(8);
        power / (1.0 + power)
    } else {
        1.0 / (1.0 + ratio.recip().powi(8))
    }
}

fn ambient_matrix(mut covariance: Covariance, gate: f64) -> AmbientMatrix {
    if !covariance.ll.is_finite()
        || !covariance.rr.is_finite()
        || !covariance.lr.re.is_finite()
        || !covariance.lr.im.is_finite()
        || covariance.ll + covariance.rr <= gate
    {
        return AmbientMatrix::default();
    }
    covariance.ll = covariance.ll.max(0.0);
    covariance.rr = covariance.rr.max(0.0);
    let cross_limit = (covariance.ll * covariance.rr).sqrt();
    if covariance.lr.norm() > cross_limit {
        covariance.lr *= cross_limit / covariance.lr.norm();
    }
    let lambda_max = (covariance.ll
        + covariance.rr
        + ((covariance.ll - covariance.rr).powi(2) + 4.0 * covariance.lr.norm_sqr()).sqrt())
        * 0.5;
    if lambda_max <= EPS || !lambda_max.is_finite() {
        return AmbientMatrix::default();
    }
    AmbientMatrix {
        ll: covariance.rr / lambda_max,
        lr: -covariance.lr / lambda_max,
        rr: covariance.ll / lambda_max,
    }
}

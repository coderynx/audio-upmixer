use super::AMBIENT_BAND_MIN_BINS;

pub(super) struct Band {
    pub(super) start: usize,
    pub(super) end: usize,
    pub(super) centre_erb: f64,
}

#[derive(Clone, Copy)]
pub(super) struct BandInterpolation {
    pub(super) lower: usize,
    pub(super) upper: usize,
    pub(super) amount: f64,
}

pub(super) fn erb_bands(
    bins: usize,
    sample_rate: u32,
    n: usize,
) -> (Vec<Band>, Vec<BandInterpolation>) {
    let bin_hz = sample_rate as f64 / n as f64;
    let mut bands: Vec<Band> = Vec::new();
    let mut start = 0;
    while start < bins {
        let remaining = bins - start;
        if remaining < AMBIENT_BAND_MIN_BINS && !bands.is_empty() {
            bands.last_mut().expect("band exists").end = bins;
            break;
        }
        let start_hz = start as f64 * bin_hz;
        let end_hz = erb_hz(erb_rate(start_hz) + 1.0);
        let mut end = (end_hz / bin_hz).ceil() as usize;
        end = end.max(start + AMBIENT_BAND_MIN_BINS).min(bins);
        if bins - end < AMBIENT_BAND_MIN_BINS {
            end = bins;
        }
        let centre_erb = (erb_rate(start_hz) + erb_rate((end - 1) as f64 * bin_hz)) * 0.5;
        bands.push(Band {
            start,
            end,
            centre_erb,
        });
        start = end;
    }

    let mut interpolation = Vec::with_capacity(bins);
    let mut lower = 0;
    for bin in 0..bins {
        let rate = erb_rate(bin as f64 * bin_hz);
        while lower + 1 < bands.len() && rate > bands[lower + 1].centre_erb {
            lower += 1;
        }
        let upper = (lower + 1).min(bands.len() - 1);
        let amount = if lower == upper {
            0.0
        } else {
            ((rate - bands[lower].centre_erb) / (bands[upper].centre_erb - bands[lower].centre_erb))
                .clamp(0.0, 1.0)
        };
        interpolation.push(BandInterpolation {
            lower,
            upper,
            amount,
        });
    }
    (bands, interpolation)
}

fn erb_rate(hz: f64) -> f64 {
    21.4 * (1.0 + 0.00437 * hz).log10()
}

fn erb_hz(rate: f64) -> f64 {
    (10.0_f64.powf(rate / 21.4) - 1.0) / 0.00437
}

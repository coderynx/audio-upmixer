mod common;

mod ambient {
    use upmixer_dsp_core::kernels::rng::next_unit;
    use upmixer_dsp_core::routing::ambient::*;
    use upmixer_dsp_core::routing::ambient_expander::{ambient_expander_fir, FixedAmbientExpander};

    const SR: u32 = 48_000;
    const N: usize = 48_000;
    const ALL_DESTINATIONS: [&str; 8] = ["SL", "SR", "BL", "BR", "TFL", "TFR", "TBL", "TBR"];

    fn noise(seed: u64, n: usize) -> Vec<f64> {
        let mut state = seed;
        (0..n).map(|_| next_unit(&mut state) * 2.0 - 1.0).collect()
    }

    fn tone(freq: f64, n: usize) -> Vec<f64> {
        (0..n)
            .map(|i| {
                let t = i as f64 / SR as f64;
                0.5 * (2.0 * std::f64::consts::PI * freq * t).sin()
            })
            .collect()
    }

    /// Ambient rear+height pairs over the whole signal, in `block` steps.
    fn split_all(left: &[f64], right: &[f64], block: usize) -> [Vec<f64>; 4] {
        let mut split = AmbientSplit::new(SR);
        let mut out: [Vec<f64>; 4] = Default::default();
        let mut start = 0;
        while start < left.len() {
            let len = block.min(left.len() - start);
            let piece = split.advance(0, left, right, start, len);
            out[0].extend_from_slice(piece.rear[0]);
            out[1].extend_from_slice(piece.rear[1]);
            out[2].extend_from_slice(piece.height[0]);
            out[3].extend_from_slice(piece.height[1]);
            start += len;
        }
        out
    }

    /// Revision-2 outputs over the whole signal, retaining the direct
    /// residual so its bounded subtraction can be compared independently.
    fn split_overlap_all(
        left: &[f64],
        right: &[f64],
        block: usize,
        rear: [f64; 2],
        height: [f64; 2],
    ) -> [Vec<f64>; 6] {
        split_overlap_all_with_cutoff(
            left,
            right,
            block,
            rear,
            height,
            AMBIENT_HEIGHT_CROSSOVER_HZ,
        )
    }

    fn split_overlap_all_with_cutoff(
        left: &[f64],
        right: &[f64],
        block: usize,
        rear: [f64; 2],
        height: [f64; 2],
        cutoff: f64,
    ) -> [Vec<f64>; 6] {
        let mut split = AmbientSplit::with_height_crossover_and_cutoff(
            SR,
            AMBIENT_HEIGHT_CROSSOVER_HZ,
            cutoff,
        );
        let mut out: [Vec<f64>; 6] = Default::default();
        let mut start = 0;
        while start < left.len() {
            let len = block.min(left.len() - start);
            let piece = split.advance_with_side_amounts(
                0, left, right, start, len, rear, height,
            );
            out[0].extend_from_slice(piece.direct[0]);
            out[1].extend_from_slice(piece.direct[1]);
            out[2].extend_from_slice(piece.rear[0]);
            out[3].extend_from_slice(piece.rear[1]);
            out[4].extend_from_slice(piece.height[0]);
            out[5].extend_from_slice(piece.height[1]);
            start += len;
        }
        out
    }

    /// Mean power over the settled part, past the overlap-add ramp-in.
    fn power(signal: &[f64]) -> f64 {
        let settled = &signal[AMBIENT_FFT_SIZE..];
        settled.iter().map(|v| v * v).sum::<f64>() / settled.len() as f64
    }

    #[test]
    fn a_perfectly_correlated_pair_stays_direct() {
        let mono = tone(440.0, N);
        let split = split_all(&mono, &mono, 512);
        let source: f64 = mono[AMBIENT_FFT_SIZE..].iter().map(|v| v * v).sum::<f64>()
            / (N - AMBIENT_FFT_SIZE) as f64;
        let ambient = power(&split[0]) + power(&split[2]);
        assert!(
            ambient / source < 1e-6,
            "correlated input leaked {:.4} of its power",
            ambient / source
        );
    }

    #[test]
    fn an_uncorrelated_pair_reaches_the_ambient_send() {
        let left = noise(1, N);
        let right = noise(2, N);
        let split = split_all(&left, &right, 512);
        let source: f64 = left[AMBIENT_FFT_SIZE..].iter().map(|v| v * v).sum::<f64>()
            / (N - AMBIENT_FFT_SIZE) as f64;
        let ambient = power(&split[0]) + power(&split[2]);
        assert!(
            ambient / source > 0.6,
            "uncorrelated input kept only {:.3} of its power",
            ambient / source
        );
    }

    #[test]
    fn a_hard_panned_primary_is_not_mistaken_for_ambient() {
        let left = tone(440.0, N);
        let right = vec![0.0; N];
        let split = split_all(&left, &right, 512);
        let source: f64 = left[AMBIENT_FFT_SIZE..].iter().map(|v| v * v).sum::<f64>()
            / (N - AMBIENT_FFT_SIZE) as f64;
        let ambient = power(&split[0]) + power(&split[2]);
        assert!(
            ambient / source < 1e-3,
            "a hard-panned tone sent {:.5} of its power to the ambient bus",
            ambient / source
        );
    }

    #[test]
    fn a_phase_shifted_primary_stays_direct() {
        let left = tone(880.0, N);
        let mut right = vec![0.0; N];
        right[16..].copy_from_slice(&left[..N - 16]);
        let split = split_all(&left, &right, 512);
        let ambient = power(&split[0]) + power(&split[2]);
        assert!(
            ambient / power(&left) < 1e-4,
            "delayed primary leaked {ambient:.5}"
        );
    }

    #[test]
    fn sustained_stereo_notes_do_not_warble() {
        for sample_rate in [44_100, 48_000, 96_000] {
            let n = sample_rate as usize;
            for spacing in [4.0, 6.0, 10.0, 20.0, 40.0] {
                let note = |frequency: f64, i: usize| {
                    0.5 * (2.0 * std::f64::consts::PI * frequency * i as f64
                        / sample_rate as f64).sin()
                };
                let (left, right): (Vec<_>, Vec<_>) = (0..n + AMBIENT_FFT_SIZE)
                    .map(|i| {
                        let (a, b) = (note(440.0, i), note(440.0 + spacing, i));
                        (a + 0.5 * b, 0.5 * a + b)
                    }).unzip();
                let mut split = AmbientSplit::new(sample_rate);
                let mut output: [Vec<f64>; 6] = Default::default();
                for start in (0..n).step_by(128) {
                    let block = split.advance_with_amounts(
                        0, &left, &right, start, 128.min(n - start), 1.0, 1.0,
                    );
                    for (output, signal) in output.iter_mut().zip(
                        block.direct.into_iter().chain(block.rear).chain(block.height)
                    ) {
                        output.extend_from_slice(signal);
                    }
                }
                // Exclude startup and provide look-ahead past the measured end.
                // Integer cycles make the sine/cosine projections orthogonal.
                for (channel, signal) in output.iter().enumerate() {
                    let signal = &signal[n / 2..n];
                    let mut residual = signal.to_vec();
                    for frequency in [440.0, 440.0 + spacing] {
                        for phase in [0.0, std::f64::consts::FRAC_PI_2] {
                            let basis: Vec<f64> = (0..signal.len()).map(|i| {
                                (2.0 * std::f64::consts::PI * frequency * i as f64
                                    / sample_rate as f64 + phase).sin()
                            }).collect();
                            let gain = signal.iter().zip(&basis).map(|(x, b)| x * b).sum::<f64>()
                                / basis.iter().map(|b| b * b).sum::<f64>();
                            for (sample, basis) in residual.iter_mut().zip(basis) {
                                *sample -= gain * basis;
                            }
                        }
                    }
                    // Sidebands below -40 dB relative to either input note.
                    let distortion = residual.iter().map(|v| v * v).sum::<f64>()
                        / (signal.len() as f64 * 0.125);
                    assert!(distortion < 1e-4,
                        "{sample_rate} Hz, spacing {spacing}, channel {channel}: modulation {distortion:.6}");
                }
            }
        }
    }

    #[test]
    fn the_height_masks_are_complementary_and_steep() {
        for crossover in [500.0, 2000.0, 4000.0] {
            for bin in 0..=AMBIENT_FFT_SIZE / 2 {
                let height =
                    height_mask(bin as f64 * SR as f64 / AMBIENT_FFT_SIZE as f64, crossover);
                let rear = 1.0 - height;
                assert!(height.is_finite() && (0.0..=1.0).contains(&height));
                assert!((rear + height - 1.0).abs() < 1e-15);
            }
            assert_eq!(height_mask(crossover, crossover), 0.5);
            assert!(height_mask(crossover * 0.5, crossover) < 0.004);
            assert!(height_mask(crossover * 2.0, crossover) > 0.996);
        }
    }

    #[test]
    fn the_split_is_independent_of_the_block_size() {
        let left = noise(5, N);
        let right = noise(6, N);
        let reference = split_all(&left, &right, N);
        for block in [128, 512, 4096] {
            let got = split_all(&left, &right, block);
            for (signal, want) in got.iter().zip(reference.iter()) {
                let worst = signal
                    .iter()
                    .zip(want)
                    .map(|(a, b)| (a - b).abs())
                    .fold(0.0_f64, f64::max);
                assert!(worst < 1e-9, "block {block} diverged by {worst:e}");
            }
        }
    }

    #[test]
    fn changing_the_revision2_cutoff_only_changes_height_voicing() {
        let left = noise(59, N);
        let right = noise(60, N);
        let low = split_overlap_all_with_cutoff(&left, &right, 512, [1.0; 2], [1.0; 2], 500.0);
        let high =
            split_overlap_all_with_cutoff(&left, &right, 512, [1.0; 2], [1.0; 2], 4000.0);
        for channel in [2, 3] {
            let worst = low[channel]
                .iter()
                .zip(&high[channel])
                .map(|(a, b)| (a - b).abs())
                .fold(0.0, f64::max);
            assert!(worst < 1e-12, "rear channel {channel} changed by {worst:e}");
        }
        let height_difference = low[4]
            .iter()
            .zip(&high[4])
            .map(|(a, b)| (a - b).abs())
            .fold(0.0, f64::max);
        assert!(height_difference > 1e-6, "height cutoff had no effect");
    }

    #[test]
    fn full_rear_overlap_does_not_remove_the_ambient_twice() {
        let left = noise(53, N);
        let right = noise(54, N);
        let rear_only = split_overlap_all(&left, &right, 512, [1.0; 2], [0.0; 2]);
        let both = split_overlap_all(&left, &right, 512, [1.0; 2], [1.0; 2]);
        for channel in 0..2 {
            let worst = rear_only[channel]
                .iter()
                .zip(&both[channel])
                .map(|(a, b)| (a - b).abs())
                .fold(0.0, f64::max);
            assert!(worst < 1e-12, "channel {channel} was subtracted twice: {worst:e}");
        }
    }

    #[test]
    fn overlapping_removal_uses_bounded_conservative_height_compensation() {
        let left = noise(61, N);
        let right = noise(62, N);
        let output = split_overlap_all(&left, &right, 512, [0.6; 2], [0.8; 2]);
        for (source, (direct, (rear, height))) in left
            .iter()
            .zip(output[0].iter().zip(output[2].iter().zip(&output[4])))
        {
            let expected = source - 0.6 * rear - 0.8_f64.min(1.0 - 0.6) * height;
            assert!((direct - expected).abs() < 1e-12);
        }
    }

    #[test]
    fn an_absent_source_side_keeps_its_direct_anchor() {
        let left = noise(55, N);
        let right = noise(56, N);
        let output = split_overlap_all(&left, &right, 512, [1.0, 0.0], [1.0, 0.0]);
        let worst = output[1]
            .iter()
            .zip(&right)
            .map(|(a, b)| (a - b).abs())
            .fold(0.0, f64::max);
        assert!(worst < 1e-12, "absent right destinations removed {worst:e}");
    }

    #[test]
    fn overlapping_residual_keeps_the_raw_start_sample_and_partition() {
        let left = noise(57, N);
        let right = noise(58, N);
        let whole = split_overlap_all(&left, &right, N, [0.8; 2], [0.6; 2]);
        assert_eq!(whole[0][0], left[0]);
        assert_eq!(whole[1][0], right[0]);
        let blocked = split_overlap_all(&left, &right, 257, [0.8; 2], [0.6; 2]);
        for (actual, expected) in blocked.iter().zip(&whole) {
            let worst = actual
                .iter()
                .zip(expected)
                .map(|(a, b)| (a - b).abs())
                .fold(0.0, f64::max);
            assert!(worst < 1e-9, "block partition changed output by {worst:e}");
        }
    }

    #[test]
    fn a_decaying_tail_reaches_the_send_and_the_note_under_it_does_not() {
        // What the stage exists for: a centred note with a decorrelated tail
        // behind it, the shape a separated stem actually has.
        let tail_l = noise(9, N);
        let tail_r = noise(10, N);
        let mut left = vec![0.0; N];
        let mut right = vec![0.0; N];
        let note = tone(660.0, N);
        for i in 0..N {
            let t = i as f64 / SR as f64;
            let note_gain = if t < 0.5 { 1.0 } else { 0.0 };
            let tail_gain = 0.2 * (-3.0 * t).exp();
            left[i] = note[i] * note_gain + tail_l[i] * tail_gain;
            right[i] = note[i] * note_gain + tail_r[i] * tail_gain;
        }
        let split = split_all(&left, &right, 512);
        let window = |signal: &[f64], from: usize, to: usize| {
            signal[from..to].iter().map(|v| v * v).sum::<f64>() / (to - from) as f64
        };
        let source = |signal: &[f64], from: usize, to: usize| {
            signal[from..to].iter().map(|v| v * v).sum::<f64>() / (to - from) as f64
        };
        let (note_from, note_to) = (SR as usize / 4, SR as usize / 2);
        let (tail_from, tail_to) = (SR as usize * 3 / 4, N);
        let under_note = (window(&split[0], note_from, note_to)
            + window(&split[2], note_from, note_to))
            / source(&left, note_from, note_to);
        let under_tail = (window(&split[0], tail_from, tail_to)
            + window(&split[2], tail_from, tail_to))
            / source(&left, tail_from, tail_to);
        assert!(
            under_tail > 20.0 * under_note,
            "tail {under_tail:.4} vs note {under_note:.4} — the split does not separate them"
        );
    }

    /// Broad-band matrix regularization must not create musical-noise motion.
    #[test]
    fn the_matrix_does_not_perforate_the_spectrum() {
        // Dense spectrum, so every bin has something to measure: a shared
        // (coherent) noise bed under an independent (diffuse) one.
        let common = noise(21, N);
        let (diffuse_l, diffuse_r) = (noise(22, N), noise(23, N));
        let left: Vec<f64> = (0..N).map(|i| common[i] + 0.5 * diffuse_l[i]).collect();
        let right: Vec<f64> = (0..N).map(|i| common[i] + 0.5 * diffuse_r[i]).collect();

        let split = split_all(&left, &right, 512);
        let ambient_l: Vec<f64> = split[0].iter().zip(&split[2]).map(|(a, b)| a + b).collect();
        let ambient_r: Vec<f64> = split[1].iter().zip(&split[3]).map(|(a, b)| a + b).collect();

        let (roughness, movement) = gain_statistics([&left, &right], [&ambient_l, &ambient_r]);
        assert!(
            roughness < 2.5,
            "matrix jumps {:.2} dB between neighbouring bands",
            roughness
        );
        assert!(
            movement < 2.0,
            "mask moves {:.2} dB over time, per bin",
            movement
        );
    }

    #[test]
    fn subtracting_the_ambient_send_keeps_the_direct_path_smooth() {
        let common = noise(31, N);
        let (diffuse_l, diffuse_r) = (noise(32, N), noise(33, N));
        let left: Vec<f64> = (0..N).map(|i| common[i] + 0.5 * diffuse_l[i]).collect();
        let right: Vec<f64> = (0..N).map(|i| common[i] + 0.5 * diffuse_r[i]).collect();
        let split = split_all(&left, &right, 512);
        let direct_l: Vec<f64> = left
            .iter()
            .zip(&split[0])
            .zip(&split[2])
            .map(|((source, rear), height)| source - rear - height)
            .collect();
        let direct_r: Vec<f64> = right
            .iter()
            .zip(&split[1])
            .zip(&split[3])
            .map(|((source, rear), height)| source - rear - height)
            .collect();

        let (roughness, movement) = gain_statistics([&left, &right], [&direct_l, &direct_r]);
        assert!(
            roughness < 1.5,
            "direct path jumps {:.2} dB between bins",
            roughness
        );
        assert!(
            movement < 3.0,
            "direct path moves {:.2} dB over time",
            movement
        );
    }

    fn gain_statistics(source: [&[f64]; 2], ambient: [&[f64]; 2]) -> (f64, f64) {
        use upmixer_dsp_core::kernels::fft::RealFft;
        use upmixer_dsp_core::kernels::stft::hann_periodic;

        let n = AMBIENT_FFT_SIZE;
        let hop = n / 2;
        let fft = RealFft::new(n);
        let window = hann_periodic(n);
        let bins = n / 2 + 1;
        let (lo, hi) = (bins / 32, bins * 2 / 3);
        let mut tracks: Vec<Vec<f64>> = Vec::new();
        let mut start = n;
        while start + n <= source[0].len() {
            let take = |signal: &[f64]| -> Vec<f64> {
                let framed: Vec<f64> = (0..n).map(|i| signal[start + i] * window[i]).collect();
                fft.rfft(&framed).iter().map(|v| v.norm()).collect()
            };
            let source_l = take(source[0]);
            let source_r = take(source[1]);
            let ambient_l = take(ambient[0]);
            let ambient_r = take(ambient[1]);
            tracks.push(
                (lo..hi)
                    .step_by(12)
                    .map(|start| {
                        let end = (start + 12).min(hi);
                        let source_power = (start..end)
                            .map(|bin| source_l[bin].powi(2) + source_r[bin].powi(2))
                            .sum::<f64>();
                        let ambient_power = (start..end)
                            .map(|bin| ambient_l[bin].powi(2) + ambient_r[bin].powi(2))
                            .sum::<f64>();
                        10.0 * (ambient_power / source_power.max(1e-18)).max(1e-8).log10()
                    })
                    .collect(),
            );
            start += hop;
        }
        let roughness = tracks
            .iter()
            .map(|frame| {
                frame.windows(2).map(|w| (w[1] - w[0]).abs()).sum::<f64>()
                    / (frame.len() - 1) as f64
            })
            .sum::<f64>()
            / tracks.len() as f64;
        let width = tracks[0].len();
        let movement = (0..width)
            .map(|bin| {
                let column: Vec<f64> = tracks.iter().map(|frame| frame[bin]).collect();
                let mean = column.iter().sum::<f64>() / column.len() as f64;
                (column.iter().map(|v| (v - mean).powi(2)).sum::<f64>() / column.len() as f64)
                    .sqrt()
            })
            .sum::<f64>()
            / width as f64;
        (roughness, movement)
    }

    /// The same pin `packages/core/tests/test_ambient_split.py` asserts
    /// through the wheel: a wasm preview and a Python export built from
    #[test]
    fn a_reset_split_repeats_itself() {
        let left = noise(7, 8192);
        let right = noise(8, 8192);
        let mut split = AmbientSplit::new(SR);
        let first: Vec<f64> = split.advance(0, &left, &right, 0, 4096).rear[0].to_vec();
        split.reset();
        let second: Vec<f64> = split.advance(0, &left, &right, 0, 4096).rear[0].to_vec();
        assert_eq!(first, second);
    }

    #[test]
    fn fixed_expander_is_partition_invariant_and_has_no_tail_past_its_fir() {
        let inputs = [
            noise(41, 4096),
            noise(42, 4096),
            noise(43, 4096),
            noise(44, 4096),
        ];
        let refs = [&inputs[0][..], &inputs[1], &inputs[2], &inputs[3]];
        let mut whole = FixedAmbientExpander::new(SR, &ALL_DESTINATIONS);
        let expected = whole.process(refs);

        let mut blocked = FixedAmbientExpander::new(SR, &ALL_DESTINATIONS);
        let mut actual = vec![Vec::new(); ALL_DESTINATIONS.len()];
        for start in (0..inputs[0].len()).step_by(127) {
            let end = (start + 127).min(inputs[0].len());
            for (output, block) in actual.iter_mut().zip(blocked.process([
                &inputs[0][start..end],
                &inputs[1][start..end],
                &inputs[2][start..end],
                &inputs[3][start..end],
            ])) {
                output.extend(block);
            }
        }
        assert_eq!(actual, expected);

        let impulse = [vec![1.0], vec![0.0], vec![0.0], vec![0.0]];
        let tail = [
            vec![0.0; 1441],
            vec![0.0; 1441],
            vec![0.0; 1441],
            vec![0.0; 1441],
        ];
        let span = ["SL", "SR", "BL", "BR", "TFL", "TFR", "TBL", "TBR"]
            .into_iter()
            .map(|destination| ambient_expander_fir(SR, destination).unwrap().span())
            .max()
            .unwrap();
        let mut expander = FixedAmbientExpander::new(SR, &ALL_DESTINATIONS);
        let _ = expander.process([&impulse[0], &impulse[1], &impulse[2], &impulse[3]]);
        for output in expander.process([&tail[0], &tail[1], &tail[2], &tail[3]]) {
            assert!(output[span..].iter().all(|sample| *sample == 0.0));
        }
    }

    #[test]
    fn fixed_expander_uses_a_unique_filter_per_destination() {
        let input = noise(45, 4096);
        let mut expander = FixedAmbientExpander::new(SR, &ALL_DESTINATIONS);
        let output = expander.process([&input, &input, &input, &input]);
        for (index, destination) in ["SL", "SR", "BL", "BR", "TFL", "TFR", "TBL", "TBR"]
            .iter()
            .enumerate()
        {
            let expected = ambient_expander_fir(SR, destination)
                .expect("canonical destination")
                .process(&input);
            assert_eq!(output[index], expected, "{destination}");
        }
        assert_ne!(output[0], output[2]);
        assert_ne!(output[4], output[6]);
        let energy = |signal: &[f64]| {
            signal[1440..]
                .iter()
                .map(|sample| sample * sample)
                .sum::<f64>()
        };
        let mut correlation = 0.0;
        let mut pairs = 0;
        for left in 0..output.len() {
            for right in left + 1..output.len() {
                let dot = output[left][1440..]
                    .iter()
                    .zip(&output[right][1440..])
                    .map(|(a, b)| a * b)
                    .sum::<f64>();
                correlation +=
                    (dot / (energy(&output[left]) * energy(&output[right])).sqrt()).abs();
                pairs += 1;
            }
        }
        assert!(correlation / (pairs as f64) < 0.1);
    }

    #[test]
    fn layout_expander_keeps_filters_attached_to_destination_labels() {
        let inputs = [
            noise(101, 4096),
            noise(102, 4096),
            noise(103, 4096),
            noise(104, 4096),
        ];
        let canonical = ["SL", "SR", "BL", "BR", "TFL", "TFR", "TBL", "TBR"];
        let permuted = ["TBR", "SL", "TFL", "BR", "SR", "TBL", "BL", "TFR"];
        let refs = [&inputs[0][..], &inputs[1], &inputs[2], &inputs[3]];

        let mut first = FixedAmbientExpander::new(SR, &canonical);
        let mut second = FixedAmbientExpander::new(SR, &permuted);
        let first_output = first.process(refs);
        let second_output = second.process(refs);
        for name in canonical {
            let first_index = first.destinations().position(|destination| destination == name);
            let second_index = second.destinations().position(|destination| destination == name);
            assert_eq!(first_output[first_index.unwrap()], second_output[second_index.unwrap()]);
        }
    }

    #[test]
    fn layout_expander_only_uses_the_matching_side_and_zone_input() {
        let inputs = [
            noise(111, 4096),
            noise(112, 4096),
            noise(113, 4096),
            noise(114, 4096),
        ];
        let destinations = ["SR", "TBL", "SL", "TFR"];
        let mut expander = FixedAmbientExpander::new(SR, &destinations);
        let output = expander.process([&inputs[0], &inputs[1], &inputs[2], &inputs[3]]);
        for (index, destination) in destinations.into_iter().enumerate() {
            let source = match destination {
                "SL" | "BL" => &inputs[0],
                "SR" | "BR" => &inputs[1],
                "TFL" | "TBL" => &inputs[2],
                "TFR" | "TBR" => &inputs[3],
                _ => unreachable!(),
            };
            let expected = ambient_expander_fir(SR, destination).unwrap().process(source);
            assert_eq!(output[index], expected, "{destination}");
        }
    }

    #[test]
    fn layout_expander_supports_absent_zones_ragged_blocks_and_seek() {
        let inputs = [
            noise(121, 4096),
            noise(122, 4096),
            noise(123, 4096),
            noise(124, 4096),
        ];
        let layout = ["FL", "FR", "C", "LFE", "SL", "SR", "TFL", "TFR"];
        let active = ["SL", "SR", "TFL", "TFR"];
        let refs = [&inputs[0][..], &inputs[1], &inputs[2], &inputs[3]];
        let mut whole = FixedAmbientExpander::new(SR, &layout);
        let expected = whole.process(refs);

        let mut blocked = FixedAmbientExpander::new(SR, &layout);
        let mut actual = vec![Vec::new(); active.len()];
        for start in (0..inputs[0].len()).step_by(127) {
            let end = (start + 127).min(inputs[0].len());
            let mut outputs = vec![vec![0.0; end - start]; active.len()];
            blocked.process_into(
                [
                    &inputs[0][start..end],
                    &inputs[1][start..end],
                    &inputs[2][start..end],
                    &inputs[3][start..end],
                ],
                &mut outputs,
            );
            for (all, block) in actual.iter_mut().zip(outputs) {
                all.extend(block);
            }
        }
        assert_eq!(blocked.destinations().collect::<Vec<_>>(), active);
        assert_eq!(actual, expected);

        blocked.reset();
        assert_eq!(blocked.process(refs), expected);
        blocked.seek(0);
        assert_eq!(blocked.process(refs), expected);
    }

    #[test]
    fn layout_expander_covers_supported_layout_destination_sets() {
        let cases: &[(&[&str], &[&str])] = &[
            (&["FL", "FR"], &[]),
            (&["FL", "FR", "C", "LFE", "SL", "SR"], &["SL", "SR"]),
            (
                &["FL", "FR", "C", "LFE", "SL", "SR", "BL", "BR"],
                &["SL", "SR", "BL", "BR"],
            ),
            (
                &["FL", "FR", "C", "LFE", "SL", "SR", "TFL", "TFR"],
                &["SL", "SR", "TFL", "TFR"],
            ),
            (
                &[
                    "FL", "FR", "C", "LFE", "SL", "SR", "TFL", "TFR", "TBL", "TBR",
                ],
                &["SL", "SR", "TFL", "TFR", "TBL", "TBR"],
            ),
            (
                &[
                    "FL", "FR", "C", "LFE", "SL", "SR", "BL", "BR", "TFL", "TFR",
                ],
                &["SL", "SR", "BL", "BR", "TFL", "TFR"],
            ),
            (
                &[
                    "FL", "FR", "C", "LFE", "SL", "SR", "BL", "BR", "TFL", "TFR",
                    "TBL", "TBR",
                ],
                &ALL_DESTINATIONS,
            ),
        ];
        for (layout, active) in cases {
            let expander = FixedAmbientExpander::new(SR, layout);
            assert_eq!(expander.destinations().collect::<Vec<_>>(), *active, "{layout:?}");
        }
    }

    #[test]
    fn layout_expander_keeps_state_for_surviving_destinations_on_layout_change() {
        let first = [noise(131, 512), noise(132, 512), noise(133, 512), noise(134, 512)];
        let second = [noise(135, 512), noise(136, 512), noise(137, 512), noise(138, 512)];
        let mut changed = FixedAmbientExpander::new(SR, &["SL"]);
        changed.process([&first[0], &first[1], &first[2], &first[3]]);
        changed.set_destinations(&["TFR", "SL"]);
        let actual = changed.process([&second[0], &second[1], &second[2], &second[3]]);

        let fir = ambient_expander_fir(SR, "SL").unwrap();
        let mut expected_line = upmixer_dsp_core::routing::decorrelate::VelvetLine::new(&fir);
        let mut first_block = first[0].clone();
        expected_line.process(&mut first_block);
        let mut second_block = second[0].clone();
        expected_line.process(&mut second_block);
        assert_eq!(actual[1], second_block);
    }
}

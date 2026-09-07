//! Focused streaming coverage for revisioned ambient routing.

use std::sync::Arc;
use upmixer_dsp_core::kernels::rng::next_unit;
use upmixer_dsp_core::stream::engine::{PreviewEngine, StemSource};
use upmixer_dsp_core::stream::params::{EngineParams, SendShape};
use upmixer_dsp_core::stream::routing::{StemRouteState, AMBIENT_SURROUND};

const SR: u32 = 48_000;
const N: usize = 24_000;
const CHANNELS: usize = 6;

/// A stem shaped like a separated one: a centred note with a decorrelated
/// tail behind it.
fn stem() -> Arc<StemSource> {
    let mut left = vec![0.0f32; N];
    let mut right = vec![0.0f32; N];
    let (mut seed_l, mut seed_r) = (11u64, 12u64);
    for i in 0..N {
        let t = i as f64 / SR as f64;
        let note = 0.3 * (2.0 * std::f64::consts::PI * 440.0 * t).sin();
        let tail = 0.15 * (-1.5 * t).exp();
        left[i] = (note + tail * (next_unit(&mut seed_l) * 2.0 - 1.0)) as f32;
        right[i] = (note + tail * (next_unit(&mut seed_r) * 2.0 - 1.0)) as f32;
    }
    Arc::new(StemSource { left, right })
}

/// A 5.1.2 bed whose stem is routed to the fronts only, so anything that
/// reaches SL/SR or the heights got there through an ambient send.
fn engine(rear: f64, height: f64, downmix_lock: bool) -> PreviewEngine {
    let params: EngineParams = serde_json::from_str(&format!(
        r#"{{
            "speakers": [
                {{"name": "FL", "azimuth_rad": 0.5236, "elevation_rad": 0.0, "group_gain": 1.0}},
                {{"name": "FR", "azimuth_rad": -0.5236, "elevation_rad": 0.0, "group_gain": 1.0}},
                {{"name": "SL", "azimuth_rad": 1.9199, "elevation_rad": 0.0, "group_gain": 1.0}},
                {{"name": "SR", "azimuth_rad": -1.9199, "elevation_rad": 0.0, "group_gain": 1.0}},
                {{"name": "TFL", "azimuth_rad": 0.7854, "elevation_rad": 0.7854, "group_gain": 1.0}},
                {{"name": "TFR", "azimuth_rad": -0.7854, "elevation_rad": 0.7854, "group_gain": 1.0}}
            ],
            "shapes": ["left", "right", "surround_left", "surround_right",
                       "height_left", "height_right"],
            "surround_downmix_coeff": 0.7071067811865476,
            "height_downmix_coeff": 0.7071067811865476,
            "spatial_downmix_lock": {downmix_lock},
            "sends": {{"surround_bass_cutoff_hz": 250.0,
                      "height_low_rolloff_hz": 150.0, "height_low_rolloff_gain": 0.15,
                      "height_crossover_hz": 3000.0, "height_high_shelf_gain": 1.5,
                      "height_directional_band_hz": 8000.0,
                      "height_directional_band_gain": 1.0,
                      "lfe_cutoff_hz": 120.0, "lfe_filter_order": 4, "lfe_gain": 1.0}},
            "stems": [{{"routing": [["FL", 1.0], ["FR", 1.0]], "enabled": true,
                       "ambient_rear": {rear}, "ambient_height": {height}}}],
            "master": {{}},
            "output_mode": "native",
            "bypass_mastering": true,
            "soft_limit_threshold": 0.0
        }}"#
    ))
    .expect("engine parameters");
    PreviewEngine::new(SR, params, vec![stem()])
}

fn render(rear: f64, height: f64) -> Vec<Vec<f64>> {
    let mut engine = engine(rear, height, false);
    let mut out = vec![0.0; CHANNELS * N];
    let emitted = engine.render(&mut out, N);
    (0..CHANNELS)
        .map(|ch| out[ch * N..ch * N + emitted].to_vec())
        .collect()
}

fn energy(signal: &[f64]) -> f64 {
    signal.iter().map(|v| v * v).sum()
}

fn render_block(engine: &mut PreviewEngine, frames: usize) -> Vec<Vec<f64>> {
    let mut out = vec![0.0; CHANNELS * frames];
    let emitted = engine.render(&mut out, frames);
    (0..CHANNELS)
        .map(|channel| out[channel * frames..channel * frames + emitted].to_vec())
        .collect()
}

#[test]
fn a_zero_slider_leaves_the_surrounds_and_heights_silent() {
    let bed = render(0.0, 0.0);
    for channel in 2..CHANNELS {
        assert_eq!(
            energy(&bed[channel]),
            0.0,
            "channel {channel} is not silent"
        );
    }
}

#[test]
fn an_ambient_send_reaches_speakers_the_stem_is_not_routed_to() {
    let bed = render(0.8, 0.8);
    for channel in 2..CHANNELS {
        assert!(
            energy(&bed[channel]) > 0.0,
            "channel {channel} got no ambient"
        );
    }
}

#[test]
fn what_the_sends_take_comes_out_of_the_front() {
    let dry = render(0.0, 0.0);
    let sent = render(0.9, 0.9);
    let front_dry = energy(&dry[0]) + energy(&dry[1]);
    let front_sent = energy(&sent[0]) + energy(&sent[1]);
    assert!(
        front_sent < front_dry,
        "front kept {front_sent:.4} of {front_dry:.4} — the sends are a copy, not a move"
    );
}

#[test]
fn the_heights_get_the_brighter_half_of_the_ambient() {
    let bed = render(0.8, 0.8);
    // Zero crossings stand in for a centroid: the height feed carries the
    // top of the tilt plus its own elevation shelf.
    let crossings = |signal: &[f64]| {
        signal
            .windows(2)
            .filter(|w| w[0].signum() != w[1].signum())
            .count()
    };
    assert!(
        crossings(&bed[4]) > crossings(&bed[2]),
        "height send is not brighter than the rear send"
    );
}

#[test]
fn overlapping_wet_sends_keep_rear_and_height_from_the_same_broadband_feed() {
    let bed = render(0.8, 0.8);
    assert!(energy(&bed[2]) > 0.0, "rear feed is silent");
    assert!(energy(&bed[4]) > 0.0, "height feed is silent");
    let rear_high = bed[2]
        .windows(2)
        .map(|w| (w[1] - w[0]) * (w[1] - w[0]))
        .sum::<f64>();
    assert!(rear_high > 0.0, "rear feed has no high-frequency texture");
}

#[test]
fn the_send_scales_with_its_slider() {
    let half = render(0.4, 0.0);
    let full = render(0.8, 0.0);
    let ratio = energy(&full[2]) / energy(&half[2]);
    assert!(
        (ratio - 4.0).abs() < 0.2,
        "doubling the slider scaled power by {ratio:.3}"
    );
}

#[test]
fn downmix_lock_restores_the_streaming_stem_pair() {
    let source = stem();
    let mut engine = engine(0.8, 0.8, true);
    let mut out = vec![0.0; CHANNELS * N];
    let emitted = engine.render(&mut out, N);

    for i in 0..emitted {
        let left = out[i] + 0.7071067811865476 * (out[2 * N + i] + out[4 * N + i]);
        let right = out[N + i] + 0.7071067811865476 * (out[3 * N + i] + out[5 * N + i]);
        assert!((left - source.left[i] as f64).abs() < 1e-9, "left at {i}");
        assert!(
            (right - source.right[i] as f64).abs() < 1e-9,
            "right at {i}"
        );
    }
}

#[test]
fn ambient_distribution_counts_left_and_right_destinations_separately() {
    let mut params = engine(0.0, 0.0, false).params().clone();
    params.shapes = vec![
        SendShape::SurroundLeft,
        SendShape::SurroundLeft,
        SendShape::SurroundRight,
        SendShape::HeightRight,
    ];
    let left_share = params.ambient_side_share(SendShape::SurroundLeft);
    let right_share = params.ambient_side_share(SendShape::SurroundRight);
    assert!((left_share - 2.0_f64.sqrt().recip()).abs() < 1e-15);
    assert_eq!(right_share, 1.0);
    assert_eq!(params.ambient_side_share(SendShape::HeightLeft), 0.0);
}

#[test]
fn overlapping_wet_sends_fade_through_zero_and_back() {
    let mut engine = engine(0.8, 0.8, false);
    let _ = render_block(&mut engine, 4096);

    let mut params = engine.params().clone();
    params.stems[0].ambient_rear = 0.0;
    params.stems[0].ambient_height = 0.0;
    engine.update_params(params);
    let faded = render_block(&mut engine, 4096);
    let first = energy(&faded[2][..512]);
    let last = energy(&faded[2][faded[2].len() - 512..]);
    assert!(
        last < first,
        "rear send did not fade toward zero: {first} -> {last}"
    );

    let mut params = engine.params().clone();
    params.stems[0].ambient_rear = 0.8;
    params.stems[0].ambient_height = 0.8;
    engine.update_params(params);
    let restored = render_block(&mut engine, 4096);
    let first = energy(&restored[2][..512]);
    let last = energy(&restored[2][restored[2].len() - 512..]);
    assert!(
        last > first,
        "rear send did not fade back in: {first} -> {last}"
    );
}

#[test]
fn an_asymmetric_layout_does_not_subtract_an_absent_right_send() {
    let params = engine(0.8, 0.0, false).params().clone();
    let mut route = StemRouteState::new(SR, &params.sends, None, None, None);
    route.set_ambient(SR, &params.sends, true, true, 2000.0, 2000.0, 0.0, 0.0);
    let mut left_seed = 61;
    let mut right_seed = 62;
    let left: Vec<f32> = (0..N)
        .map(|_| (next_unit(&mut left_seed) * 2.0 - 1.0) as f32)
        .collect();
    let right: Vec<f32> = (0..N)
        .map(|_| (next_unit(&mut right_seed) * 2.0 - 1.0) as f32)
        .collect();
    route.process_block(
        &left,
        &right,
        0,
        N,
        [0.8, 0.0],
        [0.0, 0.0],
        [0.0, 0.0],
        false,
        false,
    );
    let worst = route
        .signal(1)
        .iter()
        .zip(&right)
        .map(|(actual, source)| (actual - f64::from(*source)).abs())
        .fold(0.0, f64::max);
    assert!(
        worst < 1e-12,
        "right anchor changed without a right destination: {worst:e}"
    );
}

#[test]
fn height_texture_can_run_without_an_ambient_split() {
    let mut params = engine(0.0, 0.0, false).params().clone();
    params.stems[0].height_texture = 0.2;
    let mut engine = PreviewEngine::new(SR, params, vec![stem()]);
    let mut out = vec![0.0; CHANNELS * N];
    let emitted = engine.render(&mut out, N);
    assert!(energy(&out[2 * N..3 * N]) < 1e-20);
    assert!(energy(&out[3 * N..4 * N]) < 1e-20);
    assert!(energy(&out[4 * N..4 * N + emitted]) > 0.0);
    assert!(energy(&out[5 * N..5 * N + emitted]) > 0.0);
}

#[test]
fn height_texture_edits_fade_without_a_click() {
    let mut params = engine(0.0, 0.0, false).params().clone();
    params.stems[0].height_texture = 0.2;
    let mut engine = PreviewEngine::new(SR, params, vec![stem()]);
    let _ = render_block(&mut engine, 2048);

    let mut params = engine.params().clone();
    params.stems[0].height_texture = 0.0;
    engine.update_params(params);
    let faded = render_block(&mut engine, 4096);
    let first = energy(&faded[4][..512]);
    let last = energy(&faded[4][faded[4].len() - 512..]);
    assert!(
        last < first,
        "texture did not fade toward zero: {first} -> {last}"
    );
}

#[test]
fn texture_only_seek_keeps_the_configured_height_route() {
    let mut params = engine(0.0, 0.0, false).params().clone();
    params.stems[0].height_texture = 0.2;
    let mut engine = PreviewEngine::new(SR, params, vec![stem()]);
    let first = render_block(&mut engine, 1024);

    engine.seek(0);
    let replay = render_block(&mut engine, 1024);
    assert!(energy(&replay[4]) > 0.0, "texture disappeared after seek");
    for (expected, actual) in first[4].iter().zip(&replay[4]) {
        assert!(
            (expected - actual).abs() < 1e-12,
            "height texture changed after seek"
        );
    }
}

#[test]
fn live_texture_enable_ramps_up_from_zero() {
    let params = engine(0.0, 0.0, false).params().clone();
    let mut engine = PreviewEngine::new(SR, params, vec![stem()]);
    let _ = render_block(&mut engine, 2048);

    let mut params = engine.params().clone();
    params.stems[0].height_texture = 0.2;
    engine.update_params(params);
    let enabled = render_block(&mut engine, 4096);
    let first = energy(&enabled[4][..512]);
    let last = energy(&enabled[4][enabled[4].len() - 512..]);
    assert!(
        first < last * 0.5,
        "live texture enable stepped in: {first} -> {last}"
    );
}

fn routed_with_trim(trim_db: f64) -> (Vec<f64>, Vec<f64>) {
    let params = engine(0.0, 0.0, false).params().clone();
    let mut route = StemRouteState::new(SR, &params.sends, None, None, None);
    route.set_ambient(SR, &params.sends, true, true, 2000.0, 2000.0, trim_db, 0.0);
    let mut left_seed = 71;
    let mut right_seed = 72;
    let left: Vec<f32> = (0..N)
        .map(|_| (next_unit(&mut left_seed) * 2.0 - 1.0) as f32)
        .collect();
    let right: Vec<f32> = (0..N)
        .map(|_| (next_unit(&mut right_seed) * 2.0 - 1.0) as f32)
        .collect();
    route.process_block(
        &left,
        &right,
        0,
        N,
        [0.6, 0.6],
        [0.0, 0.0],
        [0.0, 0.0],
        false,
        false,
    );
    (
        route.signal(0).to_vec(),
        route.signal(AMBIENT_SURROUND).to_vec(),
    )
}

#[test]
fn ambient_trim_boosts_wet_only_and_leaves_the_anchor_unchanged() {
    let (direct, wet) = routed_with_trim(0.0);
    let (trimmed_direct, boosted) = routed_with_trim(6.0);
    let direct_error = direct
        .iter()
        .zip(&trimmed_direct)
        .map(|(a, b)| (a - b).abs())
        .fold(0.0, f64::max);
    assert!(
        direct_error < 1e-12,
        "trim changed direct residual: {direct_error:e}"
    );
    let ratio = energy(&boosted) / energy(&wet);
    assert!(
        (ratio - 4.0).abs() < 0.05,
        "+6 dB wet trim power ratio {ratio:.4}"
    );
}

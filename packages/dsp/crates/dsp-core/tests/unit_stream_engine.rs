mod engine {
    use std::sync::Arc;
    use upmixer_dsp_core::stream::engine::*;
    use upmixer_dsp_core::stream::params::EngineParams;

    fn engine(mute_lfe: bool) -> PreviewEngine {
        let params: EngineParams = serde_json::from_str(&format!(
            r#"{{
                "speakers": [
                    {{"name": "FL", "azimuth_rad": 0.5236, "elevation_rad": 0.0, "group_gain": 1.0}},
                    {{"name": "FR", "azimuth_rad": -0.5236, "elevation_rad": 0.0, "group_gain": 1.0}},
                    {{"name": "LFE", "azimuth_rad": 0.0, "elevation_rad": 0.0,
                     "group_gain": 1.0, "muted": {mute_lfe}}}
                ],
                "lfe_index": 2,
                "shapes": ["left", "right", "mono"],
                "surround_downmix_coeff": 0.7071067811865476,
                "height_downmix_coeff": 0.7071067811865476,
                "sends": {{"surround_bass_cutoff_hz": 250.0,
                          "height_low_rolloff_hz": 150.0, "height_low_rolloff_gain": 0.15,
                          "height_crossover_hz": 3000.0, "height_high_shelf_gain": 1.5,
                          "height_directional_band_hz": 8000.0,
                          "height_directional_band_gain": 1.0,
                          "lfe_cutoff_hz": 120.0, "lfe_filter_order": 4, "lfe_gain": 1.0}},
                "stems": [{{"routing": [["FL", 0.9], ["FR", 0.9], ["LFE", 1.0]], "rebalance_db": 0.0,
                           "enabled": true, "eq_fir": [], "route_scale": 1.0}}],
                "master": {{}},
                "output_mode": "native",
                "bypass_mastering": true,
                "soft_limit_threshold": 0.0
            }}"#
        ))
        .expect("engine parameters");

        let tone: Vec<f32> = (0..4096)
            .map(|i| (0.4 * (2.0 * std::f64::consts::PI * 60.0 * i as f64 / 48_000.0).sin()) as f32)
            .collect();
        PreviewEngine::new(
            48_000,
            params,
            vec![Arc::new(StemSource {
                left: tone.clone(),
                right: tone,
            })],
        )
    }

    fn probe_engine(mute_fl: bool) -> PreviewEngine {
        let params: EngineParams = serde_json::from_str(&format!(
            r#"{{
                "speakers": [
                    {{"name": "FL", "azimuth_rad": 0.5236, "elevation_rad": 0.0,
                     "group_gain": 1.0, "muted": {mute_fl}}},
                    {{"name": "FR", "azimuth_rad": -0.5236, "elevation_rad": 0.0, "group_gain": 1.0}},
                    {{"name": "SL", "azimuth_rad": 1.9, "elevation_rad": 0.0, "group_gain": 1.0}},
                    {{"name": "SR", "azimuth_rad": -1.9, "elevation_rad": 0.0, "group_gain": 1.0}},
                    {{"name": "LFE", "azimuth_rad": 0.0, "elevation_rad": 0.0, "group_gain": 1.0}}
                ],
                "lfe_index": 4,
                "shapes": ["left", "right", "left", "right", "mono"],
                "surround_downmix_coeff": 0.7071067811865476,
                "height_downmix_coeff": 0.7071067811865476,
                "sends": {{"surround_bass_cutoff_hz": 250.0,
                          "height_low_rolloff_hz": 150.0, "height_low_rolloff_gain": 0.15,
                          "height_crossover_hz": 3000.0, "height_high_shelf_gain": 1.5,
                          "height_directional_band_hz": 8000.0,
                          "height_directional_band_gain": 1.0,
                          "lfe_cutoff_hz": 120.0, "lfe_filter_order": 4, "lfe_gain": 1.0}},
                "stems": [{{"routing": [["FL", 0.7], ["FR", 0.7], ["SL", 0.5], ["SR", 0.5], ["LFE", 0.5]],
                           "rebalance_db": 0.0, "enabled": true, "eq_fir": [], "route_scale": 1.0}}],
                "master": {{"bass": {{"sub_gain_db": 0.0, "mid_gain_db": 0.0, "unify_hz": 120.0,
                            "punch": 0.0, "excite": false, "lfe_gain_db": 0.0,
                            "sub_cutoff_hz": 60.0, "mid_cutoff_hz": 200.0,
                            "excite_blend": 0.0, "excite_drive": 0.0,
                            "punch_fast_ms": 5.0, "punch_slow_ms": 50.0, "punch_max_db": 0.0,
                            "decorrelate": 0.0, "decorr_low_hz": 60.0, "decorr_high_hz": 200.0,
                            "decorr_sections": 1, "decorr_max_delay_ms": 5.0,
                            "decorr_fast_ms": 5.0, "decorr_slow_ms": 50.0}},
                          "lf_targets": [[0, 0.5], [1, 0.5]]}},
                "output_mode": "native",
                "soft_limit_threshold": 0.0
            }}"#,
        ))
        .expect("engine parameters");

        let tone: Vec<f32> = (0..8192)
            .map(|i| (0.4 * (2.0 * std::f64::consts::PI * 80.0 * i as f64 / 48_000.0).sin()) as f32)
            .collect();
        PreviewEngine::new(
            48_000,
            params,
            vec![Arc::new(StemSource {
                left: tone.clone(),
                right: tone,
            })],
        )
    }

    /// Speaker mute is a monitor control, so it has to be silent *and* inert:
    /// carrying it as a routing gain took the channel out of the shared bass
    /// pool and the linked compressor's detector, so muting `FL` quietly
    /// changed every other speaker's low end.
    #[test]
    fn muting_a_speaker_silences_it_without_touching_any_other_channel() {
        let mut muted_out = vec![0.0; 5 * 8192];
        probe_engine(true).render(&mut muted_out, 8192);
        let mut open_out = vec![0.0; 5 * 8192];
        probe_engine(false).render(&mut open_out, 8192);

        assert!(
            muted_out[0..8192].iter().all(|v| *v == 0.0),
            "muted FL should stay silent through the LF fold",
        );
        for channel in 1..5 {
            let span = channel * 8192..(channel + 1) * 8192;
            assert_eq!(
                muted_out[span.clone()],
                open_out[span],
                "channel {channel} changed when FL was muted",
            );
        }
    }

    #[test]
    fn muting_the_lfe_speaker_silences_its_bus() {
        let mut muted = engine(true);
        let mut out = vec![0.0; 3 * 4096];
        muted.render(&mut out, 4096);
        let lfe = &out[2 * 4096..3 * 4096];
        assert!(
            lfe.iter().all(|v| *v == 0.0),
            "muted LFE bus should be silent"
        );

        let mut unmuted = engine(false);
        let mut out = vec![0.0; 3 * 4096];
        unmuted.render(&mut out, 4096);
        let lfe = &out[2 * 4096..3 * 4096];
        assert!(
            lfe.iter().any(|v| v.abs() > 1e-6),
            "unmuted LFE bus should carry signal"
        );
    }
}

mod movement_transport {
    //! Movement schedules are evaluated against absolute programme time by the
    //! streaming engine, independent of render quantum and output sample rate.

    use std::sync::Arc;

    use upmixer_dsp_core::movement::{
        MovementEvent, MovementSchedule, MovementStemSchedule, INTERPOLATION_US,
    };
    use upmixer_dsp_core::stream::engine::{PreviewEngine, StemSource};
    use upmixer_dsp_core::stream::params::EngineParams;

    const CANONICAL_RATE: u32 = 48_000;
    const PROGRAMME_FRAMES: usize = CANONICAL_RATE as usize;

    fn schedule() -> MovementSchedule {
        let event = |time_us: i64, gains: Vec<f64>| MovementEvent {
            time_us,
            position: [0.0; 3],
            gains,
            right_position: None,
            right_gains: None,
            interpolation_us: INTERPOLATION_US,
        };
        MovementSchedule {
            version: 1,
            revision: 17,
            sample_rate: CANONICAL_RATE,
            duration_frames: PROGRAMME_FRAMES,
            grid_us: 20_000,
            interpolation_us: INTERPOLATION_US,
            stems: vec![MovementStemSchedule {
                stem_key: "moving-bed".to_string(),
                stem_index: 0,
                events: vec![event(0, vec![1.0, 0.0]), event(500_000, vec![0.0, 1.0])],
            }],
        }
    }

    fn params(schedule: MovementSchedule, compressor: bool) -> EngineParams {
        let mut params: EngineParams = serde_json::from_str(
            r#"{
                "speakers": [
                    {"name": "FL", "azimuth_rad": 0.5235987756, "elevation_rad": 0.0, "group_gain": 1.0},
                    {"name": "FR", "azimuth_rad": -0.5235987756, "elevation_rad": 0.0, "group_gain": 1.0}
                ],
                "lfe_index": null,
                "shapes": ["left", "right"],
                "surround_downmix_coeff": 0.7071067811865476,
                "height_downmix_coeff": 0.7071067811865476,
                "sends": {
                    "surround_bass_cutoff_hz": 250.0,
                    "height_low_rolloff_hz": 150.0,
                    "height_low_rolloff_gain": 0.15,
                    "height_crossover_hz": 3000.0,
                    "height_high_shelf_gain": 1.5,
                    "height_directional_band_hz": 8000.0,
                    "height_directional_band_gain": 1.0,
                    "lfe_cutoff_hz": 120.0,
                    "lfe_filter_order": 4,
                    "lfe_gain": 1.0
                },
                "stems": [{
                    "routing": [["FL", 1.0], ["FR", 1.0]],
                    "enabled": true,
                    "eq_fir": [],
                    "route_scale": 1.0
                }],
                "master": {},
                "output_mode": "native",
                "bypass_mastering": true,
                "soft_limit_threshold": 0.0
            }"#,
        )
        .expect("engine parameters");
        if compressor {
            params.bypass_mastering = false;
            params.master.compressor = Some(upmixer_dsp_core::mastering::compressor::CompParams {
                threshold_db: -24.0,
                ratio: 2.0,
                attack_ms: 5.0,
                release_ms: 80.0,
                knee_db: 0.0,
                makeup_db: 0.0,
                sidechain_hpf_hz: None,
            });
        }
        params.movement_schedule = Some(Arc::new(schedule));
        params.validate_movement().expect("movement parameters");
        params
    }

    fn engine(rate: u32, schedule: MovementSchedule, compressor: bool) -> PreviewEngine {
        let frames = rate as usize;
        PreviewEngine::new(
            rate,
            params(schedule, compressor),
            vec![Arc::new(StemSource {
                left: vec![1.0; frames],
                right: vec![1.0; frames],
            })],
        )
    }

    fn render_pattern(rate: u32, schedule: MovementSchedule, pattern: &[usize]) -> [Vec<f64>; 2] {
        let frames = rate as usize;
        let mut engine = engine(rate, schedule, false);
        let output_capacity = frames;
        let mut output = [
            Vec::with_capacity(output_capacity),
            Vec::with_capacity(output_capacity),
        ];
        let mut rendered = 0;
        let mut pattern_index = 0;
        while rendered < frames {
            let requested = pattern[pattern_index % pattern.len()].min(frames - rendered);
            let mut block = vec![0.0; requested * 2];
            assert_eq!(engine.render(&mut block, requested), requested);
            output[0].extend_from_slice(&block[..requested]);
            output[1].extend_from_slice(&block[requested..]);
            rendered += requested;
            pattern_index += 1;
        }
        output
    }

    #[test]
    fn schedule_is_partition_invariant_and_absolute_across_output_rates() {
        for rate in [44_100, CANONICAL_RATE, 96_000] {
            let contiguous = render_pattern(rate, schedule(), &[rate as usize]);
            let varied = render_pattern(rate, schedule(), &[127, 257, 64, 1024, 31]);
            assert_eq!(contiguous[0].len(), rate as usize);
            assert_eq!(contiguous[1].len(), rate as usize);
            for channel in 0..2 {
                for (frame, (one, many)) in
                    contiguous[channel].iter().zip(&varied[channel]).enumerate()
                {
                    assert!(
                        (one - many).abs() < 1e-12,
                        "rate {rate} channel {channel} frame {frame}: {one} != {many}"
                    );
                }
            }

            let route = schedule().stems[0].clone();
            for frame in 0..rate as usize {
                let time_us = frame as f64 * 1_000_000.0 / rate as f64;
                let mut expected = [0.0; 2];
                route.sample_into_at(time_us, false, &mut expected);
                assert!(
                    (contiguous[0][frame] - expected[0]).abs() < 1e-12,
                    "rate {rate} FL frame {frame}: {} != {}",
                    contiguous[0][frame],
                    expected[0]
                );
                assert!(
                    (contiguous[1][frame] - expected[1]).abs() < 1e-12,
                    "rate {rate} FR frame {frame}: {} != {}",
                    contiguous[1][frame],
                    expected[1]
                );
            }
        }
    }

    #[test]
    fn seek_preroll_matches_continuous_schedule_render() {
        for rate in [44_100, CANONICAL_RATE, 96_000] {
            let target = (rate as f64 * 0.82) as usize;
            let span = (rate as f64 * 0.1) as usize;
            let mut continuous = engine(rate, schedule(), true);
            let mut prefix = vec![0.0; (target + span) * 2];
            assert_eq!(continuous.render(&mut prefix, target + span), target + span);

            let mut seeked = engine(rate, schedule(), true);
            seeked.seek(target);
            let mut actual = vec![0.0; span * 2];
            assert_eq!(seeked.render(&mut actual, span), span);

            for channel in 0..2 {
                let expected = &prefix
                    [channel * (target + span) + target..channel * (target + span) + target + span];
                let got = &actual[channel * span..(channel + 1) * span];
                for (frame, (want, have)) in expected.iter().zip(got).enumerate() {
                    assert!(
                        (want - have).abs() < 1e-7,
                        "rate {rate} channel {channel} frame {frame}: {want} != {have}"
                    );
                }
            }
        }
    }

    #[test]
    fn invalid_atomic_movement_update_leaves_the_live_pair_untouched() {
        let mut live = engine(CANONICAL_RATE, schedule(), false);
        let original_revision = live
            .params()
            .movement_schedule
            .as_ref()
            .map(|value| value.revision);
        let mut replacement = params(schedule(), false);
        replacement.master.output_gain = 0.25;
        let mut invalid = schedule();
        invalid.stems[0].stem_index = 1;

        assert!(live
            .update_params_with_movement(replacement, Some(invalid))
            .is_err());
        assert_eq!(live.position(), 0);
        assert_eq!(
            live.params()
                .movement_schedule
                .as_ref()
                .map(|value| value.revision),
            original_revision
        );
        assert_eq!(live.params().master.output_gain, 1.0);
    }
}

mod master_meters {
    use std::sync::Arc;
    use upmixer_dsp_core::loudness::measure_loudness_stats;
    use upmixer_dsp_core::stream::engine::*;
    use upmixer_dsp_core::stream::params::{EngineParams, OutputMode};

    /// A hot stereo programme through a compressor and a limiter, so both
    /// gain-reduction taps have something to report.
    fn engine(frames: usize, level: f64) -> PreviewEngine {
        let params: EngineParams = serde_json::from_str(
            r#"{
                "speakers": [
                    {"name": "FL", "azimuth_rad": 0.5236, "elevation_rad": 0.0, "group_gain": 1.0},
                    {"name": "FR", "azimuth_rad": -0.5236, "elevation_rad": 0.0, "group_gain": 1.0},
                    {"name": "LFE", "azimuth_rad": 0.0, "elevation_rad": 0.0, "group_gain": 1.0}
                ],
                "lfe_index": 2,
                "shapes": ["left", "right", "mono"],
                "surround_downmix_coeff": 0.7071067811865476,
                "height_downmix_coeff": 0.7071067811865476,
                "sends": {"surround_bass_cutoff_hz": 250.0,
                          "height_low_rolloff_hz": 150.0, "height_low_rolloff_gain": 0.15,
                          "height_crossover_hz": 3000.0, "height_high_shelf_gain": 1.5,
                          "height_directional_band_hz": 8000.0,
                          "height_directional_band_gain": 1.0,
                          "lfe_cutoff_hz": 120.0, "lfe_filter_order": 4, "lfe_gain": 3.0},
                "stems": [{"routing": [["FL", 1.0], ["FR", 1.0], ["LFE", 1.0]],
                           "rebalance_db": 0.0, "enabled": true, "eq_fir": [], "route_scale": 1.0}],
                "master": {"compressor": {"threshold_db": -30.0, "ratio": 4.0, "attack_ms": 5.0,
                                          "release_ms": 80.0, "knee_db": 6.0, "makeup_db": 12.0,
                                          "sidechain_hpf_hz": null},
                           "limiter": {"ceiling_dbtp": -1.0, "lookahead_ms": 1.5,
                                       "release_ms": 50.0, "safety_margin_db": 0.1}},
                "output_mode": "native",
                "meter_weights": [1.0, 1.0, 0.0],
                "soft_limit_threshold": 0.0
            }"#,
        )
        .expect("engine parameters");

        // Two tones: the 220 Hz one drives the mains, the 60 Hz one survives
        // the LFE bus's 120 Hz low-pass so the LFE curve has a peak of its own.
        let tone: Vec<f32> = (0..frames)
            .map(|i| {
                let t = i as f64 / 48_000.0;
                let half = level * 0.5;
                (half * (2.0 * std::f64::consts::PI * 220.0 * t).sin()
                    + half * (2.0 * std::f64::consts::PI * 60.0 * t).sin()) as f32
            })
            .collect();
        PreviewEngine::new(
            48_000,
            params,
            vec![Arc::new(StemSource {
                left: tone.clone(),
                right: tone,
            })],
        )
    }

    fn render(engine: &mut PreviewEngine, frames: usize) -> Vec<Vec<f64>> {
        let block = 128;
        let mut out = vec![0.0; 3 * block];
        let mut collected = vec![Vec::new(); 3];
        let mut done = 0;
        while done < frames {
            let written = engine.render(&mut out, block);
            if written == 0 {
                break;
            }
            for (channel, sink) in collected.iter_mut().enumerate() {
                sink.extend_from_slice(&out[channel * block..channel * block + written]);
            }
            done += written;
        }
        collected
    }

    /// The live windows read the programme that was actually emitted: after a
    /// whole render their short-term reading is the offline kit's last 3 s.
    #[test]
    fn the_loudness_windows_read_the_emitted_programme() {
        let mut engine = engine(240_000, 0.2);
        let out = render(&mut engine, 240_000);
        let tail = out[0].len().saturating_sub(3 * 48_000);
        let want =
            measure_loudness_stats(&[(1.0, &out[0][tail..]), (1.0, &out[1][tail..])], 48_000);
        let got = engine.meters().master.short_term_lkfs;
        assert!(
            (got - want.max_short_term_lkfs).abs() < 0.1,
            "short-term {got} vs offline {}",
            want.max_short_term_lkfs,
        );
    }

    /// Both gain-reduction taps report on a programme hot enough to work
    /// them, and a quiet one leaves every stage at rest.
    #[test]
    fn the_gain_reduction_taps_follow_the_stages() {
        let mut hot = engine(96_000, 0.9);
        render(&mut hot, 96_000);
        let master = hot.meters().master;
        assert!(
            master.comp_gr_db > 1.0,
            "compressor GR {}",
            master.comp_gr_db
        );
        assert!(
            master.limiter_gr_db > 0.0,
            "limiter GR {}",
            master.limiter_gr_db
        );
        assert!(
            master.limiter_lfe_gr_db > 0.0,
            "LFE GR {}",
            master.limiter_lfe_gr_db
        );

        let mut quiet = engine(96_000, 0.002);
        render(&mut quiet, 96_000);
        let master = quiet.meters().master;
        assert_eq!(master.comp_gr_db, 0.0);
        assert_eq!(master.limiter_gr_db, 0.0);
        assert_eq!(master.limiter_lfe_gr_db, 0.0);
        assert!(
            master.momentary_lkfs < -40.0,
            "momentary {}",
            master.momentary_lkfs
        );
    }

    #[test]
    fn output_gain_reaches_the_limiter_before_stereo_collapse() {
        let mut preview = engine(96_000, 0.9);
        let mut params = preview.params().clone();
        params.output_mode = OutputMode::Stereo;
        params.master.output_gain = 0.25;
        preview.update_params(params);

        let mut out = vec![0.0; 2 * 128];
        while preview.render(&mut out, 128) > 0 {}

        assert!(preview.meters().master.limiter_gr_db.abs() < 1e-12);
    }
}

mod measure {
    use std::sync::Arc;
    use upmixer_dsp_core::stream::engine::PreviewEngine;
    use upmixer_dsp_core::stream::engine::StemSource;
    use upmixer_dsp_core::stream::measure::*;
    use upmixer_dsp_core::stream::params::EngineParams;

    fn engine(frames: usize) -> PreviewEngine {
        let params: EngineParams = serde_json::from_str(
            r#"{
                "speakers": [
                    {"name": "FL", "azimuth_rad": 0.5236, "elevation_rad": 0.0, "group_gain": 1.0},
                    {"name": "FR", "azimuth_rad": -0.5236, "elevation_rad": 0.0, "group_gain": 1.0}
                ],
                "lfe_index": null,
                "shapes": ["left", "right"],
                "surround_downmix_coeff": 0.7071067811865476,
                "height_downmix_coeff": 0.7071067811865476,
                "sends": {"surround_bass_cutoff_hz": 250.0,
                          "height_low_rolloff_hz": 150.0, "height_low_rolloff_gain": 0.15,
                          "height_crossover_hz": 3000.0, "height_high_shelf_gain": 1.5,
                          "height_directional_band_hz": 8000.0,
                          "height_directional_band_gain": 1.0,
                          "lfe_cutoff_hz": 120.0, "lfe_filter_order": 4, "lfe_gain": 0.316},
                "stems": [{"routing": [["FL", 0.9], ["FR", 0.9]], "rebalance_db": 0.0,
                           "enabled": true, "eq_fir": [], "route_scale": 1.0}],
                "master": {"lf_targets": [[0, 0.5], [1, 0.5]]},
                "output_mode": "stereo",
                "soft_limit_threshold": 0.0
            }"#,
        )
        .expect("engine parameters");

        let tone: Vec<f32> = (0..frames)
            .map(|i| {
                let t = i as f64 / 48_000.0;
                (0.4 * (2.0 * std::f64::consts::PI * 220.0 * t).sin()
                    + 0.1 * (2.0 * std::f64::consts::PI * 3300.0 * t).sin()) as f32
            })
            .collect();
        PreviewEngine::new(
            48_000,
            params,
            vec![Arc::new(StemSource {
                left: tone.clone(),
                right: tone,
            })],
        )
    }

    #[test]
    fn slicing_a_measurement_matches_the_blocking_one() {
        let mut reference = engine(120_000);
        let want = reference.measure(&[1.0, 1.0]);

        for slice in [128usize, 1024, 9000] {
            let live = engine(120_000);
            let mut pass = MeasurementPass::new(&live, &[1.0, 1.0]);
            let mut result = None;
            let mut guard = 0;
            while result.is_none() {
                result = pass.advance(slice);
                guard += 1;
                assert!(guard < 100_000, "slice {slice} never finished");
            }
            let [lkfs, dbtp, _, _] = result.expect("measured");
            assert!(
                (lkfs - want.0).abs() < 1e-9,
                "slice {slice}: {lkfs} vs {want:?}"
            );
            assert!(
                (dbtp - want.1).abs() < 1e-9,
                "slice {slice}: {dbtp} vs {want:?}"
            );
        }
    }

    #[test]
    fn measuring_leaves_the_live_transport_alone() {
        let mut live = engine(48_000);
        let mut out = vec![0.0; 2 * 4096];
        live.render(&mut out, 4096);
        let before = live.position();

        let mut pass = MeasurementPass::new(&live, &[1.0, 1.0]);
        while pass.advance(4096).is_none() {}

        assert_eq!(live.position(), before);
        let mut next = vec![0.0; 2 * 4096];
        assert_eq!(live.render(&mut next, 4096), 4096);
    }

    #[test]
    fn measurement_ignores_the_live_output_gain() {
        let live = engine(48_000);
        let mut corrected = engine(48_000);
        let mut params = corrected.params().clone();
        params.master.output_gain = 0.25;
        corrected.update_params(params);

        let (plain_lkfs, plain_dbtp) = run(&mut MeasurementPass::new(&live, &[1.0, 1.0]), 4096);
        let (corrected_lkfs, corrected_dbtp) =
            run(&mut MeasurementPass::new(&corrected, &[1.0, 1.0]), 4096);

        assert!((plain_lkfs - corrected_lkfs).abs() < 1e-9);
        assert!((plain_dbtp - corrected_dbtp).abs() < 1e-9);
    }

    #[test]
    fn progress_climbs_to_one() {
        let live = engine(48_000);
        let mut pass = MeasurementPass::new(&live, &[1.0, 1.0]);
        assert_eq!(pass.progress(), 0.0);
        pass.advance(4096);
        let partial = pass.progress();
        assert!(partial > 0.0 && partial < 1.0, "{partial}");
        while pass.advance(4096).is_none() {}
        assert_eq!(pass.progress(), 1.0);
    }

    fn run(pass: &mut MeasurementPass, slice: usize) -> (f64, f64) {
        let mut result = None;
        let mut guard = 0;
        while result.is_none() {
            result = pass.advance(slice);
            guard += 1;
            assert!(guard < 100_000, "never finished");
        }
        let [lkfs, dbtp, _, _] = result.expect("measured");
        (lkfs, dbtp)
    }

    #[test]
    fn an_excerpt_plan_spanning_the_whole_programme_matches_the_blocking_measurement() {
        let mut reference = engine(120_000);
        let want = reference.measure(&[1.0, 1.0]);

        let live = engine(120_000);
        let mut pass = MeasurementPass::new_excerpts(&live, &[1.0, 1.0], 1, 120_000, 0);
        let (lkfs, dbtp) = run(&mut pass, 1024);
        assert!((lkfs - want.0).abs() < 1e-9, "{lkfs} vs {want:?}");
        assert!((dbtp - want.1).abs() < 1e-9, "{dbtp} vs {want:?}");
    }

    #[test]
    fn a_sparse_excerpt_plan_lands_close_to_the_whole_programme_measurement() {
        let mut reference = engine(480_000);
        let want = reference.measure(&[1.0, 1.0]);

        let live = engine(480_000);
        let mut pass = MeasurementPass::new_excerpts(&live, &[1.0, 1.0], 5, 20_000, 2_000);
        let (lkfs, dbtp) = run(&mut pass, 1024);
        assert!((lkfs - want.0).abs() < 1.0, "{lkfs} vs {want:?}");
        assert!((dbtp - want.1).abs() < 1.0, "{dbtp} vs {want:?}");
    }

    #[test]
    fn excerpt_progress_climbs_to_one() {
        let live = engine(480_000);
        let mut pass = MeasurementPass::new_excerpts(&live, &[1.0, 1.0], 5, 20_000, 2_000);
        assert_eq!(pass.progress(), 0.0);
        pass.advance(1024);
        let partial = pass.progress();
        assert!(partial > 0.0 && partial < 1.0, "{partial}");
        run(&mut pass, 1024);
        assert_eq!(pass.progress(), 1.0);
    }

    /// A native bed with a height pair, so the measurement programme is the
    /// 5.1 re-render rather than the delivered channels.
    fn immersive_engine(frames: usize) -> PreviewEngine {
        let params: EngineParams = serde_json::from_str(
            r#"{
                "speakers": [
                    {"name": "FL", "azimuth_rad": 0.5236, "elevation_rad": 0.0, "group_gain": 1.0},
                    {"name": "FR", "azimuth_rad": -0.5236, "elevation_rad": 0.0, "group_gain": 1.0},
                    {"name": "TFL", "azimuth_rad": 0.7854, "elevation_rad": 0.7854,
                     "group_gain": 1.0},
                    {"name": "TFR", "azimuth_rad": -0.7854, "elevation_rad": 0.7854,
                     "group_gain": 1.0}
                ],
                "lfe_index": null,
                "shapes": ["left", "right", "height_left", "height_right"],
                "surround_downmix_coeff": 0.7071067811865476,
                "height_downmix_coeff": 0.7071067811865476,
                "sends": {"surround_bass_cutoff_hz": 250.0,
                          "height_low_rolloff_hz": 150.0, "height_low_rolloff_gain": 0.15,
                          "height_crossover_hz": 3000.0, "height_high_shelf_gain": 1.5,
                          "height_directional_band_hz": 8000.0,
                          "height_directional_band_gain": 1.0,
                          "lfe_cutoff_hz": 120.0, "lfe_filter_order": 4, "lfe_gain": 0.316},
                "stems": [{"routing": [["FL", 0.7], ["FR", 0.7], ["TFL", 0.6], ["TFR", 0.6]],
                           "rebalance_db": 0.0, "enabled": true, "eq_fir": [],
                           "route_scale": 1.0}],
                "master": {"lf_targets": [[0, 0.5], [1, 0.5]]},
                "output_mode": "native",
                "soft_limit_threshold": 0.0
            }"#,
        )
        .expect("engine parameters");

        let tone: Vec<f32> = (0..frames)
            .map(|i| {
                let t = i as f64 / 48_000.0;
                (0.4 * (2.0 * std::f64::consts::PI * 220.0 * t).sin()
                    + 0.1 * (2.0 * std::f64::consts::PI * 3300.0 * t).sin()) as f32
            })
            .collect();
        PreviewEngine::new(
            48_000,
            params,
            vec![Arc::new(StemSource {
                left: tone.clone(),
                right: tone,
            })],
        )
    }

    #[test]
    fn an_immersive_bed_measures_its_five_one_re_render() {
        let mut reference = immersive_engine(120_000);
        assert!(
            reference.measurement_fold().is_some(),
            "a height pair must fold"
        );
        let want = reference.measure(&[]);

        let live = immersive_engine(120_000);
        let mut pass = MeasurementPass::new(&live, &[]);
        let (lkfs, dbtp) = run(&mut pass, 1024);
        assert!(
            (lkfs - want.0).abs() < 1e-9,
            "sliced fold {lkfs} vs blocking {want:?}"
        );
        assert!(
            (dbtp - want.1).abs() < 1e-9,
            "sliced peak {dbtp} vs blocking {want:?}"
        );

        // The heights sum into the fronts, so the folded programme is louder
        // than the same bed measured as four unity-weighted channels.
        let mut unfolded = engine(120_000);
        let flat = unfolded.measure(&[1.0, 1.0]);
        assert!(
            lkfs > -70.0 && (lkfs - flat.0).abs() > 0.1,
            "fold changed nothing: {lkfs}"
        );
    }

    #[test]
    fn a_short_programme_falls_back_to_a_single_excerpt() {
        let mut reference = engine(48_000);
        let want = reference.measure(&[1.0, 1.0]);

        let live = engine(48_000);
        let mut pass = MeasurementPass::new_excerpts(&live, &[1.0, 1.0], 5, 20_000, 2_000);
        let (lkfs, dbtp) = run(&mut pass, 1024);
        assert!((lkfs - want.0).abs() < 1e-9, "{lkfs} vs {want:?}");
        assert!((dbtp - want.1).abs() < 1e-9, "{dbtp} vs {want:?}");
    }
}

mod routing {
    use upmixer_dsp_core::kernels::butter::{butter_sos, linkwitz_riley_lowpass_sos, BandType};
    use upmixer_dsp_core::routing::decorrelate::{
        velvet_pair_seeded, VELVET_SEED, VELVET_SEED_HEIGHT,
    };
    use upmixer_dsp_core::stream::params::SendParams;
    use upmixer_dsp_core::stream::routing::*;

    fn send_params() -> SendParams {
        SendParams {
            surround_bass_cutoff_hz: 250.0,
            height_low_rolloff_hz: 150.0,
            height_low_rolloff_gain: 0.15,
            height_crossover_hz: 3000.0,
            height_high_shelf_gain: 1.5,
            height_directional_band_hz: 8000.0,
            height_directional_band_gain: 1.0,
            lfe_cutoff_hz: 120.0,
            lfe_filter_order: 4,
            lfe_gain: 0.31622776601683794,
        }
    }

    /// Run a signal through both channels of a route in uneven blocks and
    /// return the four shaped sends, concatenated.
    fn blocked(state: &mut StemRouteState, signal: &[f64]) -> [Vec<f64>; 4] {
        let mut out: [Vec<f64>; 4] = Default::default();
        let mut rest = signal;
        for size in [333usize, 999, 128].iter().cycle() {
            if rest.is_empty() {
                break;
            }
            let n = (*size).min(rest.len());
            state.process(&rest[..n], &rest[..n], true, true);
            for (index, side) in out.iter_mut().enumerate() {
                side.extend_from_slice(state.signal(3 + index));
            }
            rest = &rest[n..];
        }
        out
    }

    #[test]
    fn surround_sends_match_the_offline_highpass_and_velvet_pair() {
        use upmixer_dsp_core::kernels::biquad::sosfilt;

        let sr = 48_000;
        let signal: Vec<f64> = (0..9600).map(|i| (i as f64 * 0.04).sin()).collect();
        let p = send_params();

        // Blocked in ragged sizes: the shaped sends must not depend on how
        // the render callback happens to chop the stream up.
        let mut state = StemRouteState::new(sr, &p, None, None, None);
        let got = blocked(&mut state, &signal);

        let hp = butter_sos(
            2,
            p.surround_bass_cutoff_hz / (sr as f64 / 2.0),
            BandType::High,
        );
        let shaped = sosfilt(&hp, &signal);
        let (left, right) = velvet_pair_seeded(sr, VELVET_SEED);

        for (index, fir) in [(3, left), (4, right)] {
            let want = fir.process(&shaped);
            for (i, (a, b)) in got[index - 3].iter().zip(want.iter()).enumerate() {
                assert!(
                    (a - b).abs() < 1e-12,
                    "shape {index} sample {i}: {a} vs {b}"
                );
            }
        }
    }

    #[test]
    fn height_sends_match_the_offline_elevation_eq_and_velvet_pair() {
        use upmixer_dsp_core::routing::sends::elevation_eq;

        let sr = 48_000;
        let signal: Vec<f64> = (0..9600).map(|i| (i as f64 * 0.07).sin()).collect();

        // The default skips the band section, a lifted band runs it.
        for band_gain in [1.0, 1.6] {
            let p = SendParams {
                height_directional_band_gain: band_gain,
                ..send_params()
            };

            let mut state = StemRouteState::new(sr, &p, None, None, None);
            let got = blocked(&mut state, &signal);

            let shaped = elevation_eq(
                &signal,
                sr,
                p.height_low_rolloff_hz,
                p.height_low_rolloff_gain,
                p.height_crossover_hz,
                p.height_high_shelf_gain,
                p.height_directional_band_hz,
                p.height_directional_band_gain,
            );
            let (left, right) = velvet_pair_seeded(sr, VELVET_SEED_HEIGHT);

            for (index, fir) in [(5, left), (6, right)] {
                let want = fir.process(&shaped);
                for (i, (a, b)) in got[index - 3].iter().zip(want.iter()).enumerate() {
                    assert!(
                        (a - b).abs() < 1e-12,
                        "band {band_gain} shape {index} sample {i}: {a} vs {b}"
                    );
                }
            }
        }
    }

    /// The surround and height sends of one stem must not be copies of each
    /// other: they run different seeds, so a stem placed both around and
    /// overhead does not image as one hard phantom between the two.
    #[test]
    fn surround_and_height_sends_use_different_tap_sets() {
        let (surround, _) = velvet_pair_seeded(48_000, VELVET_SEED);
        let (height, _) = velvet_pair_seeded(48_000, VELVET_SEED_HEIGHT);
        assert_ne!(surround.taps(), height.taps());
    }

    #[test]
    fn lfe_bus_matches_the_offline_lowpass_and_gain() {
        use upmixer_dsp_core::kernels::biquad::sosfilt;

        let sr = 48_000;
        let signal: Vec<f64> = (0..4800).map(|i| (i as f64 * 0.02).sin()).collect();
        let p = send_params();

        let mut bus = LfeBus::new(sr, &p);
        let got: Vec<f64> = signal.iter().map(|v| bus.tick(*v)).collect();

        let lp =
            linkwitz_riley_lowpass_sos(p.lfe_filter_order, p.lfe_cutoff_hz / (sr as f64 / 2.0));
        let want: Vec<f64> = sosfilt(&lp, &signal)
            .iter()
            .map(|v| v * p.lfe_gain)
            .collect();

        for (a, b) in got.iter().zip(want.iter()) {
            assert!((a - b).abs() < 1e-12);
        }
    }
}

use serde_json::{json, Value};
use upmixer_dsp_core::movement::{
    compile_movement, MovementCompileRequest, MovementPlacementUpdate, MovementSchedule,
    PreparedMovement,
};
use upmixer_dsp_core::spatial::panner::{panner_to_adm, PannerLayout};

fn stem(key: &str, energies: Vec<f64>, role: &str) -> Value {
    json!({
        "stem_key": key,
        "features": {"version": 1, "sample_rate": 48000,
            "frame_count": energies.len() * 480, "window_frames": 480, "energies": energies},
        "placement": {"azimuth_deg": 60.0, "elevation_deg": 0.0,
            "width_deg": 0.0, "object_size": 0.0, "lfe": 0.0,
            "diversity": 0.0, "center_level_db": 0.0},
        "settings": {"enabled": true, "role": role, "depth": 1.0,
            "response": 1.0, "sensitivity": 0.5, "start_s": 0.0},
        "object_mode": "mono"
    })
}

fn request(stems: Vec<Value>, windows: usize) -> Value {
    json!({
        "sample_rate": 48000, "duration_frames": windows * 480, "revision": 7,
        "channels": ["FL", "FR", "C", "LFE", "SL", "SR"], "stems": stems,
        "tuning": {
            "activity_floor_db": -65.0, "activity_enter_db": 6.0,
            "activity_leave_db": 3.0, "activity_dwell_ms": 80.0,
            "activity_exit_dwell_ms": 250.0, "focus_prominence_db": 6.0,
            "focus_share": 0.85, "focus_qualification_ms": 750.0,
            "focus_attack_ms": 750.0, "focus_release_ms": 1500.0,
            "vocal_hold_ms": 1200.0, "winner_hold_ms": 1000.0,
            "challenger_db": 3.0, "supporting_span_db": 24.0,
            "percussion_rise_db": 6.0, "percussion_gate_db": 6.0,
            "percussion_retrigger_ms": 120.0, "percussion_attack_ms": 40.0,
            "percussion_return_ms": 300.0, "toms_return_ms": 500.0,
            "crash_return_ms": 1000.0
        }
    })
}

fn compile(value: Value) -> MovementSchedule {
    let request: MovementCompileRequest = serde_json::from_value(value).unwrap();
    compile_movement(&request).unwrap()
}

fn x_at(schedule: &MovementSchedule, stem_index: usize, time_us: i64) -> f64 {
    schedule
        .stem(stem_index)
        .unwrap()
        .events
        .iter()
        .rev()
        .find(|event| event.time_us <= time_us)
        .unwrap()
        .position[0]
}

#[test]
fn live_placement_updates_match_full_compilation_and_reject_invalid_batches() {
    for role in ["auto", "supporting", "featured"] {
        let mut energies = vec![0.0; 800];
        energies[100..400].fill(0.01);
        let mut guitar = stem("Guitar", energies.clone(), role);
        guitar["object_mode"] = json!("linked-stereo");
        let mut request: MovementCompileRequest = serde_json::from_value(request(
            vec![
                guitar,
                stem("Lead Vocals", energies.clone(), "auto"),
                stem("Crowd", energies, "supporting"),
            ],
            800,
        ))
        .unwrap();
        let mut prepared = PreparedMovement::new(request.clone()).unwrap();
        for azimuth in [-120.0, 90.0, 0.0] {
            request.revision += 1;
            request.stems[0].placement.azimuth_deg = azimuth;
            request.stems[0].placement.elevation_deg = 30.0;
            request.stems[0].placement.width_deg = 56.0;
            request.stems[0].placement.object_size = 0.2;
            let update = || MovementPlacementUpdate {
                stem_key: "Guitar".into(),
                placement: request.stems[0].placement.clone(),
                home_gains: vec![],
                home_right_gains: vec![],
            };
            let before = prepared.schedule.clone();
            assert!(prepared
                .update_placements(request.revision, vec![update(), update()])
                .is_err());
            assert_eq!(prepared.schedule, before);
            let mut invalid = update();
            invalid.placement.elevation_deg = f64::NAN;
            assert!(prepared
                .update_placements(request.revision, vec![invalid])
                .is_err());
            assert_eq!(prepared.schedule, before);
            let mut partial = update();
            partial.placement.left_right = Some(0.25);
            assert!(prepared
                .update_placements(request.revision, vec![partial])
                .is_err());
            assert_eq!(prepared.schedule, before);
            assert_eq!(
                *prepared
                    .update_placements(request.revision, vec![update()])
                    .unwrap(),
                compile_movement(&request).unwrap()
            );
        }
    }
}

#[test]
fn editing_resting_placement_preserves_the_featured_destination() {
    for role in ["auto", "featured"] {
        let mut energies = vec![0.0; 800];
        energies[100..400].fill(0.01);
        let mut original = stem("Guitar", energies, role);
        original["object_mode"] = json!("linked-stereo");
        original["placement"]["width_deg"] = json!(32.0);
        let mut edited = original.clone();
        edited["placement"]["azimuth_deg"] = json!(-120.0);
        edited["placement"]["elevation_deg"] = json!(30.0);
        let before = compile(request(vec![original], 800));
        let after = compile(request(vec![edited], 800));

        assert_ne!(x_at(&before, 0, 0), x_at(&after, 0, 0));
        for schedule in [&before, &after] {
            assert_eq!(x_at(schedule, 0, 7_000_000), x_at(schedule, 0, 0));
        }
        let featured: Vec<_> = [&before, &after]
            .iter()
            .map(|schedule| {
                schedule
                    .stem(0)
                    .unwrap()
                    .events
                    .iter()
                    .rev()
                    .find(|event| event.time_us <= 3_000_000)
                    .unwrap()
            })
            .collect();
        assert_eq!(featured[0].position, featured[1].position);
        assert_eq!(featured[0].right_position, featured[1].right_position);
        assert_eq!(featured[0].gains, featured[1].gains);
        assert_eq!(featured[0].right_gains, featured[1].right_gains);
    }
}

#[test]
fn manual_featured_moves_only_during_activity_with_attack_and_return() {
    let mut energies = vec![0.0; 600];
    energies[100..300].fill(0.01);
    let schedule = compile(request(vec![stem("Guitar", energies, "featured")], 600));
    let home = x_at(&schedule, 0, 0);
    assert_eq!(x_at(&schedule, 0, 1_040_000), home);
    let early = x_at(&schedule, 0, 1_200_000);
    assert!(
        early.abs() < home.abs() && early.abs() > 0.3,
        "attack must start at activity, not track zero"
    );
    assert!(x_at(&schedule, 0, 2_000_000).abs() < 1e-9);
    let returning = x_at(&schedule, 0, 3_600_000);
    assert!(
        returning.abs() > 0.0 && returning.abs() < home.abs(),
        "release must be a gradual return"
    );
    assert!((x_at(&schedule, 0, 5_000_000) - home).abs() < 1e-9);
}

#[test]
fn odd_window_percussion_hit_is_retained_and_recovers() {
    let mut energies = vec![0.0; 300];
    energies[101] = 0.1;
    let schedule = compile(request(vec![stem("Toms", energies, "supporting")], 300));
    let events = &schedule.stem(0).unwrap().events;
    let home = events[0].position;
    assert!(
        events
            .iter()
            .any(|event| event.time_us >= 1_000_000 && event.position != home),
        "a 10 ms hit between 20 ms ticks must create an excursion"
    );
    assert!((x_at(&schedule, 0, 1_800_000) - home[0]).abs() < 1e-9);
}

#[test]
fn vocal_priority_survives_half_second_gap_even_with_vocal_motion_off() {
    let mut vocals = vec![0.0; 700];
    vocals[0..200].fill(0.02);
    vocals[250..400].fill(0.02);
    let mut vocal = stem("Vocals", vocals, "auto");
    vocal["settings"]["enabled"] = json!(false);
    vocal["settings"]["depth"] = json!(0.0);
    let schedule = compile(request(
        vec![vocal, stem("Guitar", vec![0.01; 700], "auto")],
        700,
    ));
    let events = &schedule.stem(1).unwrap().events;
    let home = events[0].position;
    assert!(
        events
            .iter()
            .filter(|event| event.time_us < 5_200_000)
            .all(|event| event.position == home),
        "the vocal gap must not promote guitar"
    );
    assert!(events
        .iter()
        .any(|event| event.time_us > 5_200_000 && event.position != home));
}

#[test]
fn off_and_ordinary_stereo_keep_saved_speaker_weights_exactly() {
    for stereo in [false, true] {
        let mut input = stem("Crowd", vec![0.01; 200], "featured");
        input["object_mode"] = Value::Null;
        input["settings"]["enabled"] = json!(stereo);
        let weights = if stereo {
            vec![0.3, 0.4]
        } else {
            vec![0.3, 0.4, 0.0, 0.7, 0.0, 0.0]
        };
        input["home_gains"] = json!(weights);
        let mut value = request(vec![input], 200);
        if stereo {
            value["channels"] = json!(["FL", "FR"]);
        }
        let schedule = compile(value);
        let events = &schedule.stem(0).unwrap().events;
        assert_eq!(events.len(), 1);
        assert_eq!(events[0].gains, weights);
    }
}

#[test]
fn unknown_custom_stem_can_be_manually_featured() {
    let schedule = compile(request(
        vec![stem("Synth Layer", vec![0.01; 300], "featured")],
        300,
    ));
    assert!(x_at(&schedule, 0, 2_000_000).abs() < 1e-9);
}

#[test]
fn short_interval_moves_and_finishes_return_before_stop() {
    let mut input = stem("Guitar", vec![0.01; 300], "featured");
    input["settings"]["start_s"] = json!(1.0);
    input["settings"]["end_s"] = json!(1.2);
    let schedule = compile(request(vec![input], 300));
    let events = &schedule.stem(0).unwrap().events;
    let home = events[0].position;
    assert!(events.iter().any(|event| event.position != home));
    assert_eq!(x_at(&schedule, 0, 1_200_000), home[0]);
    assert!(
        events
            .iter()
            .skip(1)
            .all(|event| event.time_us >= 1_000_000 && event.time_us + 5208 <= 1_200_000),
        "short intervals must shorten the envelope, not overrun the stop"
    );
}

#[test]
fn finite_interval_reserves_a_gradual_return_before_its_end() {
    let mut input = stem("Guitar", vec![0.01; 600], "featured");
    input["settings"]["end_s"] = json!(5.0);
    let schedule = compile(request(vec![input], 600));
    let home = x_at(&schedule, 0, 0);
    let returning = x_at(&schedule, 0, 4_200_000);
    assert!(
        returning.abs() > 0.0 && returning.abs() < home.abs(),
        "an explicit stop must reserve the return, not snap home in the final event"
    );
    assert!((x_at(&schedule, 0, 5_000_000) - home).abs() < 1e-12);
}

#[test]
fn linked_object_metadata_endpoints_reproduce_shared_gains_and_transition() {
    let mut input = stem("Guitar", vec![0.01; 300], "featured");
    input["object_mode"] = json!("linked-stereo");
    input["placement"]["width_deg"] = json!(30.0);
    // Saved bed weights are not the authored object's two endpoint routes.
    input["home_gains"] = json!([0.3, 0.4, 0.0, 0.7, 0.0, 0.0]);
    input["home_right_gains"] = input["home_gains"].clone();
    let schedule = compile(request(vec![input], 300));
    let route = schedule.stem(0).unwrap();
    let layout = PannerLayout::new(&["FL", "FR", "C", "LFE", "SL", "SR"]);
    for event in &route.events {
        for (position, gains) in [
            (event.position, &event.gains),
            (
                event.right_position.unwrap(),
                event.right_gains.as_ref().unwrap(),
            ),
        ] {
            let rendered = layout.cartesian_object_route(panner_to_adm(position), 0.0, false, &[]);
            assert!(rendered
                .iter()
                .zip(gains)
                .all(|(a, b)| (a - b).abs() <= 1e-12));
        }
    }
    assert_ne!(
        route.events[0].position,
        route.events[0].right_position.unwrap()
    );
    let next = &route.events[1];
    let mut gains = vec![0.0; 6];
    route.sample_into_at(next.time_us as f64 + 2604.0, false, &mut gains);
    for (channel, gain) in gains.iter().enumerate() {
        assert!(
            (gain - (route.events[0].gains[channel] + next.gains[channel]) * 0.5).abs() < 1e-12
        );
    }
}

#[test]
fn focus_rejects_bleed_and_isolated_full_level_glitches() {
    for mut energies in [vec![1e-7; 500], vec![0.0; 500]] {
        energies[100] = 1.0;
        let schedule = compile(request(vec![stem("Guitar", energies, "auto")], 500));
        assert_eq!(schedule.stem(0).unwrap().events.len(), 1);
    }
}

#[test]
fn competition_ties_are_stable_under_input_reordering() {
    let guitar = stem("Guitar", vec![0.01; 500], "auto");
    let piano = stem("Piano", vec![0.01; 500], "auto");
    let first = compile(request(vec![guitar.clone(), piano.clone()], 500));
    let second = compile(request(vec![piano, guitar], 500));
    for key in ["Guitar", "Piano"] {
        let a = first.stems.iter().find(|s| s.stem_key == key).unwrap();
        let b = second.stems.iter().find(|s| s.stem_key == key).unwrap();
        assert_eq!(
            serde_json::to_value(&a.events).unwrap(),
            serde_json::to_value(&b.events).unwrap()
        );
    }
    // Increase sensitivity enough that equal shares qualify; exact ties use key order.
    let mut value = request(
        vec![
            stem("Piano", vec![0.01; 500], "auto"),
            stem("Guitar", vec![0.01; 500], "auto"),
        ],
        500,
    );
    for input in value["stems"].as_array_mut().unwrap() {
        input["settings"]["sensitivity"] = json!(1.0);
    }
    let schedule = compile(value);
    assert!(x_at(&schedule, 1, 2_000_000).abs() < 1e-9);
    assert_eq!(schedule.stem(0).unwrap().events.len(), 1);
}

#[test]
fn supporting_lift_is_exactly_static_without_heights_and_at_ceiling() {
    for key in ["Backing Vocals", "Crowd"] {
        let mut input = stem(key, vec![0.01; 500], "supporting");
        input["placement"]["elevation_deg"] = json!(-20.0);
        let schedule = compile(request(vec![input.clone()], 500));
        assert_eq!(schedule.stem(0).unwrap().events.len(), 1);
        let mut value = request(vec![input], 500);
        value["channels"] = json!(["FL", "FR", "C", "LFE", "SL", "SR", "TFL", "TFR", "TBL", "TBR"]);
        value["stems"][0]["placement"]["elevation_deg"] = json!(90.0);
        let schedule = compile(value);
        assert_eq!(schedule.stem(0).unwrap().events.len(), 1);
    }
}

#[test]
fn finite_supporting_intervals_shorten_lift_and_return_smoothly() {
    for key in ["Backing Vocals", "Crowd"] {
        for end in [1.4, 5.0] {
            for response in [0.5, 1.0, 2.0] {
                let mut input = stem(key, vec![0.01; 600], "supporting");
                input["settings"]["start_s"] = json!(1.0);
                input["settings"]["end_s"] = json!(end);
                input["settings"]["response"] = json!(response);
                let mut value = request(vec![input], 600);
                value["channels"] =
                    json!(["FL", "FR", "C", "LFE", "SL", "SR", "TFL", "TFR", "TBL", "TBR"]);
                let schedule = compile(value);
                let events = &schedule.stem(0).unwrap().events;
                assert!(
                    events.iter().any(|event| event.position[2] > 0.0),
                    "{key} at response {response} before {end}s"
                );
                assert!(
                    events
                        .windows(2)
                        .filter(|pair| pair[1].position[2] < pair[0].position[2])
                        .count()
                        >= 2,
                    "{key} must return gradually at response {response} before {end}s"
                );
                assert!(events.last().unwrap().position[2].abs() < 1e-12);
                assert!(events
                    .iter()
                    .skip(1)
                    .all(|event| event.time_us + 5208 <= (end * 1_000_000.0) as i64));
            }
        }
    }
}

#[test]
fn shorter_stems_return_to_supporting_after_their_pcm_ends() {
    let schedule = compile(request(
        vec![stem("Guitar", vec![0.01; 200], "featured")],
        600,
    ));
    let home = x_at(&schedule, 0, 0);
    assert!(x_at(&schedule, 0, 1_500_000).abs() < 1e-9);
    assert!((x_at(&schedule, 0, 5_000_000) - home).abs() < 1e-9);
}

#[test]
fn supporting_lift_preserves_bed_lfe_and_crowd_is_smaller() {
    let mut inputs = vec![
        stem("Backing Vocals", vec![0.01; 600], "supporting"),
        stem("Crowd", vec![0.01; 600], "supporting"),
    ];
    for input in &mut inputs {
        input["object_mode"] = Value::Null;
        input["home_gains"] = json!([0.3, 0.4, 0.0, 0.7, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0]);
    }
    let mut value = request(inputs, 600);
    value["channels"] = json!(["FL", "FR", "C", "LFE", "SL", "SR", "TFL", "TFR", "TBL", "TBR"]);
    let schedule = compile(value);
    let mut peaks = Vec::new();
    for route in &schedule.stems {
        peaks.push(
            route
                .events
                .iter()
                .map(|e| e.position[2])
                .fold(0.0, f64::max),
        );
        assert!(route.events.len() > 1);
        for event in &route.events {
            assert_eq!(event.gains[3], 0.7);
            let main_norm = event
                .gains
                .iter()
                .enumerate()
                .filter(|(i, _)| *i != 3)
                .map(|(_, g)| g * g)
                .sum::<f64>()
                .sqrt();
            assert!((main_norm - 0.5).abs() < 1e-12);
        }
    }
    assert!(peaks[1] > 0.0 && peaks[1] < peaks[0]);
}

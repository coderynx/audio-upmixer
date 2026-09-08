use crate::movement::MovementStemSchedule;
use crate::spatial::panner::{PannerLayout, StemPlacement};
use crate::stream::params::{EngineParams, ObjectMode, SendShape, SpeakerParams, StemParams};
use crate::stream::routing::{ambient_expanded_slot, shape_index, StemRouteState, AMBIENT_TEXTURE};

use super::PreviewEngine;

pub(crate) struct StemMixRoute {
    pub regular: Vec<(usize, usize, f64)>,
    pub dynamic_regular: Vec<(usize, usize, f64)>,
    pub lfe_weight: f64,
    pub objects: Option<Vec<ObjectMixRoute>>,
    pub ambient: Vec<(usize, usize, f64)>,
    pub ambient_texture: Vec<(usize, usize, f64)>,
    pub needs_surround: bool,
    pub needs_height: bool,
    pub has_surround: [bool; 2],
    pub has_height: [bool; 2],
}

pub(crate) struct ObjectMixRoute {
    pub authored_channel: usize,
    pub signal: usize,
    pub endpoint: usize,
    pub speakers: Vec<(usize, f64)>,
    pub gain: f64,
}

/// Playback keeps authored objects in their authored channels; normalization
/// projects them onto speakers using panning gains. Object metadata gain is
/// applied later by the speaker renderer, outside this per-stem assembly.
#[derive(Clone, Copy)]
pub(crate) enum StemAssemblyPolicy {
    Render,
    Normalization,
}

/// Add one already-routed stem to a caller-owned bed. LFE remains outside this
/// module because `LfeBus` owns its stateful contribution at bed level.
pub(crate) fn assemble_stem_into(
    mix: &StemMixRoute,
    route: &StemRouteState,
    speakers: &[SpeakerParams],
    count: usize,
    policy: StemAssemblyPolicy,
    bed: &mut [Vec<f64>],
    mut lfe_sum: Option<&mut [f64]>,
    movement: Option<&MovementStemSchedule>,
    start_frame: usize,
    sample_rate: u32,
    steady_gains: Option<(f64, f64)>,
    mut gain_at: impl FnMut() -> f64,
    mut route_scale_at: impl FnMut() -> f64,
) {
    let mut movement_gains = vec![0.0; speakers.len()];
    let mut projected_movement = movement
        .filter(|_| mix.objects.is_none() || matches!(policy, StemAssemblyPolicy::Normalization));
    if mix.objects.is_none() {
        let us_per_sample = 1_000_000.0 / sample_rate.max(1) as f64;
        if let Some(gains) = projected_movement.and_then(|schedule| {
            schedule.constant_gains_for_range(
                start_frame as f64 * us_per_sample,
                (start_frame + count.saturating_sub(1)) as f64 * us_per_sample,
                false,
            )
        }) {
            movement_gains.copy_from_slice(gains);
            projected_movement = None;
        }
    }
    if let Some((gain, route_scale)) = steady_gains.filter(|_| {
        matches!(policy, StemAssemblyPolicy::Render)
            && (mix.objects.is_some() || projected_movement.is_none())
    }) {
        // Held gains allow contiguous channel loops and SIMD, without changing
        // the sample-wise path used by fader ramps or movement transitions.
        let routed_gain = gain * route_scale;
        let mut add = |channel: usize, signal: usize, weight: f64, group: f64| {
            for (target, source) in bed[channel][..count].iter_mut().zip(route.signal(signal)) {
                *target += source * weight * group * routed_gain;
            }
        };
        if let Some(objects) = &mix.objects {
            for object in objects {
                add(object.authored_channel, object.signal, 1.0, 1.0);
            }
        } else if movement.is_some() {
            for &(channel, signal, group) in &mix.dynamic_regular {
                add(channel, signal, movement_gains[channel], group);
            }
        } else {
            for &(channel, signal, weight) in &mix.regular {
                add(channel, signal, weight, speakers[channel].group_gain);
            }
        }
        if route.has_ambient() {
            for &(channel, signal, weight) in mix.ambient.iter().chain(&mix.ambient_texture) {
                add(channel, signal, weight, 1.0);
            }
        }
        if let Some(lfe_sum) = lfe_sum {
            for (target, source) in lfe_sum[..count]
                .iter_mut()
                .zip(route.signal(shape_index(SendShape::Mono)))
            {
                *target += source * mix.lfe_weight * gain;
            }
        }
        return;
    }
    for i in 0..count {
        let gain = gain_at();
        let route_scale = route_scale_at();
        let routed_gain = gain * route_scale;
        if let Some(schedule) = projected_movement {
            schedule.sample_into_at(
                (start_frame as f64 + i as f64) * 1_000_000.0 / sample_rate.max(1) as f64,
                false,
                &mut movement_gains,
            );
        }
        if let Some(objects) = &mix.objects {
            for object in objects {
                let sample = route.signal(object.signal)[i];
                match policy {
                    StemAssemblyPolicy::Render => {
                        // Authored object channels are projected in the
                        // speaker renderer later; apply the measured route
                        // scale here so that renderer and bed paths agree.
                        bed[object.authored_channel][i] += sample * routed_gain;
                    }
                    StemAssemblyPolicy::Normalization => {
                        if let Some(schedule) = movement {
                            if object.endpoint == 1 {
                                schedule.sample_into_at(
                                    (start_frame as f64 + i as f64) * 1_000_000.0
                                        / sample_rate.max(1) as f64,
                                    true,
                                    &mut movement_gains,
                                );
                            }
                            for (channel, weight) in movement_gains.iter().copied().enumerate() {
                                if weight > 0.0 && speakers[channel].name != "LFE" {
                                    bed[channel][i] += sample * gain * weight;
                                }
                            }
                        } else {
                            for &(channel, weight) in &object.speakers {
                                bed[channel][i] += sample * gain * weight;
                            }
                        }
                    }
                }
            }
        } else {
            if movement.is_some() {
                for &(channel, signal, group_gain) in &mix.dynamic_regular {
                    bed[channel][i] += route.signal(signal)[i]
                        * movement_gains[channel]
                        * group_gain
                        * routed_gain;
                }
            } else {
                for &(channel, signal, weight) in &mix.regular {
                    bed[channel][i] += route.signal(signal)[i]
                        * weight
                        * speakers[channel].group_gain
                        * routed_gain;
                }
            }
        }
        if route.has_ambient() {
            for &(channel, signal, weight) in &mix.ambient {
                bed[channel][i] += route.signal(signal)[i] * weight * routed_gain;
            }
            for &(channel, signal, weight) in &mix.ambient_texture {
                bed[channel][i] += route.signal(signal)[i] * weight * routed_gain;
            }
        }
        if let Some(lfe_sum) = lfe_sum.as_deref_mut() {
            // LFE is an explicit effect send. Route normalization belongs to
            // the full-range bed, so a newly measured scale must not change
            // the LFE level (the Python route adds it after normalization).
            lfe_sum[i] += route.signal(shape_index(SendShape::Mono))[i] * mix.lfe_weight * gain;
        }
    }
}

pub(crate) fn build_stem_mix_routes(
    params: &EngineParams,
    layout: &PannerLayout,
) -> Vec<StemMixRoute> {
    let mut next_object = params.speakers.len();
    params
        .stems
        .iter()
        .enumerate()
        .map(|(stem_index, stem)| {
            let objects = direct_object_routes(params, stem, layout).map(|(gain, routes)| {
                routes
                    .into_iter()
                    .map(|(signal, speakers)| {
                        let authored_channel = next_object;
                        next_object += 1;
                        ObjectMixRoute {
                            authored_channel,
                            signal,
                            endpoint: usize::from(signal == 1),
                            speakers,
                            gain,
                        }
                    })
                    .collect()
            });
            let mut lfe_weight = 0.0;
            let mut regular = Vec::new();
            let dynamic_regular = if objects.is_none() {
                params
                    .speakers
                    .iter()
                    .enumerate()
                    .filter_map(|(channel, speaker)| {
                        (params.lfe_index != Some(channel)).then_some((
                            channel,
                            shape_index(params.shapes[channel]),
                            speaker.group_gain,
                        ))
                    })
                    .collect()
            } else {
                Vec::new()
            };
            for (name, weight) in &stem.routing {
                if *weight == 0.0 {
                    continue;
                }
                if name == "LFE" {
                    if objects.is_none() {
                        lfe_weight += weight;
                    }
                } else if objects.is_none() {
                    if let Some(channel) = params.speaker_index(name) {
                        regular.push((channel, shape_index(params.shapes[channel]), *weight));
                    }
                }
            }
            let (needs_surround, needs_height) = movement_route_flags(params, stem_index);
            StemMixRoute {
                regular,
                dynamic_regular,
                lfe_weight,
                objects,
                ambient: ambient_feeds(params),
                ambient_texture: ambient_texture_feeds(params),
                needs_surround,
                needs_height,
                has_surround: [
                    params.ambient_side_share(SendShape::SurroundLeft) > 0.0,
                    params.ambient_side_share(SendShape::SurroundRight) > 0.0,
                ],
                has_height: [
                    params.ambient_side_share(SendShape::HeightLeft) > 0.0,
                    params.ambient_side_share(SendShape::HeightRight) > 0.0,
                ],
            }
        })
        .collect()
}

fn movement_route_flags(params: &EngineParams, stem_index: usize) -> (bool, bool) {
    let Some(stem) = params.stems.get(stem_index) else {
        return (false, false);
    };
    let mut needs_surround = false;
    let mut needs_height = false;
    for (name, weight) in &stem.routing {
        if *weight == 0.0 || name == "LFE" {
            continue;
        }
        if let Some(channel) = params.speaker_index(name) {
            needs_surround |= matches!(
                params.shapes[channel],
                SendShape::SurroundLeft | SendShape::SurroundRight
            );
            needs_height |= matches!(
                params.shapes[channel],
                SendShape::HeightLeft | SendShape::HeightRight
            );
        }
    }
    if let Some(schedule_stem) = params
        .movement_schedule
        .as_ref()
        .and_then(|schedule| schedule.stem(stem_index))
    {
        for event in &schedule_stem.events {
            for (channel, gain) in event.gains.iter().copied().enumerate() {
                if gain <= 0.0 || channel >= params.shapes.len() {
                    continue;
                }
                needs_surround |= matches!(
                    params.shapes[channel],
                    SendShape::SurroundLeft | SendShape::SurroundRight
                );
                needs_height |= matches!(
                    params.shapes[channel],
                    SendShape::HeightLeft | SendShape::HeightRight
                );
            }
        }
    }
    (needs_surround, needs_height)
}

pub(crate) fn update_movement_route_flags(params: &EngineParams, routes: &mut [StemMixRoute]) {
    for (stem_index, route) in routes.iter_mut().enumerate() {
        if route.objects.is_none() {
            let (needs_surround, needs_height) = movement_route_flags(params, stem_index);
            route.needs_surround = needs_surround;
            route.needs_height = needs_height;
        }
    }
}

fn direct_object_routes(
    params: &EngineParams,
    stem: &StemParams,
    layout: &PannerLayout,
) -> Option<(f64, Vec<(usize, Vec<(usize, f64)>)>)> {
    let mode = stem.object_mode?;
    let placement = stem.object_placement.as_ref()?;
    let point = StemPlacement::new(
        placement.azimuth_deg,
        placement.elevation_deg,
        placement.width_deg,
        placement.object_size,
        0.0,
    );
    let routes: Vec<(usize, Vec<f64>)> = match mode {
        ObjectMode::LinkedStereo => layout
            .object_routes_with_metadata(
                &point,
                placement.channel_lock,
                &placement
                    .zone_exclusion
                    .iter()
                    .map(String::as_str)
                    .collect::<Vec<_>>(),
            )
            .into_iter()
            .enumerate()
            .collect(),
        ObjectMode::Mono => vec![(
            2,
            layout.exact_object_route_with_metadata(
                placement.azimuth_deg,
                placement.elevation_deg,
                placement.object_size,
                placement.channel_lock,
                &placement
                    .zone_exclusion
                    .iter()
                    .map(String::as_str)
                    .collect::<Vec<_>>(),
            ),
        )],
    };
    Some((
        placement.gain.max(0.0),
        routes
            .into_iter()
            .map(|(signal, route)| {
                let speakers = route
                    .into_iter()
                    .enumerate()
                    .filter_map(|(channel, gain)| {
                        (params.lfe_index != Some(channel) && gain > 0.0).then_some((channel, gain))
                    })
                    .collect();
                (signal, speakers)
            })
            .collect(),
    ))
}

fn ambient_feeds(params: &EngineParams) -> Vec<(usize, usize, f64)> {
    let mut feeds = Vec::new();
    for channel in 0..params.speakers.len() {
        let shape = params.shapes[channel];
        let slot = match ambient_expanded_slot(&params.speakers[channel].name) {
            Some(slot)
                if matches!(
                    shape,
                    SendShape::SurroundLeft
                        | SendShape::SurroundRight
                        | SendShape::HeightLeft
                        | SendShape::HeightRight
                ) =>
            {
                slot
            }
            _ => continue,
        };
        let weight = params.ambient_side_share(shape) * params.speakers[channel].group_gain;
        feeds.push((channel, slot, weight));
    }
    feeds
}

fn ambient_texture_feeds(params: &EngineParams) -> Vec<(usize, usize, f64)> {
    params
        .speakers
        .iter()
        .enumerate()
        .filter_map(|(channel, speaker)| {
            let side = match params.shapes[channel] {
                SendShape::HeightLeft => 0,
                SendShape::HeightRight => 1,
                _ => return None,
            };
            Some((
                channel,
                AMBIENT_TEXTURE + side,
                params.ambient_side_share(params.shapes[channel]) * speaker.group_gain,
            ))
        })
        .collect()
}

impl PreviewEngine {
    pub(crate) fn assemble_stem_for_normalization_into(
        &self,
        stem: usize,
        start_frame: usize,
        count: usize,
        speakers: &mut [Vec<f64>],
    ) {
        let Some(mix) = self.graph.stem_mix_routes.get(stem) else {
            return;
        };
        let route = &self.graph.routes[stem];
        assemble_stem_into(
            mix,
            route,
            &self.params.speakers,
            count,
            StemAssemblyPolicy::Normalization,
            speakers,
            None,
            self.params
                .movement_schedule
                .as_ref()
                .and_then(|schedule| schedule.stem(stem)),
            start_frame,
            self.sample_rate,
            None,
            || 1.0,
            || 1.0,
        );
    }
}

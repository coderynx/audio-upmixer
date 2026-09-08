//! Live parameter edits and the rebuilds a topology change forces.

use super::mix::{build_stem_mix_routes, update_movement_route_flags};
use super::{build_decorrelator, build_unifier, graph::EngineGraph, PreviewEngine, GAIN_RAMP_MS};
use crate::mastering::dyneq::DynamicEq;
use crate::stream::limiter::StreamingLimiter;
use crate::stream::params::{EngineParams, SendParams, StemParams};
use crate::stream::routing::StemRouteState;
use crate::stream::state::{OnePole, StreamingCompressor};

/// One stem's routing state, including the ambient half when the stem asks
/// for one. Both the initial build and a stem-count change come through here.
pub(crate) fn build_route(
    sample_rate: u32,
    sends: &SendParams,
    stem: &StemParams,
    destinations: &[&str],
) -> StemRouteState {
    let mut route = StemRouteState::new_for_layout(
        sample_rate,
        sends,
        stem.eq.clone(),
        stem.dynamic_eq.clone(),
        stem.dynamics,
        destinations,
    );
    route.set_ambient(
        sample_rate,
        sends,
        stem.wants_ambient_or_texture(),
        stem.wants_ambient(),
        stem.ambient_height_crossover_hz,
        stem.ambient_height_cutoff_hz,
        stem.ambient_trim_db,
        stem.height_texture,
    );
    route
}

/// Whether anything the per-stem routing reads has moved: the signals it
/// builds, the speakers it sends them to, or the gains on the way.
fn routing_changed(old: &EngineParams, new: &EngineParams) -> bool {
    if old.stems.len() != new.stems.len()
        || movement_schedule_key(old) != movement_schedule_key(new)
        || old.shapes != new.shapes
        || old
            .speakers
            .iter()
            .map(|speaker| &speaker.name)
            .ne(new.speakers.iter().map(|speaker| &speaker.name))
    {
        return true;
    }
    let gains: Vec<f64> = old.speakers.iter().map(|s| s.group_gain).collect();
    if gains
        != new
            .speakers
            .iter()
            .map(|s| s.group_gain)
            .collect::<Vec<f64>>()
    {
        return true;
    }
    old.stems.iter().zip(&new.stems).any(|(a, b)| {
        a.routing != b.routing
            || a.eq != b.eq
            || a.dynamics != b.dynamics
            || a.ambient_rear != b.ambient_rear
            || a.ambient_height != b.ambient_height
            || a.ambient_trim_db != b.ambient_trim_db
            || a.height_texture != b.height_texture
            || a.ambient_height_cutoff_hz != b.ambient_height_cutoff_hz
            || a.ambient_height_crossover_hz != b.ambient_height_crossover_hz
            || a.object_mode != b.object_mode
            || a.object_placement != b.object_placement
    })
}

/// Whether the compact speaker/source route description itself changed. A
/// per-stem ambient slider, filter, or texture edit is consumed by the live
/// `StemRouteState`; rebuilding this table for those edits needlessly repans
/// every static object. Movement revisions only alter the bed destination
/// flags, refreshed separately below.
fn mix_routes_changed(old: &EngineParams, new: &EngineParams) -> bool {
    if old.stems.len() != new.stems.len()
        || old.shapes != new.shapes
        || old
            .speakers
            .iter()
            .map(|speaker| (&speaker.name, speaker.group_gain))
            .ne(new
                .speakers
                .iter()
                .map(|speaker| (&speaker.name, speaker.group_gain)))
    {
        return true;
    }
    old.stems.iter().zip(&new.stems).any(|(a, b)| {
        a.routing != b.routing
            || a.object_mode != b.object_mode
            || a.object_placement != b.object_placement
    })
}

fn movement_schedule_key(params: &EngineParams) -> Option<(u64, u32, usize)> {
    params.movement_schedule.as_ref().map(|schedule| {
        (
            schedule.revision,
            schedule.sample_rate,
            schedule.duration_frames,
        )
    })
}

fn object_topology(params: &EngineParams) -> Vec<usize> {
    params
        .stems
        .iter()
        .map(|stem| {
            if stem.object_placement.is_none() {
                return 0;
            }
            match stem.object_mode {
                Some(crate::stream::params::ObjectMode::LinkedStereo) => 2,
                Some(crate::stream::params::ObjectMode::Mono) => 1,
                None => 0,
            }
        })
        .collect()
}

fn master_changed_without_firs(
    old: &crate::stream::params::MasterParams,
    new: &crate::stream::params::MasterParams,
) -> bool {
    old.head != new.head
        || old.reference_gain != new.reference_gain
        || old.eq_strength != new.eq_strength
        || old.dynamic_eq != new.dynamic_eq
        || old.compressor != new.compressor
        || old.bass != new.bass
        || old.clip != new.clip
        || old.limiter != new.limiter
        || old.lf_targets != new.lf_targets
        || old.output_gain != new.output_gain
}

fn decorrelator_topology_changed(
    old: Option<crate::mastering::bass::BassParams>,
    new: Option<crate::mastering::bass::BassParams>,
) -> bool {
    match (old, new) {
        (Some(a), Some(b)) => {
            a.unify_hz != b.unify_hz
                || a.decorr_low_hz != b.decorr_low_hz
                || a.decorr_high_hz != b.decorr_high_hz
                || a.decorr_sections != b.decorr_sections
                || a.decorr_max_delay_ms != b.decorr_max_delay_ms
                || a.decorr_fast_ms != b.decorr_fast_ms
                || a.decorr_slow_ms != b.decorr_slow_ms
        }
        (None, None) => false,
        _ => true,
    }
}

impl PreviewEngine {
    /// Replace one stem's FIR without putting every tap through JSON.
    pub fn set_stem_eq_taps(&mut self, index: usize, taps: Vec<f64>) {
        let Some(stem) = self.params.stems.get_mut(index) else {
            return;
        };
        if stem.eq_fir == taps {
            return;
        }
        stem.eq_fir = taps;
        if let Some(route) = self.graph.routes.get_mut(index) {
            route.retune(
                self.sample_rate,
                &self.params.sends,
                self.params.stems[index].eq.clone(),
                self.params.stems[index].dynamic_eq.clone(),
                self.params.stems[index].dynamics,
                false,
                true,
                false,
                false,
            );
        }
        self.clear_route_scales();
    }

    /// Replace the mastering EQ FIR without putting its taps through JSON.
    pub fn set_master_eq_taps(&mut self, taps: Vec<f64>) {
        if self.params.master.eq_fir == taps {
            return;
        }
        self.params.master.eq_fir = taps;
        for chain in &mut self.graph.causal {
            chain.set_eq_fir(&self.params.master.eq_fir);
        }
    }

    /// Replace the reference-match FIR without putting its taps through JSON.
    pub fn set_reference_taps(&mut self, taps: Vec<f64>) {
        if self.params.master.reference_fir == taps {
            return;
        }
        self.params.master.reference_fir = taps;
        for chain in &mut self.graph.causal {
            chain.set_reference_fir(&self.params.master.reference_fir);
        }
    }

    /// Replace the parameter block, keeping the loaded stems, the playhead,
    /// and — outside a channel-layout change — every filter's carried state
    /// and both look-ahead queues.
    ///
    /// Mute, solo, rebalance, routing, mastering and output-mode changes all
    /// arrive this way, so there is one path for "the mix changed" rather
    /// than a special case per control. Each stage only re-derives the parts
    /// of itself that actually moved: nothing here re-renders a preroll or
    /// discards `pre`/`post`, so playback never gaps. The new mix reaches
    /// the speakers once the look-ahead already rendered under the old
    /// params has drained — audible lag on the order of the LF unifier's
    /// horizon plus the limiter's lookahead, not audible silence.
    pub fn update_params(&mut self, mut params: EngineParams) {
        // A monitor update can arrive before its replacement schedule. Keep
        // the installed immutable schedule live until the atomic installer
        // supplies a new one; otherwise one ordinary update needlessly
        // rebuilds the render graph twice and briefly falls back to static
        // routing.
        if params.movement_schedule.is_none() {
            params.movement_schedule = self.params.movement_schedule.clone();
        }
        self.update_params_with_schedule_policy(params)
    }

    /// Replace the ordinary parameters and their compiled movement schedule as
    /// one validated update. A `None` schedule intentionally clears the
    /// installed schedule; unlike [`Self::update_params`], this method never
    /// carries the previous schedule forward.
    pub fn update_params_with_movement(
        &mut self,
        mut params: EngineParams,
        schedule: Option<crate::movement::MovementSchedule>,
    ) -> Result<(), String> {
        params.movement_schedule = schedule.map(std::sync::Arc::new);
        params.validate_movement()?;
        self.update_params_with_schedule_policy(params);
        Ok(())
    }

    fn update_params_with_schedule_policy(&mut self, params: EngineParams) {
        let firs_changed = !params.transferred_firs;
        let mut old = std::mem::replace(&mut self.params, params);
        if !firs_changed {
            for (new, old) in self.params.stems.iter_mut().zip(&mut old.stems) {
                new.eq_fir = std::mem::take(&mut old.eq_fir);
            }
            self.params.master.reference_fir = std::mem::take(&mut old.master.reference_fir);
            self.params.master.eq_fir = std::mem::take(&mut old.master.eq_fir);
        }
        let topology_changed = old.speakers.len() != self.params.speakers.len()
            || old.lfe_index != self.params.lfe_index
            || old.speakers.iter().map(|speaker| &speaker.name).ne(self
                .params
                .speakers
                .iter()
                .map(|speaker| &speaker.name))
            || object_topology(&old) != object_topology(&self.params);
        if topology_changed {
            let position = self.emitted;
            self.clear_route_scales();
            self.rebuild_for_new_topology();
            self.begin_seek(position);
            return;
        }

        if self.graph.routes.len() != self.params.stems.len() {
            self.rebuild_routes();
        }

        let sends_changed = old.sends != self.params.sends;
        if sends_changed {
            self.graph
                .lfe_bus
                .retune(self.sample_rate, &self.params.sends);
        }
        // A measured route scale belongs to the mix it was measured on. Every
        // input the routing reads is compared here rather than the whole
        // block, so a fader or a mastering edit — which change neither the
        // routed signals nor their weights — keeps the measurement.
        let routes_changed = routing_changed(&old, &self.params);
        let mix_table_changed = mix_routes_changed(&old, &self.params);
        let movement_changed = movement_schedule_key(&old) != movement_schedule_key(&self.params);
        if sends_changed || routes_changed {
            self.clear_route_scales();
        }
        if mix_table_changed {
            self.graph.stem_mix_routes =
                build_stem_mix_routes(&self.params, &self.graph.panner_layout);
        } else if movement_changed {
            update_movement_route_flags(&self.params, &mut self.graph.stem_mix_routes);
        }
        for (i, route) in self.graph.routes.iter_mut().enumerate() {
            let new_eq = self.params.stems.get(i).and_then(|s| s.eq.clone());
            let old_eq = old.stems.get(i).and_then(|s| s.eq.clone());
            let eq_changed = new_eq != old_eq;
            let new_dynamics = self.params.stems.get(i).and_then(|s| s.dynamics);
            let old_dynamics = old.stems.get(i).and_then(|s| s.dynamics);
            let dynamics_changed = new_dynamics != old_dynamics;
            let new_dynamic_eq = self.params.stems.get(i).and_then(|s| s.dynamic_eq.clone());
            let old_dynamic_eq = old.stems.get(i).and_then(|s| s.dynamic_eq.clone());
            let dynamic_eq_changed = new_dynamic_eq != old_dynamic_eq;
            if sends_changed || eq_changed || dynamic_eq_changed || dynamics_changed {
                route.retune(
                    self.sample_rate,
                    &self.params.sends,
                    new_eq,
                    new_dynamic_eq,
                    new_dynamics,
                    sends_changed,
                    eq_changed,
                    dynamic_eq_changed,
                    dynamics_changed,
                );
            }
            let wants_ambient = self
                .params
                .stems
                .get(i)
                .is_some_and(|s| s.wants_ambient_or_texture());
            let crossover_changed = old.stems.get(i).is_none_or(|stem| {
                stem.ambient_height_crossover_hz != self.params.stems[i].ambient_height_crossover_hz
                    || stem.ambient_height_cutoff_hz
                        != self.params.stems[i].ambient_height_cutoff_hz
                    || stem.ambient_trim_db != self.params.stems[i].ambient_trim_db
                    || stem.height_texture != self.params.stems[i].height_texture
            });
            if wants_ambient != route.has_ambient() || sends_changed || crossover_changed {
                route.set_ambient(
                    self.sample_rate,
                    &self.params.sends,
                    wants_ambient,
                    self.params.stems[i].wants_ambient(),
                    self.params.stems[i].ambient_height_crossover_hz,
                    self.params.stems[i].ambient_height_cutoff_hz,
                    self.params.stems[i].ambient_trim_db,
                    self.params.stems[i].height_texture,
                );
            }
        }

        let authored = self.graph.authored_channels;
        let speakers = self.params.speakers.len();
        // Rebuilt only when the band list actually moved: a rebuild restarts
        // every detector envelope cold.
        if !self
            .graph
            .dyn_eq
            .as_ref()
            .is_some_and(|s| s.matches(&self.params.master.dynamic_eq))
        {
            self.graph.dyn_eq = if authored > speakers {
                DynamicEq::new_linked(
                    self.sample_rate,
                    authored,
                    self.params.lfe_index,
                    speakers,
                    self.params.lfe_index,
                    &self.params.master.dynamic_eq,
                )
            } else {
                DynamicEq::new(
                    self.sample_rate,
                    speakers,
                    self.params.lfe_index,
                    &self.params.master.dynamic_eq,
                )
            };
        }
        match self.params.master.compressor {
            None => self.graph.compressor = None,
            Some(c) => match &mut self.graph.compressor {
                Some(existing) => existing.retune(c, self.sample_rate),
                None => {
                    self.graph.compressor = Some(StreamingCompressor::new(
                        c,
                        self.sample_rate,
                        if authored > speakers {
                            speakers
                        } else {
                            authored
                        },
                    ))
                }
            },
        }

        let old_unify_hz = old.master.bass.and_then(|bass| bass.unify_hz);
        let new_unify_hz = self.params.master.bass.and_then(|bass| bass.unify_hz);
        let old_unifier_active = old_unify_hz.is_some() && !old.master.lf_targets.is_empty();
        let new_unifier_active =
            new_unify_hz.is_some() && !self.params.master.lf_targets.is_empty();
        if !new_unifier_active {
            self.graph.unifier = None;
        } else if old_unifier_active && old_unify_hz == new_unify_hz {
            if let (Some(unifier), Some(bass)) = (&mut self.graph.unifier, self.params.master.bass)
            {
                unifier.retune(bass, self.params.master.lf_targets.clone());
            } else {
                self.graph.unifier =
                    build_unifier(self.sample_rate, speakers, &self.params, self.unify_done);
            }
        } else {
            self.graph.unifier =
                build_unifier(self.sample_rate, speakers, &self.params, self.unify_done);
        }
        if decorrelator_topology_changed(old.master.bass, self.params.master.bass) {
            self.graph.decorrelator =
                build_decorrelator(self.sample_rate, speakers, &self.params, self.unify_done);
        } else if let (Some(decorrelator), Some(bass)) =
            (&mut self.graph.decorrelator, self.params.master.bass)
        {
            decorrelator.retune_amount(bass.decorrelate);
        } else if self
            .params
            .master
            .bass
            .is_some_and(|bass| bass.decorrelate > 0.0)
        {
            self.graph.decorrelator =
                build_decorrelator(self.sample_rate, speakers, &self.params, self.unify_done);
            if let Some(decorrelator) = &mut self.graph.decorrelator {
                decorrelator.fade_in();
            }
        }

        if (firs_changed && old.master != self.params.master)
            || (!firs_changed && master_changed_without_firs(&old.master, &self.params.master))
        {
            for chain in &mut self.graph.causal {
                chain.retune(
                    self.sample_rate,
                    &old.master,
                    &self.params.master,
                    firs_changed,
                );
            }
        }

        match self.params.master.limiter {
            None => self.graph.limiter = None,
            Some(l) if self.graph.limiter.is_none() || old.master.limiter != Some(l) => {
                self.graph.limiter = Some(StreamingLimiter::new(
                    l,
                    self.sample_rate,
                    self.graph.post_channels(),
                    self.params.lfe_index,
                ));
            }
            Some(_) => {}
        }

        if old.speakers != self.params.speakers
            || old.output_mode != self.params.output_mode
            || old.voicing != self.params.voicing
            || old.soft_limit_threshold != self.params.soft_limit_threshold
        {
            self.graph.output.retune(self.sample_rate, &self.params);
        }

        if old.speakers != self.params.speakers
            || old.output_mode != self.params.output_mode
            || old.meter_weights != self.params.meter_weights
        {
            self.rebuild_loudness_meter();
        }
    }

    /// Install a compiled schedule atomically with its stem settings. The
    /// schedule is validated at the FFI boundary before this method is called.
    pub fn set_movement_schedule(&mut self, schedule: Option<crate::movement::MovementSchedule>) {
        if schedule.as_ref().is_some_and(|value| {
            value.validate(self.params.speakers.len()).is_err()
                || value
                    .stems
                    .iter()
                    .any(|stem| stem.stem_index >= self.params.stems.len())
        }) {
            return;
        }
        let schedule = schedule.map(std::sync::Arc::new);
        if movement_schedule_key(&self.params)
            == schedule
                .as_ref()
                .map(|value| (value.revision, value.sample_rate, value.duration_frames))
        {
            return;
        }
        self.params.movement_schedule = schedule;
        self.clear_route_scales();
        update_movement_route_flags(&self.params, &mut self.graph.stem_mix_routes);
    }

    /// Rebuild every topology-dependent stage through the same constructor
    /// as a new engine, then let the caller seek and warm its transport.
    fn rebuild_for_new_topology(&mut self) {
        self.collapsed = vec![Vec::new(); self.params.speakers.len().max(2)];
        self.graph = EngineGraph::new(
            self.sample_rate,
            &self.params,
            self.decode_taps_override.as_deref(),
            self.xtc_taps_override.as_deref(),
            self.unify_done,
        );
        self.prime_output(128);
    }

    /// Rebuild the per-stem routing state and gain smoothers to match
    /// `self.params.stems` — used when a stem was added or removed, where
    /// there is no previous per-index state to retune.
    fn rebuild_routes(&mut self) {
        let speaker_names: Vec<&str> = self
            .params
            .speakers
            .iter()
            .map(|speaker| speaker.name.as_str())
            .collect();
        self.graph.routes = self
            .params
            .stems
            .iter()
            .map(|s| build_route(self.sample_rate, &self.params.sends, s, &speaker_names))
            .collect();
        self.stem_gain = self
            .params
            .stems
            .iter()
            .map(|s| {
                let target = if s.enabled {
                    10.0_f64.powf(s.rebalance_db / 20.0)
                } else {
                    0.0
                };
                OnePole::new_at(GAIN_RAMP_MS, self.sample_rate as f64, target)
            })
            .collect();
        self.stem_route_scale = self
            .params
            .stems
            .iter()
            .map(|s| OnePole::new_at(GAIN_RAMP_MS, self.sample_rate as f64, s.route_scale))
            .collect();
        self.graph.stem_mix_routes = build_stem_mix_routes(&self.params, &self.graph.panner_layout);
    }
}

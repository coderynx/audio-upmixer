//! Fixed-filter ambient expansion for extracted rear/height sends.

use super::decorrelate::{velvet_pair_seeded, VelvetFir, VelvetLine};

/// Canonical destinations, their fixed seeds, and pair side.
///
/// The table is keyed by channel label rather than layout position. A layout
/// can therefore omit destinations or list them in any channel order without
/// changing the filter assigned to a surviving destination.
pub const AMBIENT_EXPANDER_714: [(&str, u64, usize); 8] = [
    ("SL", 12, 1),
    ("SR", 13, 0),
    ("BL", 16, 0),
    ("BR", 21, 0),
    ("TFL", 39, 0),
    ("TFR", 49, 1),
    ("TBL", 55, 0),
    ("TBR", 81, 1),
];

const SOURCES: [usize; 8] = [0, 1, 0, 1, 2, 3, 2, 3];

#[derive(Clone, Copy, Debug, PartialEq, Eq)]
struct Destination {
    name: &'static str,
    source: usize,
    seed: u64,
    side: usize,
    span: usize,
}

fn lookup_destination(name: &str) -> Option<Destination> {
    AMBIENT_EXPANDER_714
        .iter()
        .enumerate()
        .find(|(_, (candidate, _, _))| *candidate == name)
        .map(|(index, (name, seed, side))| Destination {
            name,
            source: SOURCES[index],
            seed: *seed,
            side: *side,
            span: 0,
        })
}

/// One fixed, pure-wet velvet FIR for a canonical destination.
pub fn ambient_expander_fir(sample_rate: u32, destination: &str) -> Option<VelvetFir> {
    lookup_destination(destination).map(|destination| {
        let pair = velvet_pair_seeded(sample_rate, destination.seed);
        if destination.side == 0 {
            pair.0
        } else {
            pair.1
        }
    })
}

/// Stateful fixed filtering from rear/height left/right inputs to the active
/// destinations in one speaker layout.
pub struct FixedAmbientExpander {
    sample_rate: u32,
    destinations: Vec<Destination>,
    lines: Vec<VelvetLine>,
}

impl FixedAmbientExpander {
    /// Build one stateful line for every rear/height destination in `channels`.
    /// Other channels, including front, centre, and LFE, are ignored.
    pub fn new(sample_rate: u32, channels: &[&str]) -> Self {
        let mut expander = Self {
            sample_rate,
            destinations: Vec::new(),
            lines: Vec::new(),
        };
        expander.set_destinations(channels);
        expander
    }

    /// Active destinations in the same order as the corresponding process output.
    pub fn destinations(&self) -> impl ExactSizeIterator<Item = &'static str> + '_ {
        self.destinations.iter().map(|destination| destination.name)
    }

    /// Change the active layout while retaining state for destinations that remain active.
    /// New destinations start cold; removed destinations and their FIR tails are dropped.
    pub fn set_destinations(&mut self, channels: &[&str]) {
        let requested: Vec<Destination> = channels
            .iter()
            .filter_map(|channel| lookup_destination(channel))
            .collect();
        let mut old_destinations = std::mem::take(&mut self.destinations);
        let mut old_lines = std::mem::take(&mut self.lines);
        let mut destinations = Vec::with_capacity(requested.len());
        let mut lines = Vec::with_capacity(requested.len());

        for requested in requested {
            if let Some(index) = old_destinations
                .iter()
                .position(|active| active.name == requested.name)
            {
                destinations.push(old_destinations.swap_remove(index));
                lines.push(old_lines.swap_remove(index));
            } else {
                let fir = ambient_expander_fir(self.sample_rate, requested.name)
                    .expect("canonical ambient destination");
                destinations.push(Destination {
                    span: fir.span(),
                    ..requested
                });
                lines.push(VelvetLine::new(&fir));
            }
        }
        self.destinations = destinations;
        self.lines = lines;
    }

    /// Number of input frames needed to flush the longest active FIR tail.
    pub fn span(&self) -> usize {
        self.destinations
            .iter()
            .map(|destination| destination.span)
            .max()
            .unwrap_or(0)
    }

    /// Filter rear-left, rear-right, height-left, and height-right independently.
    /// Outputs follow [`Self::destinations`].
    pub fn process(&mut self, inputs: [&[f64]; 4]) -> Vec<Vec<f64>> {
        let mut outputs = self
            .destinations
            .iter()
            .map(|destination| Vec::with_capacity(inputs[destination.source].len()))
            .collect::<Vec<_>>();
        self.process_into(inputs, &mut outputs);
        outputs
    }

    /// Allocation-free after `outputs` has enough capacity for each input.
    /// Allocate and retain these buffers outside the audio callback.
    pub fn process_into(&mut self, inputs: [&[f64]; 4], outputs: &mut [Vec<f64>]) {
        assert_eq!(
            outputs.len(),
            self.destinations.len(),
            "ambient output count must match active destinations"
        );
        for ((destination, line), output) in self
            .destinations
            .iter()
            .zip(&mut self.lines)
            .zip(outputs.iter_mut())
        {
            output.clear();
            output.extend_from_slice(inputs[destination.source]);
            line.process(output);
        }
    }

    pub fn reset(&mut self) {
        for line in &mut self.lines {
            line.reset();
        }
    }

    /// Clear FIR history at a transport seek. The expander has no absolute
    /// position state, so seeking always starts the active lines cold.
    pub fn seek(&mut self, _frame: usize) {
        self.reset();
    }
}

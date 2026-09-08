# Preview/export parity

The browser preview is a user-facing decision surface. For any DSP stage both
preview and export run, they use the shared Rust DSP core and must produce the
same audible result within the stage's tested tolerance.

Preview-only buffering, control latency, and realtime resource limits may
differ. They must not change routing, mastering, delivery, or render choices.
Run the package parity and realtime checks after changing shared DSP, preview
wiring, or exported parameter data.

Movement decisions use canonical 10 ms energies from prepared float32 PCM,
persistent mix state, and core-owned tuning. Proxy encoding, monitor Solo,
speaker mutes, repair, and mastering do not change those decisions. Ordinary
stereo bypasses movement; spatial two-channel delivery and surround downmixes
retain it.

All renderers consume the same immutable schedule on a 20 ms absolute-time
grid. The first target is immediate; later events interpolate previous-to-target
speaker gains over 5208 microseconds, then hold. Seeking uses the schedule at
the absolute programme time. Settings and their matching schedule must be
installed together before measurement; obsolete revisions cannot replace them.

Timed ADM retains Cartesian endpoints and linked-stereo pairing. Its blocks
cover the programme without gaps or overlaps. Layout adaptation happens only
at serialization: finished height channels outside the legal bed become fixed
mono objects once. QC and companion downmix render the complete programme.

Check these invariants with the movement fixtures in the shared routing parity
harness (maximum absolute error ≤1e-6), ADM timeline/layout round-trips, and
preview revision/transport tests. Lossy proxy checks compare schedules, not PCM.

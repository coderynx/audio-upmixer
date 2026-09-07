# Preview/export parity

The browser preview is a user-facing decision surface. For any DSP stage both
preview and export run, they use the shared Rust DSP core and must produce the
same audible result within the stage's tested tolerance.

Preview-only buffering, control latency, and realtime resource limits may
differ. They must not change routing, mastering, delivery, or render choices.
Run the package parity and realtime checks after changing shared DSP, preview
wiring, or exported parameter data.

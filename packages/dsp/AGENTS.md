# DSP Guide

`dsp-core` owns pure Rust DSP; `dsp-py` and `dsp-wasm` expose it. Inference,
IO, manifests, and orchestration stay in core.

Use `f64` internally. Acoustic tuning enters as parameters; structural DSP
constants stay here. A streaming change must pass the web WASM build, parity
check, and realtime benchmark. Keep `dsp_core_version()` exported by both
bindings.

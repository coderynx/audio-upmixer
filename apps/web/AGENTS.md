# Web Guide

The web app is a delivery layer. DSP lives in `packages/dsp`; preview wiring
only supplies the shared parameter block.

Use `index.css` tokens in both themes. Component colour literals are limited to
the established instrument-display and stem-identity modules. After web work,
run tests and build; after shared DSP work, also build WASM and benchmark it.

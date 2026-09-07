# DSP

The shared DSP context serves both offline processing and realtime preview.

## Language

**Parity**: The audible equivalence required where preview and export share a stage.
_Avoid_: approximation

**Realtime budget**: The processing time available to one audio callback.
_Avoid_: performance target

**Parameter block**: The complete runtime DSP settings transferred to preview.
_Avoid_: UI state

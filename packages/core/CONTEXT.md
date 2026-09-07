# Core

The processing context turns source audio into deliverable spatial audio.

## Language

**Bed**: A named output speaker layout.
_Avoid_: surround file, channel set

**Stem**: A named audio estimate used as an independently routed mix input.
_Avoid_: track, source

**Stem plan**: The requested set of stems and the dependency tree that prepares it.
_Avoid_: model list

**Source zone**: A stereo or preserved source region processed independently.
_Avoid_: input channel group

**Delivery target**: A named loudness and true-peak requirement for an output.
_Avoid_: preset

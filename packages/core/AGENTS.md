# Core Guide

Core owns library behaviour, manifests, separation, and mastering; it has no
CLI or web policy. Delivery layers use only public core APIs.

`StemSeparator` and `StemUpmixPipeline` are the separation boundary. Inference
internals stay private. Preserve the public mastering compatibility shims.

For model, ensemble, cleanup, mastering, or restoration changes, consult the
sibling `~/Projects/upmixer-knowledge/` repository and run the focused
evaluation/output checks. Report separation SDR, fullness, and bleedless
together.

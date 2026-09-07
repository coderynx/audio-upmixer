# API Guide

The API is a delivery layer over public `upmixer` APIs. It owns web state,
capability checks, and error presentation, not DSP or device control.

Keep endpoints in their feature slice. Shared infrastructure remains under
`shared/`; do not create a cross-feature route or schema grab-bag.

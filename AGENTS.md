# Repository Guide

This is a uv monorepo: `packages/core` owns Python processing and manifests,
`packages/dsp` owns shared Rust DSP, and `apps/cli`, `apps/api`, and `apps/web`
are delivery layers. Read the closest package guide before changing code.

Keep public imports stable across packages. Parameter precedence is CLI flags,
manifest values, profile defaults, then `UpmixConfig` defaults. Validate at
external boundaries; keep code and comments direct.

Run the affected package suite. Audio changes also verify output. Web changes
run `npm test` and `npm run build`; shared DSP changes also run the package
parity and realtime checks.

Read a standards document only when its subject is changed:

- ADM/BWF: `docs/standards/adm_metadata_bs2076.md`
- Atmos delivery: `docs/standards/dolby_atmos_profile.md`
- Loudness, true peak, layouts, LFE, or downmix: `docs/standards/`
- Binaural, transaural, or preview/export parity: `docs/standards/` and
  `docs/contracts/preview_export_parity.md`

Use concise Conventional Commit subjects. Pull requests state behavioural or
audio impact and validation run.

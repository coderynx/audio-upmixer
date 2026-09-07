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

## Documentation hygiene

- Treat code, tests, package manifests, schemas, and CLI help as the source of
  current behaviour. Keep prose for constraints they cannot carry.
- Keep `docs/` limited to external standards, asset provenance, and compact
  cross-package behavioural contracts. A contract states the invariant and its
  check, never an implementation tour.
- Put research, experimental evidence, plans, and historical decisions in
  `~/Projects/upmixer-knowledge/`; extend its relevant evidence ledger instead
  of adding a phase report or duplicate reference here.
- Keep agent guides as precise context pointers. Prefer the closest package
  guide or relevant `CONTEXT.md` over a new map, overview, or repeated rule.
- Keep `CONTEXT.md` files glossary-only. Record an ADR only for a hard-to-reverse,
  surprising trade-off.

## Agent skills

### Issue tracker

Issues live in this repo's GitHub Issues. See `docs/agents/issue-tracker.md`.

### Triage labels

Default canonical labels are used. See `docs/agents/triage-labels.md`.

### Domain docs

Multi-context layout. See `docs/agents/domain.md`.

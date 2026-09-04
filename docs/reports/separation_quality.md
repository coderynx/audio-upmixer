# Separation quality implementation ledger

Updated: 2026-09-05
Status: Q01 is in progress. Q00 is blocked only by the licensed-corpus and
listening gates.

This is the resumable implementation handoff for the frozen
[Q00 protocol](separation_quality_protocol.md) and its
[evaluation harness contract](../evaluation_harness.md).

## Frozen identity and scope

| Key | Value |
| --- | --- |
| Research baseline | `2877054` (`287705467f65a2bdc52b09ceffeccd4f17821548`) |
| Current code revision | `ebb6ba8` (`ebb6ba8213886a1fb3a08bd566a6e912970d9684`) |
| Protocol | `upmixer-separation-q00-v1` |
| Synthetic corpus | `upmixer-synthetic-v1` |
| Licensed split | unavailable; no real split is claimed |
| Clean baseline | `1343 passed, 38 deselected` |

The frozen synthetic corpus contains three four-second, stereo float32 items at
the generator's 44.1 kHz rate: `default`, `dense_synth`, and `choir_cluster`.
They use duplicated mono channels, unity additive mixing, no master-bus
processing, and generator seed `20260728`. Their purpose is deterministic
plumbing and category grouping; they do not substitute for musical material.
Future lawful assets must use stable recording/item IDs, hashes, explicit
provenance, and recording-group split membership. Every excerpt, alternate
rate, master, and stem from one recording group stays in one split. The
required corpus and category rules are frozen in the protocol report.

## Gate status

### Q00 — blocked

Revision `eef1bf0` froze the protocol in
[the Q00 report](separation_quality_protocol.md). No original or licensed
multitrack stems are available in this workspace, so the corpus gate remains
blocked. A qualifying future corpus needs permission and hashes, mixture and
reference provenance, aligned parent/child stems, and at least 12 tuning and
12 held-out recording groups overall, with at least three independent held-out
groups for each promoted target category. The screened categories are dense/
electronic, vocal, acoustic/tonal, drum kit, live/spatial, stereo adversarial,
temporal/context, and bandwidth/conditioning.

No listening panel or listening assets are available, so the listening gate is
also blocked. It requires blinded randomized A/B material with stable
anonymous IDs and an answer key, original-level and remixed checks, available
listener coverage, playback/randomization/repeatability records, a track-level
95% preference interval, and a defect ledger for leakage, fullness/detail,
musical noise, attacks/decay, tonality/phase, stereo image, and continuity.
Synthetic numbers cannot clear either gate.

The frozen promotion thresholds remain in the protocol: median paired SDR
improvement at least `0.2 dB` or fullness/bleedless at least `0.01`, with the
track-bootstrap 95% lower bound above zero; no supported co-metric loss beyond
`0.1 dB` SDR or `0.005` fullness/bleedless; flag individual losses above
`0.5 dB` SDR or `0.02` fullness/bleedless; default added RTF at most `0.1` and
job-time increase at most `10%`; optional extra-view time at most `2x` and
within the measured device memory ceiling.

### Q01 — in progress

Q01 remains evaluation plumbing only. The slices currently present are:

| Slice | Revision | Delivered behavior |
| --- | --- | --- |
| Q01a | `ed73831` | Stable recording/item/split identities; strict required-estimate, rate, shape, channel, length, finite-value, and settings validation; public metric-helper truncation preserved. |
| Q01b | `55e4fa6` | Actual model metadata and TTA/pitch settings are recorded while legacy report formatting and public imports remain compatible. |
| Q01c | `98ed432` | Explicit unavailable references and identity-bearing coverage rows; manifest parsing and unavailable-only items report coverage without fake metric rows; required/scored failures remain strict. |
| Q01d | `6aadb85` + correctness follow-up `ebb6ba8` | Recording means and fixed-seed paired bootstrap over recording groups, with split-aware identities/groups, unavailable coverage, and per-group `n_recordings`, status, and `None` CI bounds. Later Q01 serialization, diagnostics, and runner work remains. |

Q01 source and focused test files are
`packages/core/src/eval/{__init__,corpus,harness,report}.py` and
`packages/core/tests/{test_eval_boundaries,test_eval_harness,test_eval_report}.py`.
The handoff evidence is:

- Q01a focused boundary/metric checks: `24 passed`; full core: `1204 passed,
  38 deselected`.
- Through Q01c: eval slice `30 passed, 1 deselected`; full core
  `1210 passed, 38 deselected`; Ruff and `git diff --check` passed.
- Q01d-focused check after the correctness follow-up: `40 passed, 1 deselected`.
  Full core after Q01d has not been rerun in this handoff.

No production separation algorithm, default, cache, store, or cross-package
API was changed. Existing constructors retain compatible defaults, and the
strict boundary stays at `evaluate_corpus`; rollback is by reverting the
individual eval slice commits.

## Recorded synthetic smoke

The recorded handoff invocation was:

```bash
uv run pytest packages/core/tests -m perf -k eval -s
```

It reported `1 passed, 1247 deselected` in `41.12s` on MPS using
`BS-Roformer-SW.ckpt` at 44.1 kHz. TTA and pitch shift were off; the effective
model-native rate was 44.1 kHz. The reported means were:

| Group | SDR (dB) | Fullness | Bleedless |
| --- | ---: | ---: | ---: |
| Bass | 1.87 | 0.661 | 0.165 |
| Drums | -0.00 | 0.000 | 0.438 |
| Other | 0.30 | 0.521 | 0.874 |
| Vocals | 0.00 | 0.000 | 0.379 |
| `choir_cluster` | 0.99 | 0.185 | 0.505 |
| `default` | 0.82 | 0.215 | 0.415 |
| `dense_synth` | -1.25 | 0.499 | 0.846 |

These are synthetic harness metrics only. They provide no musical-quality,
held-out ranking, promotion, or listener result. Peak memory was not measured;
no listening result is available.

## Q10 disposition

The Q10 scout is **ready after Q01/Q02/Q03 plumbing**. Q10 has no implementation
or quality result yet. Its prerequisite is a completed Q03 full-tree baseline
with the licensed corpus and listening evidence above; until then Q10 cannot
claim an optimization or a quality change.

## Next eligible work and restart

1. Finish and validate the remaining Q01 slices, including serialization,
   diagnostics, and the corpus/variant runner, then record their exact command
   and results.
2. Run Q02 deterministic incumbent/candidate regression checks against the Q01
   interfaces.
3. Run Q03 full-tree baseline and listening/defect review once the two Q00
   gates are supplied; keep the status blocked if either is still missing.
4. Start Q10 only after Q03, with duplicate-forward equivalence plus measured
   runtime and peak-memory evidence.

Resume from the shared branch and preserve any in-progress eval work:

```bash
git switch feature/improve-stem-separation
git log --oneline --decorate -6
git status --short
uv run pytest packages/core/tests/test_eval_boundaries.py packages/core/tests/test_eval_harness.py packages/core/tests/test_eval_metrics.py packages/core/tests/test_eval_report.py -q
uv run pytest packages/core/tests -m perf -k eval -s
```

Run `uv run pytest packages/core/tests -q` after the pending Q01 slices are
green, then repeat the protocol smoke and record the actual settings, metrics,
runtime, and peak memory in a new dated report.

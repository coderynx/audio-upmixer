# Separation quality Q00 protocol

Frozen: 2026-09-04  
Status: synthetic/plumbing protocol frozen; licensed-corpus and listening gates
are blocked.

| Identifier | Value |
| --- | --- |
| Protocol | `upmixer-separation-q00-v1` |
| Synthetic corpus | `upmixer-synthetic-v1` |

These IDs are stable report keys. A changed generator, membership, split rule,
or acceptance rule gets a new version; generated paths, hosts, and timestamps
do not change an ID. No licensed audio or private absolute path is committed.

## What is frozen now

The current synthetic source is
[`synthetic_corpus`](../../packages/core/src/eval/corpus.py), with the canonical
metric definitions and harness controls in the
[evaluation harness](../evaluation_harness.md). Every report carries SDR,
fullness, and bleedless together, grouped by stem and regression category.
Real-model evaluation uses the model-native 44.1 kHz rate and records the
effective model, sample rate, segment size, overlap, batch size, and ensemble
settings. Metric-only plumbing may use another explicitly recorded rate.

The generated corpus has three four-second items. Each item is stereo float32
WAV with duplicated mono channels, unity additive mixing, and no master-bus
processing. The fixed generator seed is `20260728`; the generator code remains
the source of truth for exact samples.

| Stable item key | Category | Reference stems | Probe content |
| --- | --- | --- | --- |
| `upmixer-synthetic-v1/default` | `default` | `Vocals`, `Bass`, `Drums`, `Other` | 220 Hz vocal, 55 Hz bass, half-second noise hits, and noise Other |
| `upmixer-synthetic-v1/dense_synth` | `dense_synth` | `Vocals`, `Other` | 12 overlapping detuned partials with a 330 Hz vocal |
| `upmixer-synthetic-v1/choir_cluster` | `choir_cluster` | `Vocals`, `Other` | five near-unison detuned 246 Hz voices with noise Other |

Each mixture is the known sum of its references. These items exercise lawful,
deterministic harness plumbing and category grouping; they do not establish
musical model rankings or a quality claim. Keep real-model clips comfortably
above the documented approximately three-second short-input floor.

The reproducible checks frozen for this stage are:

```bash
uv run pytest packages/core/tests/test_eval_metrics.py -q
uv run pytest packages/core/tests -m perf -k eval -s
```

The first command is offline metric and harness plumbing. The second is a
separately marked real-model smoke run that may download weights; it is not a
held-out music-quality gate. Until the Q01 runner exists, these tests are the
shareable synthetic invocation; no real-corpus result is implied.

## Future corpus, provenance, and splits

The real corpus remains an access-controlled local directory. When lawful
assets arrive, commit only a versioned `corpus.json`, an adjacent provenance
and split manifest, and shareable derived reports. Use stable `recording_id` and
`item_id` values, relative manifest paths, and hashes; never use private
absolute paths as identity.

Each recording/item entry must include:

- source and license/use permission, file hashes, original recording/group
  identity, and known overlap with pretrained or benchmark material (or an
  explicit unknown-overlap value);
- mixture construction, common gain, master-bus processing, and any residual
  when the mixture does not equal the reference sum;
- sample rate, channels, duration, aligned canonical stem names, and available
  parent/child references; and
- split and category tags.

Use independent recording groups for tuning and holdout. Keep every excerpt,
rate conversion, alternate master, and stem from one recording group in the
same split. Freeze hashes and split membership before tuning. Screen the
required failure categories from the plan: dense/electronic, vocal,
acoustic/tonal, drum kit, live/spatial, stereo adversarial, temporal/context,
and bandwidth/conditioning. Start with at least 12 tuning and 12 held-out
recording groups overall, with at least three independent holdout groups for
each promoted target category; expand when uncertainty requires it.

Unavailable child references are recorded as unavailable rather than silently
excluded. Kit, karaoke, crowd, or similar scores then have limited claims;
known-source controls may exercise plumbing, but real child references and
listening are required before promotion for that child.

## Gates still blocked

No original or licensed multitrack stems are available in this workspace.
Therefore the following are not claimed: a licensed `corpus.json`, provenance
and split manifests for real recordings, tuning/holdout measurements, a
baseline on musical material, or category-specific child-reference coverage.

No listening panel or listening assets are available. The missing gate is a
blinded randomized A/B set with stable anonymous IDs and a separate answer
key; solos, subtraction residuals, and modest ±3/6 dB remixes; common
comparison gain plus an original-level check; at least three listeners where
available; playback conditions, randomization seed, ties, repeatability, and
uncertainty; track-level aggregation with a 95% preference interval; and a
defect ledger covering leakage, detail/fullness, musical noise, attacks/decay,
tonality/phase, stereo image, and continuity.

Until both gates are supplied, synthetic results may validate determinism,
metric plumbing, and exact-transfer behavior only. They cannot promote a
quality feature, change a default, rank models for music, or support an audible
improvement claim.

## Frozen acceptance thresholds

Compare paired deltas with the relevant incumbent, holding all other settings
fixed. Before viewing holdout results, declare target stems/categories, primary
outcome, and runtime class. These starting thresholds may change only with a
written rationale based on baseline repeatability and tuning material.

| Gate | Requirement |
| --- | --- |
| Deterministic numerical equivalence | Float32 waveform comparison with `atol=1e-6`, `rtol=1e-5`; calibrate backend repeatability separately |
| Exact transfer/conservation | Normalized float test signals: maximum absolute sum error `<= 1e-6`; non-silent real audio: residual energy `<= -100 dB` relative to parent where exact closure is promised |
| Length/shape/level | Exact declared frame count, finite stereo float32, correct rate, zero integer-sample offset on resampler impulse controls, and gain restored within calibrated resampler tolerance |
| Material improvement | On the predeclared target set, median paired SDR improvement `>= 0.2 dB` or fullness/bleedless improvement `>= 0.01`, with the track-bootstrap 95% CI lower bound above zero for that outcome |
| Other co-reported metrics | No statistically supported loss beyond `0.1 dB` SDR or `0.005` fullness/bleedless on target or protected categories |
| Individual regression screen | Flag any track/stem loss `> 0.5 dB` SDR or `> 0.02` fullness/bleedless for examination and listening |
| Perception | Blinded held-out preference evidence for the claimed benefit, with no repeatable new stereo, transient, or detail defect; an improved null or SDR cannot override an audible regression |
| Default-path DSP/resampling | Added RTF `<= 0.1` and total job-time increase `<= 10%` on each supported target device |
| Optional extra-view mode | First experiment uses at most one additional view per selected stage; provisional complete-job time is `<= 2x` incumbent, including required descendants |

Freeze a numerical peak-memory ceiling per device from usable memory and the
incumbent's measured peak. A default path may not exceed that peak by more than
10% without a revised budget. Optional modes stay within the device ceiling
without swapping or freezing; backend tuning and OOM fallback are recorded as
variant settings. A missing device leaves its promotion gate open. Q10 is an
optimization gate: require equivalent audio with lower evaluated-window work
and measured non-regressing time/memory rather than a quality gain.

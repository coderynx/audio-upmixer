# Separation quality implementation ledger

Updated: 2026-09-05
Status: Q01/Q02 plumbing and regression work is complete; Q03 is partial; Q10
passes its objective equivalence and memory gate. Q00 promotion remains open
for category coverage and listening evidence.

This is the resumable handoff for the frozen
[Q00 protocol](separation_quality_protocol.md) and its
[evaluation harness contract](../evaluation_harness.md).

## Frozen identity and current corpus

| Key | Value |
| --- | --- |
| Research baseline | `287705467f65a2bdc52b09ceffeccd4f17821548` |
| Assigned code revision | `b85880e23160691d2a593d772bd014a1feacd3ba` |
| Protocol | `upmixer-separation-q00-v1` |
| Synthetic corpus | `upmixer-synthetic-v1` |
| Selected real corpus | `upmixer-musdb18hq-v1-c5ba6b34513f` |
| Real corpus content hash | `c5ba6b34513f09632e33c9334de8a5849d0cd8181d20cba4507e7632493f0237` |
| Real split | 12 tuning and 12 heldout distinct recordings/artists; 120 WAVs; 44.1 kHz stereo |
| Archive identity | 22,656,664,047 bytes; MD5 `12d4f2ecd55245a4688754dd76363103` |
| Local corpus bytes | 5,086,843,660 |
| Use | Restricted educational/research use only; no commercial or redistribution permission is claimed |
| Model overlap | SCNet: none by the known list; other checkpoints: unknown |
| Aggregate `Other` mapping | `Guitar + Piano + Other` under `b85880e` |

The real corpus is external at `$UPMIXER_MUSDB_ROOT`; no audio is committed.
Its 12/12 split is enough for a broad baseline, but the required category
coverage and listening/promotion gates remain open. The synthetic corpus still
contains three four-second stereo float32 items at 44.1 kHz (`default`,
`dense_synth`, and `choir_cluster`) for deterministic plumbing only.

## Gate status

### Q00 — corpus available; promotion open

The selected official MUSDB18-HQ test-subset corpus is available under the
restricted research terms above. It meets the broad 12 tuning / 12 heldout
recording-group minimum. The current category assignment does not yet provide
the required independent heldout coverage for every promoted target category,
and no full 24-recording baseline is claimed.

The listening gate is still open: no human listening panel or listening assets
are available. Promotion therefore remains blocked even where objective
measurements exist. Do not treat the corpus as a full Q00 promotion.

The frozen thresholds remain in the
[protocol](separation_quality_protocol.md): paired SDR improvement of at least
`0.2 dB` or fullness/bleedless improvement of at least `0.01`, bootstrap lower
bound above zero, no supported co-metric loss beyond `0.1 dB` SDR or `0.005`
fullness/bleedless, default job-time increase at most `10%`, and the measured
device memory ceiling.

### Q01/Q02 — complete plumbing and regression work

Q01 added strict corpus/report validation, unavailable coverage, paired
recording-group statistics, serialization, the offline runner, production-tree
evaluation, and execution provenance. Its relevant history runs from
`ed73831` through the tree/provenance fixes ending at `b85880e`.

Q02 added deterministic schedule, rate/level, plan, zone, persistence, and
retry regression coverage (`ceb90da`, `c2866f3`, `f600c86`, `3518c3d`,
`b3f3d71`, `d98b264`, and `04d1c5e`). These are plumbing and regression checks;
they do not establish a separation-quality improvement or change a default.

The final Python suite was:

```text
uv run pytest packages/core/tests apps/api/tests apps/cli/tests -q
1552 passed, 38 deselected, 24 warnings in 40.66s
```

### Q03 — partial baseline

One full tuning track, `Hollow Ground - Ill Fate`, has a valid current
production-tree report. All four requested references scored:

| Stem/category | SDR (dB) | Fullness | Bleedless |
| --- | ---: | ---: | ---: |
| Bass | 2.65 | 0.340 | 0.910 |
| Drums | 8.63 | 0.672 | 0.915 |
| Other (aggregate) | 6.96 | 0.815 | 0.881 |
| Vocals | 8.89 | 0.751 | 0.902 |
| Category aggregate | 6.78 | 0.645 | 0.902 |

This is a one-track tuning result, with no human listening panel and no full
24-track run. Q03 is therefore partial and cannot close the category,
listening, or Q00 promotion gates.

### Q10 — implemented; objective audit passes

`168ed7b` reuses identical Roformer tail windows. The full comparison is in the
external artifact
`$UPMIXER_EVAL_ROOT/q10/full/Hollow-Ground-Ill-Fate/equivalence.json`, with
`UPMIXER_EVAL_ROOT` set to the local evaluation root.

| Measure | Old `a6feaab` | Current `b85880e` | Delta |
| --- | ---: | ---: | ---: |
| Cold runtime | 188.33 s | 193.78 s | +2.894% |
| Peak RSS | 3,096,068,096 B | 3,314,614,272 B | +7.059% |
| Memory footprint | 9,685,011,840 B | 9,787,674,728 B | +1.060% |

All six decoded float32 arrays are exact on MPS (`max diff = 0`). The audio
equivalence and `<10%` memory gates pass. Tests demonstrate less evaluated
duplicate tail work. The audited full-track configs reduce evaluated windows
from 18 to 17 for Deux and from 15 to 14 for SW, 33 to 31 total (`6.061%`).
The single cold runtime is within measurement noise, so no speed claim is made.
Intermediate callback cadence changes, while final completion is preserved.
CPU and CUDA real-model backend gates remain open.

The old report's raw `Other` metric used the wrong taxonomy. The current
`b85880e` mapping (`Guitar + Piano + Other`) is the valid comparison.

## Restart commands

Keep private audio and generated reports outside the repository:

```bash
export UPMIXER_MUSDB_ROOT=/path/to/selected-musdb18hq
export UPMIXER_EVAL_ROOT=/path/to/upmixer-eval/separation-quality

git switch feature/improve-stem-separation
git log --oneline --decorate -6
git status --short

uv run python scripts/prepare_musdb18hq_corpus.py \
  --dataset-root "$UPMIXER_MUSDB_ROOT" \
  --output-dir "$UPMIXER_MUSDB_ROOT"
uv run python scripts/run_eval.py \
  --corpus "$UPMIXER_MUSDB_ROOT" \
  --variant production-tree --sample-rate 44100 \
  --stems vocals,bass,drums,guitar,piano,other --retain-stems \
  --output-dir "$UPMIXER_EVAL_ROOT/q03/full/current"
```

The Q10 audit artifact is read from
`$UPMIXER_EVAL_ROOT/q10/full/Hollow-Ground-Ill-Fate/equivalence.json`.
Q20 is the next eligible experiment; retain the open category, listening, and
promotion gates until its controls and the missing Q03 evidence are complete.

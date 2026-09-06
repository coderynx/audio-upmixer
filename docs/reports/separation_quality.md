# Separation quality implementation ledger

Updated: 2026-09-06. Current branch: `feature/improve-stem-separation`.

This is the resumable handoff for the frozen
[Q00 protocol](separation_quality_protocol.md) and the
[evaluation harness contract](../evaluation_harness.md).

Current status: Q01/Q02 plumbing and regression work is complete; Q03 has a
12-recording tuning baseline; Q10 passes its objective equivalence and memory
gate; Q20 remains partial, while the Q21 native-rate candidate passed its
automatic production parity gate and was accepted after the owner's 2026-09-06
quick listening OK; Q21 remains opt-in/default-off. Q30 and Q40 experiments
are complete but their candidates are rejected for production; both Q50
candidates (aggregate-Drums v1 and kit-sibling v2) are complete and rejected
after automatic failures, and their experimental helpers have been removed.
The Q90 pitch-adviser preflight for the Q40/Q50 ambiguity is complete and
rejected; Q60/Q80 are ineligible, Q70 is blocked, and Q100 for accepted Q21 is
next. Q31 and later quality changes have not started.
No production default has changed.

## Progress/stage

| Stage | Status | Next eligible work |
| --- | --- | --- |
| Q00 | Available; promotion open | Finish category, listening, and heldout gates. |
| Q01/Q02 | Complete | — |
| Q03 | Partial | — |
| Q10 | Complete; objective gate passes | — |
| Q20 | Partial; native-rate automatic gate passed | Finish residual and heldout evidence as needed for later promotion. |
| Q21 | Accepted after automatic gate and quick owner OK; opt-in/default-off | Q50 was evaluated next and rejected; default promotion remains separate. |
| Q30 | Complete; rejected | No Q31; Q50 was evaluated next and rejected under the sequential candidate workflow. |
| Q40 | Complete; rejected | No Q41. Q50 is complete/rejected for both candidate arms; Q60/Q80 lack prerequisites; Q70 is blocked; Q90 requires its follow-up preflight. |
| Q50 | Complete; aggregate v1 and kit v2 rejected | No Q51. Q60/Q80 lack prerequisites; Q70 is blocked; Q90 may be considered only if the Q40/Q50 result is treated as a named ambiguity plus classical-cue failure and adviser provenance preflight passes. |
| Q60 | Ineligible; no measured spatial/residual defect | No Q60 experiment. |
| Q70 | Blocked; no accepted magnitude correction | Reopen only after an accepted correction. |
| Q80 | Ineligible; no measured Q03 quiet-input failure | No Q80 experiment. |
| Q90 | Preflight complete; pitch advisers rejected for the Q40/Q50 ambiguity | No Q90 implementation. Q100 for accepted Q21 is next. |
| Q100 | Next for accepted Q21 | Complete delivery, parity, and rollback evidence before promotion. |

## Frozen identity and corpus

| Key | Value |
| --- | --- |
| Research baseline | `287705467f65a2bdc52b09ceffeccd4f17821548` |
| Evaluated implementation revision | `5ff2524ff2df6ccf54c4c06a95db2e5f0b57ad3f` |
| Q03/Q20 matrix revision | `7a2806cf5fa7f0d8a84af3b8139b92c68a8c242b` |
| Protocol | `upmixer-separation-q00-v1` |
| Synthetic corpus | `upmixer-synthetic-v1`; deterministic harness checks only |
| Selected real corpus | `upmixer-musdb18hq-v1-c5ba6b34513f` |
| Real corpus content hash | `c5ba6b34513f09632e33c9334de8a5849d0cd8181d20cba4507e7632493f0237` |
| Real split | Official test subset; 12 tuning and 12 heldout recordings/artists; 120 WAVs; 44.1 kHz stereo |
| Aggregate reference mapping | `Other = Guitar + Piano + Other` |
| Use | Restricted educational/research use; no commercial or redistribution permission claimed |
| Model overlap | No known SCNet training overlap; overlap unknown for the other checkpoints |

The real corpus and all generated audio are external to the repository under
`$UPMIXER_MUSDB_ROOT`. The synthetic corpus is not musical-quality evidence.

The fixed Q03/Q20 excerpt corpus is 60–72 seconds from each selected recording,
with source start frame 2,646,000 and 529,200 source frames. It contains 12
tuning and 12 heldout recordings at 44.1, 48, and 96 kHz. The selected FIR is
`upmixer.resample.resample_channels:120dB-kaiser-fir`; target frames use
`round(source_frames * target_rate / source_rate)`, then pad and trim. Corpus
construction verified 360 audio files, finite stereo float32 samples, frame
counts, source hashes, generated hashes, and 44.1 kHz mixture/reference sums.
Heldout inference has not run.

The excerpt helper hash is
`ab7bdb5b645b8fff4acfa63afc23d8cb1fd25c8e32700a618810b0818c96c875`; the
generated `SHA256SUMS` hash is
`734c6ab7ca2d393a1ee6d148980c2ab46946e8582555ca68c801388e73902bfc`.

| Variant | Corpus manifest SHA-256 | Provenance SHA-256 |
| --- | --- | --- |
| `44.1k/tuning` | `7a87278b1cdb59f2550b07c10b18a2944d0f9ce5b4bd5d60bd55898cc97dda28` | `e1c23a0653655dae493085fdfaa1267d9045facc74a76b1c324e435bf50c3eff` |
| `44.1k/heldout` | `a8fd0af00ab3d4e6360bc7b307a60be29030697ab6a3a565d1fb721fd13b3689` | `4d8eea0b30517080b11d96b7c45015397702856640332eb86618fcd6f223a4d0` |
| `48k/tuning` | `283bf470275215a77eeb31c8a6b485fd9bf8c8808c6e8b38ae887c89329958ac` | `439dc92210cfb417d84e117fd3683f1819c22fa32e8020843e87d8045fa76732` |
| `48k/heldout` | `b3afb7ed743183c233ef82a14dfe5f89a67e7f2203a770f17c6f17b6fc6f1d73` | `11bb125cfd03e6d7146454d31b06cf77d856a881c1c760cb4da603367ccad72a` |
| `96k/tuning` | `6120d9b7039a8e1aabf2984a28f2b7bb09e836d0561376a08a2814f1eb03aec6` | `2a834a82651113de191da38d469493185dd10b78b47cda5430a733ec9c3f1d55` |
| `96k/heldout` | `464e1cc096f2024e6bddbb726d382e9155be261b89987534869f21ce7adacfcb` | `18fdff7b200f05e27ee330787dc7d52f7bc84eb310c8ae384693115a62b73632` |

## Selected supplementary data

All selections are stored on the external SSD under `$UPMIXER_DATASET_ROOT`.
They are tuning, component, transfer, or bandwidth assets and do not enlarge
the heldout clean-music corpus.

| Selection | What is present | Evidence and license boundary |
| --- | --- | --- |
| MUSDB18-HQ | Official test subset, 24 recordings, four references plus mixture | 44.1 kHz broad baseline; educational/research only; `Other` is aggregate and exact child references are unavailable |
| Freischütz Digital | 14 selected stereo WAVs, 48 kHz/24-bit: three final mixes and available singer/string/horn/woodwind group references | AudioLabs and Zenodo license metadata conflict; apply stricter CC BY-NC-SA 4.0 interpretation: internal research/listening, attribution/share-alike, no redistribution. Source pages warn of microphone bleed, so no clean additive claim without closure evidence |
| Celtic Constellation preview | Two full mixes and two aligned dry excerpts, 96 kHz/24-bit | CC BY-NC 4.0, research/listening only, no commercial use or redistribution. Dry excerpts are about 22 s while mixes are about 148 s and cannot reconstruct them |
| IDMT-SMT-Drums selection | One `WaveDrum02_01` mix with Kick, Snare, and Hi-Hat components, 44.1 kHz mono, 17.07 s | CC BY-NC-ND 4.0; attribution/non-commercial/no derivatives. Controlled sample-library recording, no Toms/Ride/Crash, and no heldout item |
| MedleyDB sample | `LizNelson_Rainfall` mix, three singer candidates, two guitar stems, raw files and metadata | CC BY-NC-SA 4.0; research-only, attribution/share-alike, no republishing. No lead/backing annotation, no crowd/drum descendants, and no heldout item |
| CrowdioSet selection | Two train CC0 files: one 267.64 s ambience and one 11.50 s event | Selected rows are CC0 1.0 with source metadata retained. Standalone crowd recordings support transfer/mixture checks; they are not aligned multitrack music and have no heldout item |

## Gate status

### Q00 — available, promotion open

The selected MUSDB18-HQ subset clears access and broad corpus-size gates, but
category coverage is not sufficient for every promoted target. No full
24-recording baseline, heldout result, or human listening panel is claimed.
Three blinded listening packs exist, but they are tuning-only and unrated.
Promotion and any default change remain open.

### Q01/Q02 — complete plumbing and regression work

The harness now validates corpus/report coverage, unavailable references,
paired recording-group statistics, serialization, production-tree execution,
rate/level/plan/zone/persistence/retry behavior, and execution provenance.
The Q01/Q02 baseline at code revision `8683132` reported
`1584 passed, 38 deselected, 24 warnings`; Q40 validation at `5ff2524` is
recorded below.

### Q03 — 12-recording tuning baseline, partial

Artifact:
`$UPMIXER_EVAL_ROOT/q03/excerpts-60s-72s/tuning-44k-7a2806c/report.json`

Report SHA-256:
`4a234e9132b4254c1fb28c4d12f01ccb6bc6b7cf8411e5d0ac315b71de490368`.
The MPS production tree scored 48 rows across 12 fixed 12-second tuning
excerpts. Means are:

| Stem/category | SDR (dB) | Fullness | Bleedless |
| --- | ---: | ---: | ---: |
| Bass | 9.6961 | 0.73590 | 0.89857 |
| Drums | 13.0912 | 0.82385 | 0.93182 |
| Other (aggregate) | 8.1954 | 0.71899 | 0.91221 |
| Vocals | 13.3138 | 0.84893 | 0.90449 |
| Category aggregate | 11.0741 | 0.78192 | 0.91177 |

Guitar and Piano remain unavailable as separate MUSDB references. Q03 cannot
close category, listening, or Q00 promotion gates.

### Q10 — objective optimization gate passes

`168ed7b` reuses identical Roformer tail windows. The full comparison is at
`$UPMIXER_EVAL_ROOT/q10/full/Hollow-Ground-Ill-Fate/equivalence.json`.
All six decoded MPS float32 arrays were exact (`max diff = 0`); peak memory
rose 7.059%, within the 10% gate. Audited configs reduced evaluated windows
from 33 to 31 (6.061%). The single cold runtime is within measurement noise,
so no speed claim is made. CPU and CUDA real-model checks remain open.

### Q20 — corrected rate experiment, partial

The Q20 implementation and provenance commits are `7d52093`, `7a2806c`,
`0186944`, `fa63413`, `62c67d2`, `5e3eb59`, and `8683132`. `--rate-arm`
remains opt-in evaluation behavior. `delivery` reproduces the incumbent
requested-rate condition; `native` runs the tree at the model's 44.1 kHz rate
and converts once to delivery with the selected 120 dB FIR. Reports record
the arm, input/separation/output rates and frame counts, resampler identity,
stage settings, checkpoint/config hashes, and MPS device.

The production-tree experiment now requests all public terminal outputs and
unconsumed private complement leaves for Q20 only. Consumed parents are
excluded from the terminal set. Private runs bypass supplied/public cache and
resume artifacts so requested-only state cannot contaminate the experiment;
the normal requested-only behavior is unchanged. The retained `stems/index.json`
maps each item to its relative stem files, recording/item IDs, split/category,
sample rate, and rate provenance.

The corrected 12-recording matrix has 48 scored rows per arm (Bass, Drums,
aggregate Other, Vocals); Guitar and Piano arrays are retained in the index but
cannot be scored separately. The exact reports and hashes are:

| Artifact | SHA-256 |
| --- | --- |
| `$UPMIXER_EVAL_ROOT/q20/tuning-12x12s/comparison.json` | `1da5df3e31e416259a840b5f1fb11d8cd88452a4f02f2b73894d723dfeb67716` |
| `$UPMIXER_EVAL_ROOT/q20/tuning-12x12s/compare_tuning_reports.py` | `6b36da75af4ec72bfbba361a83fcac162ce5f8157b88d0d19e2106f4128d7a22` |
| `$UPMIXER_EVAL_ROOT/q20/tuning-12x12s/48k-delivery/report.json` | `87f95f4cb5e6c2a97f7b48a2919fee3193bba36006e77bc68862f4ab6415eada` |
| `$UPMIXER_EVAL_ROOT/q20/tuning-12x12s/48k-native/report.json` | `fca2047e66488b76d17a67830fe97407d5ef297e1746a4a0e00e25ec6456393b` |
| `$UPMIXER_EVAL_ROOT/q20/tuning-12x12s/96k-delivery/report.json` | `3439ea506dc470b26fa40629a09b2797551e63e9123a58234b3e6e6d8d20cff7` |
| `$UPMIXER_EVAL_ROOT/q20/tuning-12x12s/96k-native/report.json` | `8f023326e0fb91fae2b815faeb41b34ee13aa22fbe32364483ce84e0ee18ca1a` |
| `$UPMIXER_EVAL_ROOT/q20/tuning-12x12s/SHA256SUMS` | `8c9bd34947eedf95e709e9a95d37968e2634f3b725472c7ac69df92d95f94c5e` |

Paired bootstrap deltas are native minus delivery, with 10,000 resamples and
12 recordings:

| Delivery rate | SDR (dB), 95% CI | Fullness, 95% CI | Bleedless, 95% CI | Individual flags |
| --- | --- | --- | --- | ---: |
| 48 kHz | `+0.468 [0.272, 0.680]` | `+0.00889 [0.00368, 0.01543]` | `+0.00280 [0.00075, 0.00540]` | 3 |
| 96 kHz | `+2.677 [1.587, 3.859]` | `+0.04945 [0.00225, 0.09493]` | `+0.05936 [0.02826, 0.09311]` | 11 |

The 96 kHz delivery arm is a different high-rate model condition; the native
96 kHz arm is the same 44.1 kHz separation as native 48 kHz followed by a
different output conversion. These numbers are tuning signals only and do
not support promotion.

On this MPS run, native 48 kHz took 168.29 s versus 165.23 s for delivery
(+1.9%) with a roughly +1.9% RSS change; native 96 kHz took 177.04 s versus
359.83 s (-50.8%) with roughly -6.2% RSS. The provisional runtime screens
pass, but no formal device memory ceiling has been frozen.

The individual regression screen uses the frozen `>0.5 dB` SDR or `>0.02`
fullness/bleedless thresholds. These entries require blinded listening review:

| Rate | Stem | Recording | Flag |
| --- | --- | --- | --- |
| 48 kHz | Other | Girls Under Glass — We Feel Alright | fullness `-0.0295` |
| 48 kHz | Bass | Lyndsey Ollard — Catching Up | bleedless `-0.0201` |
| 48 kHz | Other | Side Effects Project — Sing With Me | fullness `-0.0221` |
| 96 kHz | Bass | Girls Under Glass — We Feel Alright | fullness `-0.0850` |
| 96 kHz | Bass | Hollow Ground — Ill Fate | fullness `-0.7839` |
| 96 kHz | Drums | Hollow Ground — Ill Fate | fullness `-0.0507` |
| 96 kHz | Other | Hollow Ground — Ill Fate | bleedless `-0.1222` |
| 96 kHz | Bass | Lyndsey Ollard — Catching Up | fullness `-0.0560` |
| 96 kHz | Drums | Side Effects Project — Sing With Me | SDR `-2.1501`; bleedless `-0.0641` |
| 96 kHz | Other | Side Effects Project — Sing With Me | SDR `-3.2101`; fullness `-0.4305` |
| 96 kHz | Other | The Easton Ellises — Falcon 69 | SDR `-0.7148`; fullness `-0.0388` |
| 96 kHz | Vocals | The Easton Ellises — Falcon 69 | SDR `-2.4204`; bleedless `-0.1159` |
| 96 kHz | Bass | The Mountaineering Club — Mallory | fullness `-0.0427` |
| 96 kHz | Bass | We Fell From The Sky — Not You | bleedless `-0.0923` |

No human listening panel or defect ledger has been completed. The listening
packs below are prepared but unrated; a flag is a triage item, not a listening
result.

The corrected branch-smoke matrix is at
`$UPMIXER_EVAL_ROOT/q20/branch-smokes-8683132`, with analysis SHA-256
`25b0b730c4d2c22e8f590ebbb90a4959b0014bc65ed80f4808a962f125507a48`.
It covers five corrected tuning cases and nine scored rows per arm: the AM
Contra and crowd-transfer items use 60–72 s, DrumSep uses WaveDrum02_01 at
0–12 s, and the role proxy uses LizNelson_Rainfall at 220–232 s. Native minus
delivery deltas are shown as SDR dB / fullness / bleedless:

| Case | Scored deltas |
| --- | --- |
| `crowd-transfer` | Crowd `+1.252412 / +0.000784 / +0.027710` |
| `drumsep-wavedrum02-01` | Hi-Hat `+0.176098 / -0.010573 / -0.001513`; Kick `+0.576593 / +0.026243 / -0.005729`; Snare `-0.222599 / +0.004664 / -0.058382` |
| `ensemble-bass-drums` | Bass `+0.334726 / -0.011054 / -0.000821`; Drums `+0.956146 / +0.008842 / +0.001426` |
| `lead-backing-role-proxy` | Backing Vocals `+0.050265 / -0.003361 / +0.001206`; Lead Vocals `+0.394745 / +0.001489 / +0.001253` |
| `vocals-only` | Vocals `+0.552196 / +0.001663 / -0.002379` |

The prior frame-0 artifacts are quarantined under
`analysis-zero-to-12-invalid/`, `crowd-transfer-zero-to-12-invalid/`,
`ensemble-bass-drums-zero-to-12-invalid/`, and
`vocals-only-zero-to-12-invalid/`, each with `INVALID.json`; the contaminated
overlap rerun is also quarantined under
`vocals-only-contaminated-overlap-invalid/`. None is used for conclusions.
Crowd transfer, DrumSep, and lead/backing are artificial or controlled role
proxies, so proxy/synthetic evidence is not clean musical evidence. The
vocals and ensemble cases remain tuning-only short excerpts.

The context pilot is at
`$UPMIXER_EVAL_ROOT/q20/context-pilot-8683132`, with analysis SHA-256
`31f40a8eb7bff57798ab1b46616098b97dc46dd70de0866d95a033d15cc9bc74`.
On one 60–90 s tuning excerpt, fixed-sample context (`segment_size=1724`)
was retained. Matched-duration deltas (matched minus fixed; SDR dB / fullness /
bleedless) were `48 kHz: -0.023188 / +0.000760 / -0.000862` and
`96 kHz: -1.474285 / -0.068377 / -0.045346`; the 96 kHz matched-duration arm
(`segment_size=3752`) was rejected. The 48/96 kHz variants are upsampled from
44.1 kHz, so they are not upper-band evidence. No heldout inference ran.

The CPU smoke is at
`$UPMIXER_EVAL_ROOT/q20/backend-smoke-8683132/cpu`, with analysis SHA-256
`d402fd66f236a8993ef3559b4026e8b60fa1ccaf666e4809cc51f2f6edc8ac3a`.
It ran both 48 kHz production-tree arms on the AM Contra 60–72 s tuning
excerpt, forced `cpu`, retained all six terminal outputs, and had zero OOM
fallbacks. Native minus delivery deltas (SDR dB / fullness / bleedless) were:

| Stem | Delta |
| --- | --- |
| Bass | `+0.542762 / +0.000039 / +0.005859` |
| Drums | `+1.634830 / +0.010380 / +0.001137` |
| Other | `-0.000569 / -0.001783 / +0.003653` |
| Vocals | `+0.552195 / +0.001663 / -0.002379` |

| Arm | Wall / user / sys | Maximum RSS / peak footprint |
| --- | --- | --- |
| delivery | `30.62 / 148.56 / 3.00 s` | `4,942,839,808 / 4,668,657,984 B` |
| native | `30.91 / 150.20 / 2.97 s` | `4,940,709,888 / 4,832,153,920 B` |

CUDA is unavailable on this Apple host and was not simulated; no device
memory ceiling is frozen.

Three blinded packs now exist under
`$UPMIXER_EVAL_ROOT/q20/listening/`. All are tuning-only and unrated; their
`ratings.csv` files are blank, answer keys are mode `0600`, and this ledger
does not reveal A/B mappings.

| Pack | Cases | Manifest SHA-256 | `SHA256SUMS` SHA-256 |
| --- | ---: | --- | --- |
| `tuning-rate-flags-v1` | 14 | `39babea46f084c4323a2b6b81fc589b0fcde0e180f499aa0dc6d721e44f8b929` | `d2cbba378a5bf0645f5c7f3c9e8ab0355aacd9ba4d60bb7df534764a2c01f513` |
| `branch-smoke-flags-v1` | 1 proxy case | `49bb7ed95c42ba5154e8d764b5cc910ee39e74259507abee7b96219be2adc1b7` | `094bda9411236a06961d00cabcb15c693aacbae50918b8cceb6700bb945053c5` |
| `residual-policy-v1` | 4 | `d09441bac064a3dbec6c0b90c3df43f3154ff4fa05e59a9ba1689c605165c38d` | `463390f2455cb742e882cde7cd5b4a3ad66685c72643d0cc46c2073648489a3e` |

The local reviewer is implemented by commits `70be357`, `abf4a2c`, and the
post-review fix `7d716bf`. From the repository root, run one pack at a time on
its default port:

```bash
uv run python scripts/run_listening_review.py \
  "$UPMIXER_EVAL_ROOT/q20/listening/tuning-rate-flags-v1" --port 8765
uv run python scripts/run_listening_review.py \
  "$UPMIXER_EVAL_ROOT/q20/listening/branch-smoke-flags-v1" --port 8765
uv run python scripts/run_listening_review.py \
  "$UPMIXER_EVAL_ROOT/q20/listening/residual-policy-v1" --port 8765
```

The server binds to `127.0.0.1` only (localhost). It reads each pack's
`manifest.json` and `ratings.csv`, serves audio only through the manifest
allowlist with paths confined to the pack, never reads or serves the answer
key, and saves ratings atomically with a flushed and synced temporary CSV
replacement. Prior saved rows are not exposed by the page or API; only
current-browser saved rows are cached for navigation. Validation passed: 2
focused tests; Ruff check and format check; full suite
`1621 passed, 38 deselected, 24 warnings`.

The residual pack's independent recomputation passed all four cases:
separate reconstruction maximum absolute error
`1.862645149230957e-09`; assigned-policy reconstruction maximum absolute error
`5.960464477539063e-08`. The separate-Unassigned residual contract remains
provisional: policy 1 leaves `Other` raw, while policy 2 adds the
delivery-domain residual. The Q21 native-rate candidate is now implemented,
its automatic parity gate passes, and its quick owner listening verdict was
`OK` on 2026-09-06. Any default change remains blocked pending policy freeze
and one single heldout run. Heldout inference has not run.

The corrected one-item smoke artifacts are:

- `$UPMIXER_EVAL_ROOT/q20/short-12s/comparison.json`
  (`9727e78fd50881591a01de8b8081220e32eedda2bd7944c152e540a1780d54b7`).
  It is an upsampled 44.1 kHz MUSDB common-band control, not upper-band
  evidence. Assigning the delivery residual to `Other` reduces native
  `Other` fullness by about `0.0205`, beyond the frozen `0.005` co-metric
  allowance, so that policy is rejected provisionally.
- `$UPMIXER_EVAL_ROOT/q20/highrate-smoke/analysis.json`
  (`2994714a2baa60a43157ae206d6703b89bf3cd57141efe2bf2b764b549423b62`).
  Freischütz conservation is about 30.10 dB SDR in both arms; Celtic is
  43.68 dB delivery versus 39.84 dB native. Separate unassigned remainder
  reconstruction is within `9.4e-10` maximum absolute error in these smokes.
  These are conservation/bandwidth checks, not instrument-quality evidence;
  the Celtic source has measurable content above 22.05 kHz, while MUSDB
  upsampling cannot create that evidence.

All earlier Q20 outputs that used the default
`scipy.signal.resample_poly` window are archived under
`$UPMIXER_EVAL_ROOT/q20/**/***-scipy-default-invalid`, with `INVALID.json`
markers. Their comparison files (`short-12s/comparison-scipy-default-invalid.json`
and `highrate-smoke/analysis-scipy-default-invalid.json`) are audit-only and
must not be used for conclusions. The corrected artifacts use the selected
120 dB FIR above.

### Q21 — native-rate production candidate: accepted for opt-in use

The opt-in production path is implemented by commits `6540f9a`, `9dafcf3`,
`c8c7855`, and `81d4d79`. The final fix pre-resamples each non-native source
zone with the selected 120 dB FIR before native inference, so the production
path and Q20's native evaluator arm receive identical samples. Cache identity,
resume identity, delivery conversion, exact rounded frame counts, supplied
stems, and silence-skip paths are covered by the owning tests.
The current pre-resample uses the zone array already loaded by the pipeline;
bounded streaming conversion remains a follow-up before any broad default
promotion.

The direct production smoke used the genuine 48 kHz tuning input
`freidi-48k/smoke/q20-conservation-12s`, not heldout material. It ran the
Deux → BS-Roformer-SW tree on MPS with native inference at 44.1 kHz, delivery
at 48 kHz, batch size 1, overlap 2, and the default silence-skip policy. The
six public terminal WAVs were stereo, finite, 576,000 frames, and bit-identical
to the retained Q20 native arm: maximum absolute and RMS differences were both
`0.0`. Runtime was `27.46 s` wall time with maximum RSS
`1,869,807,616` bytes; no OOM fallback occurred. The focused native-rate and
rate-experiment suite passed `24` tests.

Retained evidence is at
`$UPMIXER_EVAL_ROOT/q21/production-native-smoke-freidi-48k-81d4d79`:
`comparison.json` SHA-256
`6dfd5863b6aac908f28df9325cc75d51de2d530891fdd6983406b42097597648` and
`SHA256SUMS` SHA-256
`d967f72d18853400bc2c7e27d25c23f5edd4f0ff03493af4eb2d591caafd6652`.
The code revision is
`81d4d79f7ff8df444734d39c47d8aaa52e02fa2a`.

The automatic gate passes. The owner's 2026-09-06 quick listening verdict was
`OK`. Under the sequential candidate workflow, Q21 is accepted for opt-in use
and Q50 was evaluated next and rejected. The path remains opt-in/default-off; no
panel, ratings, defect ledger, or heldout inference is claimed.

### Q30 — origin-view experiment complete; candidate rejected

Q30 code is in commits `2007ca6` and `f500a8a`; the evaluated code revision is
`f500a8abdd2b046275a07c41b0b97699b9c02e69`. The opt-in
`--extra-origin-samples 11138` path is evaluation-only direct-model plumbing:
it runs linked-stereo origin 0 and a second view with 11,138 zero samples on
both sides, crops the second output back to the exact source span, and fuses
the two float32 waveforms at 50/50. It retains per-view outputs, timing, and
provenance, and enforces matching settings/rate plus stereo, exact-shape,
finite-output, and float32 checks. The flag is restricted to direct models and
cannot be combined with production-tree/ensemble, rate-arm, TTA, or pitch
shift. Ordinary report JSON is unchanged; origin fields are opt-in.

Validation at this revision was `1594 passed, 38 deselected, 24 warnings` for
the full suite, `1317 passed, 18 skipped, 35 deselected` for the core suite,
and `26 passed` for the focused Q30 tests.

The base protocol root is
`/Volumes/External SSD/upmixer-eval/separation-quality/q30/direct-origin-11138-v1`
(`q30-direct-origin-11138-v1`, protocol SHA-256
`eb9f33ceb510c020912d33257480507c3ebe2b1da17936d1d86509df240485dd`). The
stage-input protocol is nested at
`/Volumes/External SSD/upmixer-eval/separation-quality/q30/direct-origin-11138-v1/stage-sw-after-deux-v1`
(`q30-stage-sw-after-deux-v1`, protocol SHA-256
`1ea35bd87039ce1fedaef1ff1124074e1c4a740f1a6feff284749ff58a4e8952`), with
corpus SHA-256
`9d26d6cb242e1032423d52c740b2f376d7cf99fac8020584125da482a01bc39f` and
fixed Deux input SHA-256
`c1f5f4e7748b3b30e81fb016772f94f9efe77e8fd8cafead27587275163031d5`.

| Screen | Candidate minus baseline (SDR / fullness / bleedless) | Runtime | Analysis SHA-256 | `SHA256SUMS` SHA-256 |
| --- | --- | ---: | --- | --- |
| Deux, one tuning item, Vocals | `+0.0206775665 / +0.0002805484 / +0.0012732680` | `1.465x` | `8b04f0b52cd778d182da2b4d805e442a259dc76a5843076e8f69a7bbd50c9461` | `665c6565b64625f90e10e96294a5aa07200ae2357e5f8b34dba9c7211bdc0131` |
| SW on the raw mixture, Drums | `+0.100725174 / +0.000468951 / +0.000852201` | `1.490x` | `777586335e75a7fc3b1b9bc69949e6688ce70bcb479294af8adf680800c1af0a` | `877fc44b5acc9b008077a3ef22523f001418b6d3a6848b1aa025873529482fb3` |
| SW after Deux stage input, Drums | `+0.103761673 / +0.001047498 / +0.001054209` | `12.18 s → 19.35 s (1.589x)` | `b28dfe18ae9da54774fd1152e5dfee32482960a87a8559f8f134ba30254ca14a` | `33b4b5ea357e52cf0b8c29372f2eb14fe89a19eb28692b0675714790f2b40003` |

Raw-mixture SW Bass and Other scores were near zero and are diagnostic,
non-production-equivalent evidence. Stage-input Bass and Other deltas were
negligible. The controlled stage-input repeat measured RSS of
`2153627648` bytes baseline and `2319253504` bytes candidate (`1.077x`), with
zero swaps.

All Q30 artifacts passed exact origin-0/baseline array checks, exact float32
fusion checks, shape/rate/finite checks, provenance totals
`scheduled_windows/evaluated_unique_windows/tail_replay_contributions/model_forward_calls`
of `4/2/2/2`, and the individual regression screen had no flags. The
experiment/eval plumbing is complete, but every one-item gain is below the
frozen material gate. The candidate is rejected for production and Q31 is not
implemented. No full tuning or heldout run was made; bootstrap, listening, and
numeric memory-ceiling gates remain open. No default or web product behavior
changed, and Q30 was not promoted.

### Q40 — Deux cascade pilot complete; candidate rejected

Task Q40 status: `rejected` (completed experiment, promotion prohibited). The
three reviewable implementation slices were `c104a36`, `24a32dc`, and
`5ff2524`, evaluated at code revision
`5ff2524ff2df6ccf54c4c06a95db2e5f0b57ad3f`. Their exact changed files were:

| Commit | Changed files |
| --- | --- |
| `c104a36` | `packages/core/src/eval/__init__.py`, `packages/core/src/eval/cascade.py`, `packages/core/src/eval/harness.py`, `packages/core/src/eval/report.py`, `packages/core/tests/test_eval_cascade.py`, `scripts/run_eval.py` |
| `24a32dc` | `packages/core/src/eval/__init__.py`, `packages/core/src/eval/cascade.py`, `packages/core/src/eval/harness.py`, `packages/core/src/eval/report.py`, `packages/core/src/eval/report_format.py`, `packages/core/src/eval/retention.py`, `packages/core/tests/test_eval_cascade.py`, `scripts/run_eval.py` |
| `5ff2524` | `packages/core/src/eval/cascade.py`, `packages/core/src/eval/harness.py`, `packages/core/src/eval/report.py`, `packages/core/src/eval/report_format.py`, `packages/core/src/eval/scoring.py`, `packages/core/tests/test_eval_cascade.py`, `scripts/run_eval.py` |

The frozen item was one 12-second tuning excerpt, `Hollow Ground - Ill Fate`,
at 44.1 kHz. No heldout inference, listening review, deferred arm, or
expansion run was made. The protocol, corpus, and provenance identities are:

| Artifact | Identity |
| --- | --- |
| Protocol `q40-deux-counterfactual-half-v1` | SHA-256 `264429e1133bc004b61104f036f5374f51b80705cc5ea1548a8e06abe44ab7f2` |
| Q40 corpus | SHA-256 `b835022cb61d8b61197f4d0521686e3f42628c864a36138f46921c0cef0c694e` |
| Provenance | SHA-256 `163b3c1d80c5630ecde8e6e021ec9b4c23717730a06c6302e8d4663acb2151da` |
| Final result root | `/Volumes/External SSD/upmixer-eval/separation-quality/q40/deux-counterfactual-half-v1/results` |
| Analysis | SHA-256 `9c23fd1f78ae04aacd124b872761a909909070cc184d172a653545f0692c8f9e` |
| Results manifest (`SHA256SUMS`) | SHA-256 `1ad6a51a128fdf7c582b81ed9ad137d4a837abf8cfb702545e7f4df1c0f56faa` |

The scored candidate was the fixed half recipe against `complementary-v0`:

| Arm | Stem | SDR (dB) | Fullness | Bleedless |
| --- | --- | ---: | ---: | ---: |
| `complementary-v0` | Vocals | 8.730073 | 0.754295 | 0.887570 |
| `complementary-v0` | Instrumental | 20.810938 | 0.925277 | 0.984917 |
| `fixed-half-recipe` | Vocals | 8.773351 | 0.736398 | 0.899439 |
| `fixed-half-recipe` | Instrumental | 20.854319 | 0.930158 | 0.983209 |

Candidate minus control median deltas were SDR `+0.043329 dB`, fullness
`-0.006508`, and bleedless `+0.005081`, against material thresholds of
`0.2 dB`, `0.01`, and `0.01`. None passed; there were no individual regression
flags. The fixed recipe therefore missed the material screen. The
counterfactual and refined-complement arms remain diagnostic evidence only.

The candidate/control wall times were `17.88 s / 12.15 s = 1.4716049x`; the
internal accounting ratio was `1.6832405x`. Maximum RSS was
`1,989,836,800` bytes for the candidate and `1,649,934,336` bytes for the
control. There were zero swaps. Aggregate accounting was
`scheduled_windows/evaluated_unique_windows/tail_replay_contributions/model_forward_calls`
`4/2/2/2`. Fourteen retained WAVs passed validation; deterministic identity
checks passed; candidate conservation passed with maximum absolute error
`2.9802322387695312e-08`.

The focused Q40 root suite passed `72`; the affected agent suite passed `83`
with `1 deselected`; the final full suite passed `1619` with `38 deselected`
and `24 warnings`. Ruff and markdown whitespace/diff checks passed. The
pre-hardening archive remains at
`/Volumes/External SSD/upmixer-eval/separation-quality/q40/deux-counterfactual-half-v1/results-pre-hardening-24a32dc`
for audit only.

The stop rule applies: no production default change was made and the evaluation
harness is retained. The combined Q20/Q21 stage remains partial: Q20 is
incomplete; Q21 was accepted for opt-in use after the owner's 2026-09-06 quick
listening `OK` and remains default-off. This Q40 result is diagnostic and
promotion-prohibited; those entry prerequisites independently continue to block
production. Under the sequential candidate workflow, Q50 was evaluated next and
rejected below. Q60 remains conditional on a measured spatial/residual defect,
and Q80 on a Q03 quiet-input failure. Q70 remains gated on an accepted
magnitude improvement and Q90 on a named Q40/Q50 ambiguity plus failed
classical cues.
There is no Q41.

Exact rerun commands (use fresh output directories):

```bash
cd /Users/coderynx/Projects/upmixer

/usr/bin/time -lp uv run python scripts/run_eval.py \
  --corpus "/Volumes/External SSD/upmixer-eval/separation-quality/q40/deux-counterfactual-half-v1" \
  --variant real-model \
  --output-dir "/Volumes/External SSD/upmixer-eval/separation-quality/q40/deux-counterfactual-half-v1/results-rerun/independent-deux" \
  --model becruily_deux.ckpt \
  --sample-rate 44100 \
  --batch-size 1 \
  --overlap 2 \
  --retain-stems

/usr/bin/time -lp uv run python scripts/run_eval.py \
  --corpus "/Volumes/External SSD/upmixer-eval/separation-quality/q40/deux-counterfactual-half-v1" \
  --variant real-model \
  --output-dir "/Volumes/External SSD/upmixer-eval/separation-quality/q40/deux-counterfactual-half-v1/results-rerun/fixed-half-recipe" \
  --model becruily_deux.ckpt \
  --sample-rate 44100 \
  --batch-size 1 \
  --overlap 2 \
  --retain-stems \
  --cascade-vocal-repair
```

### Q50 — complete/rejected after both automatic claims

Both Q50 claims failed their automatic gates and are closed: aggregate-Drums
event/repetition repair v1 and kit-sibling transfer v2. Neither changed
production behavior.

#### Aggregate-Drums event/repetition repair v1

The tuning-only candidate was implemented in commits `94bb197` and `b10eff2`.
The prescreen artifact is
`/Volumes/External SSD/upmixer-eval/separation-quality/q50/event-repair-v1/prescreen-12s`;
`report.json` SHA-256 is
`5dc4bf3137c0b8c0efcc61c92657b02afca9527e59b03cb675afd61f9450b099`, and
`SHA256SUMS` SHA-256 is
`bdda9dfed8da1f9f7562a540ea87c49e65450dcee3d98fec9e0e20bd893c2332`.

The automatic 12-item tuning prescreen recorded `0` transfers, `12` no-op
items, and median Drums deltas of `0.0 dB / 0.0 / 0.0` for SDR / fullness /
bleedless. Conservation was exact (`0.0` maximum absolute error), but the
material gate failed. Review also found unsafe false-transfer cases and
unbounded full-track STFT allocation. The candidate is rejected: no heldout
inference, listening gate, or production integration was run. The experimental
helper, exports, tests, and runner were removed; the external artifact remains
available and the two implementation commits preserve reproduction history.

#### Kit-sibling transfer v2

The second claim was implemented by `04abebb` (`feat(eval): add Q50 kit repair
arms`). That evaluation-only helper and its focused tests are removed from the
current tree; commit `04abebb` preserves the implementation for audit. The
external corpus is the 60-item, 44.1 kHz mono IDMT-SMT-Drums V2 selection at
`/Volumes/External SSD/upmixer-datasets/idmt-smt-drums-q50`.

| Metadata | Path | SHA-256 |
| --- | --- | --- |
| Corpus manifest | `/Volumes/External SSD/upmixer-datasets/idmt-smt-drums-q50/corpus.json` | `7dc04af796dc3a580bcf5dce9cb4effef7c7e3812729f9cfc3e5900c33dafe9c` |
| Provenance | `/Volumes/External SSD/upmixer-datasets/idmt-smt-drums-q50/provenance.json` | `7282b884c42a7f0b1d270d27fe82975a1c998411f69dd5b53ea06464b2eb2fdd` |
| Metadata checksums | `/Volumes/External SSD/upmixer-datasets/idmt-smt-drums-q50/SHA256SUMS` | `1bc1c99363cf6c17863170a027bff2fdefb9044b00d7ba2554d9080bfa0ba8a7` |
| Source archive | `/Volumes/External SSD/upmixer-datasets/downloads/IDMT-SMT-DRUMS-V2.zip` | `8ee68f0cab1d66c800fa19f8be72a0b400f8e96919ac67182053f752e38697bb` |

All 60 items are tuning-only: items 01–48 are the internal
`tune-development` group and 49–60 are `tune-validation`; neither is formal
heldout. The selection contains 240 reference WAVs for Kick, Snare, Hi-Hat,
and mixture; Toms, Ride, and Crash are unavailable. The archive and exact
reference paths remain external to the repository.

An abandoned 60-item DrumSep inference exceeded six minutes on its first item,
completed `0/60` items, and reached about `1.3 GB` RSS. Its retained log is
`/Volumes/External SSD/upmixer-eval/separation-quality/q50/kit-v2/drumsep-raw.run.log`,
SHA-256 `c3a929186d7ee09f0e6605483a63548736043cb7f9b04d8b08b8aeefd07db7a8`.

The retained real `WaveDrum02_01` 12-second pre-screen produced `0` transfers
and `0.0 dB / 0.0 / 0.0` SDR / fullness / bleedless deltas for both default
arms (`cue-only` and `cue-plus-repetition`) in about `0.985 s` combined. A
small threshold diagnostic was the best cue-only result: at gain `0.05` it
made one `Kick → Snare` transfer, but worsened Kick SDR by `-0.099961 dB` and
fullness by `-0.002374`, and Snare SDR by `-0.007407 dB` and bleedless by
`-0.000360`; repetition made `0` transfers and RTF was `0.0513`. The automatic
gate therefore failed.

Reviewer safety findings were also promotion blockers: heuristic exemplar
similarity did not prove ownership of a missing event at the target time, and
the ownership check ignored unavailable kit siblings (Toms, Ride, and Crash),
so false transfers remained possible. No heldout inference, listening gate, or
production integration was run for v2.

Q50 is complete/rejected only after these aggregate v1 and kit v2 results.
Q60 and Q80 lack their measured prerequisites; Q70 remains blocked. The Q40/Q50
result supplies the named ownership ambiguity and failed classical-cue evidence
needed to enter Q90, so its adviser preflight was run below.

### Q90 — adviser preflight rejected

The Q40/Q50 results identify a concrete overlapping-kit ownership ambiguity and
show that the existing onset, spectral, decay, and repetition cues did not
repair it. The pitch-adviser candidates nevertheless failed the required
modality, compatibility, and artifact-provenance checks:

| Candidate | Preflight findings | Disposition |
| --- | --- | --- |
| Basic Pitch `0.4.0` | Polyphonic note/onset metadata only; internal mono 22.05 kHz; its Python `<=3.11`/TensorFlow path is incompatible with the host's Python 3.13 on macOS; no per-artifact upstream SHA was available. See the [official repository](https://github.com/spotify/basic-pitch), [bundled weights](https://github.com/spotify/basic-pitch/tree/main/basic_pitch/saved_models/icassp_2022), and [license](https://github.com/spotify/basic-pitch/blob/main/LICENSE). | Reject |
| torchcrepe `0.0.24` | Monophonic F0 metadata only; mono 16 kHz; current environment lacks `torchaudio`, `resampy`, and `tqdm`; it supplies no kit identity or ownership masks; no per-weight upstream SHA was available. See the [official repository](https://github.com/maxrmorrison/torchcrepe), [weight assets](https://github.com/maxrmorrison/torchcrepe/tree/master/torchcrepe/assets), and [original CREPE license](https://github.com/marl/crepe/blob/master/LICENSE). | Reject |

Neither adviser resolves the observed broadband Snare/Hi-Hat loss or
overlapping kit ownership. No package or checkpoint was downloaded, no heldout
inference was run, and no repository code changed. Q90 is therefore rejected
for this ambiguity; retain the incumbent classical/no-adviser behavior. Q100
for accepted Q21 is next.

## Gates still open

- [ ] Run no heldout inference until the rate, residual, and listening policy is frozen; the generated heldout excerpts remain untouched.
- [x] Prepare blinded randomized listening assets and answer keys; three tuning-only packs are recorded above.
- [ ] Obtain the quick owner listening OK for each automatically passing candidate; no listener panel, ratings form, or manual human report is required.
- [ ] Define and persist the separate unassigned source-zone remainder, including cache/store, preview/export, subset/solo/mute, alignment, rate, and level semantics. A source anchor is not residual preservation.
- [x] Run the fixed-sample-context versus matched-duration diagnostic; retain fixed-sample context and reject the 96 kHz matched-duration arm.
- [ ] Expand and confirm eligible complete-tree real evidence where references permit. The corrected one-item branch smokes exercised the SCNet ensemble, Karaoke lead/backing path, and private `_deux_inst` on tuning/proxy material; broader coverage remains open, and the branch proxies plus supplementary IDMT, MedleyDB, CrowdioSet, Freischütz, and Celtic sets do not substitute for aligned heldout music.
- [ ] Measure CUDA when available and freeze a numerical device memory ceiling; the CPU smoke is recorded above.
- [ ] Verify the normal source-anchor product path separately with anchoring off for the experiment.
- [x] Q21 native-rate production integration, delivery conversion, cache/resume identity, and direct production parity are implemented and pass the automatic gate; bounded streaming conversion remains a follow-up.
- [x] Q21 quick owner listening OK recorded on 2026-09-06; Q21 accepted for opt-in use and remains default-off; Q50 was evaluated next and rejected.
- [x] Q90 adviser preflight completed for the Q40/Q50 ambiguity and rejected on modality, compatibility, and provenance; no downloads, heldout inference, or code changes.

## Restart commands

Keep audio, reports, and generated indexes on the external SSD. These variables
avoid placing copyrighted audio paths in the repository:

```bash
export UPMIXER_REPO_ROOT=/Users/coderynx/Projects/upmixer
export UPMIXER_SSD_ROOT="/Volumes/External SSD"
export UPMIXER_DATASET_ROOT="$UPMIXER_SSD_ROOT/upmixer-datasets"
export UPMIXER_MUSDB_ROOT="$UPMIXER_DATASET_ROOT/musdb18-hq"
export UPMIXER_EVAL_ROOT="$UPMIXER_SSD_ROOT/upmixer-eval/separation-quality"
export UPMIXER_Q20_ROOT="$UPMIXER_EVAL_ROOT/q20"

cd "$UPMIXER_REPO_ROOT"
git switch feature/improve-stem-separation
git log --oneline --decorate -8
git status --short

# Validate the frozen 24-recording excerpt corpus without changing audio.
uv run python "$UPMIXER_MUSDB_ROOT/make_q03_q20_excerpt_corpus.py" \
  --dataset-root "$UPMIXER_MUSDB_ROOT" \
  --repo-root "$UPMIXER_REPO_ROOT" --validate-only

# Re-run Q03 on the tuning excerpts into a fresh directory.
uv run python scripts/run_eval.py \
  --corpus "$UPMIXER_MUSDB_ROOT/excerpts-60s-72s/44.1k/tuning" \
  --variant production-tree --sample-rate 44100 \
  --stems vocals,bass,drums,guitar,piano,other --retain-stems \
  --output-dir "$UPMIXER_EVAL_ROOT/q03/excerpts-60s-72s/tuning-44k-rerun"

# Q20 corrected matrix; each output directory must be fresh.
uv run python scripts/run_eval.py \
  --corpus "$UPMIXER_MUSDB_ROOT/excerpts-60s-72s/48k/tuning" \
  --variant production-tree --sample-rate 48000 --rate-arm delivery \
  --stems vocals,bass,drums,guitar,piano,other --retain-stems \
  --output-dir "$UPMIXER_Q20_ROOT/tuning-12x12s/48k-delivery-rerun"
uv run python scripts/run_eval.py \
  --corpus "$UPMIXER_MUSDB_ROOT/excerpts-60s-72s/48k/tuning" \
  --variant production-tree --sample-rate 48000 --rate-arm native \
  --stems vocals,bass,drums,guitar,piano,other --retain-stems \
  --output-dir "$UPMIXER_Q20_ROOT/tuning-12x12s/48k-native-rerun"
uv run python scripts/run_eval.py \
  --corpus "$UPMIXER_MUSDB_ROOT/excerpts-60s-72s/96k/tuning" \
  --variant production-tree --sample-rate 96000 --rate-arm delivery \
  --stems vocals,bass,drums,guitar,piano,other --retain-stems \
  --output-dir "$UPMIXER_Q20_ROOT/tuning-12x12s/96k-delivery-rerun"
uv run python scripts/run_eval.py \
  --corpus "$UPMIXER_MUSDB_ROOT/excerpts-60s-72s/96k/tuning" \
  --variant production-tree --sample-rate 96000 --rate-arm native \
  --stems vocals,bass,drums,guitar,piano,other --retain-stems \
  --output-dir "$UPMIXER_Q20_ROOT/tuning-12x12s/96k-native-rerun"

# Compare the recorded matrix, then verify the correction and full suite.
uv run python "$UPMIXER_Q20_ROOT/tuning-12x12s/compare_tuning_reports.py"
uv run python "$UPMIXER_Q20_ROOT/analyze_smokes.py"
uv run pytest packages/core/tests apps/api/tests apps/cli/tests -q
```

Do not run the heldout arm until the open gates above are frozen and recorded.

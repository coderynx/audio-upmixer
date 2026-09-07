# Binaural renderer diagnosis — 2026-09-07

The shared direct-filter banks carried direction-dependent excess phase from
the measured HRIRs. Correlated feeds in adjacent virtual speakers consequently
cancelled mid/high frequencies while adding coherently in the bass. This
reproduced a bass-heavy result at 48 kHz with no project mastering, LFE, room,
or listening EQ. The correction is in filter generation and reaches both
preview and export through the same regenerated assets.

## Reproduction and cause

Drive FL, C and FR with identical impulses through the real flat 7.1.4
renderer. Mean combined ear power over 80–300 Hz was 6.87 dB above 1–4 kHz.
The same check failed on studio (6.17 dB) and listening (5.89 dB).

An offline Apple reference used `AVAudioEnvironmentNode` with mono players at
the same directions, `HRTFHQ`, headphone output, no reverb and no distance
attenuation. This is Apple's AVFAudio headphone renderer, **not a captured
PHASE or Apple Music render**, and is not a claim of equivalence to them.
Its front-bed bass/presence difference was −0.33 dB. Isolated-source magnitude
differences were much smaller than the differences after coherent summation.

Before correction, FL/FR renders in all seven layouts matched the independent
SADIE fixture within `1.969e-8` absolute sample error. The engine was faithfully
reproducing the filters; the problematic behavior was their phase interaction,
not a broken HOA convolution or an unintended post-EQ stage. Merely copying
measured HRIR waveforms exactly was an inadequate quality criterion for this
virtual-speaker renderer.

## Phase correction

`scripts/build_binaural_filters.py::condition_direct_hrirs` converts each
measured ear response to minimum phase using an 8192-point real cepstrum. It
then restores the measured excess interaural delay, fit over 200–1500 Hz after
subtracting the minimum-phase filters' own interaural phase. A shared
16-sample guard keeps fractional-delay pre-ringing in the causal bank; the
later ear receives the remaining delay. Output remains 256 taps.

This retains the directional magnitude cues and low-frequency interaural delay
without the direction-dependent excess phase and common measurement latency.
It changes phase relationships between virtual speakers intentionally. It does
not preserve every high-frequency interaural phase detail. Room tails remain
the same samples and gains, and no EQ or crossfeed settings were changed.

The layout encoder's left inverse reconstructs the **conditioned** direct
filters. All 24 profile/layout-or-legacy banks were regenerated from the same
SADIE SOFA bytes and copied identically to core and web. The transaural inverse
still models the original physical plant; its desired headphone ear signals
now come from the conditioned flat bank.

| Front-bed measurement | Before | After |
| --- | ---: | ---: |
| Flat: bass above presence | 6.87 dB | 0.57 dB |
| Studio: bass above presence | 6.17 dB | 0.44 dB |
| Listening: bass above presence | 5.89 dB | −0.10 dB |

Across all nominal measured directions, the worst layout's 95th-percentile
individual-ear magnitude error over 80 Hz–16 kHz was 0.020 dB. The maximum
single-bin error was 0.584 dB near a notch. Maximum fitted low-frequency ITD
change was 0.296 microseconds. These checks bound the effects of the finite
256-tap realization; they do not assert sample-for-sample HRIR identity.

The flat coherent-front presence band recovers approximately 6.3 dB while
its bass level stays essentially constant. Uncorrelated feeds retain their
original magnitude calibration. This addresses the measured coloration
mechanism; subjective externalization and exact Apple PHASE similarity still
require listening validation.

## Separate sample-rate correction

`binaural/decoder.py` resampled FIR taps as audio. `resample_poly` preserves
sample amplitude; convolution gain depends on the tap sum. Consequently,
changing from 48 to 96 kHz added 6.021 dB to the directional decode, without
adding that gain to LFE. At 44.1 kHz the directional decode lost approximately
0.74 dB, increasing LFE's relative weight. Post-render loudness correction
cannot restore the relative balance.

Multiplying the resampled taps by `file_sr / sample_rate` fixes that error.
`test_binaural_calibration.py` exercises the actual renderer, measured banks,
voicing and LFE bypass at 44.1, 88.2 and 96 kHz for all three profiles. All
nine cases failed before the fix; all pass afterward with speaker/LFE transfer
ratios within 0.1 dB of 48 kHz at 80, 250, 1000 and 8000 Hz.

The browser and native previews run at 48 kHz, so that separate fix does not
explain the original preview complaint. The separate XTC
loader also resamples FIRs without transfer-gain correction; that remains a
transaural follow-up outside this headphone change.

## Validation

Run `uv run pytest packages/core/tests/test_binaural_calibration.py -q` for
28 calibration/phase/asset checks. All three coherent-front cases failed before
the asset regeneration and pass afterward. The tests also check source
magnitudes, ITD, common-delay invariance and preview/export asset identity.

An actual WASM preview render with shipped banks, 128-frame blocks and all
three profiles agreed with Python's binaural render of the corresponding
speaker bed within `7e-9` maximum absolute error. No runtime DSP or filter
length changed. The desktop app was rebuilt with the new assets; browser
sessions need reloading and installed desktop apps need the rebuilt bundle.
Long-running Python processes must reload cached filter sets.

Web tests: 434 passed. Web production build and macOS Tauri app build passed.
The Python full-suite run exposed only the intentionally changed binaural
delivery snapshot; it was regenerated and the focused golden/calibration
rerun passed (31 tests). The final core/API/CLI suite passed: 1,672 tests,
38 deselected. The desktop bundle's 96 HRIR parts were verified byte-identical
to core; every room-tail sample after tap 256 is unchanged from the old bank.

The synthetic engine benchmark passed its binaural case (median mean
0.742 ms, p99 2.519 ms), but failed the unrelated native-all-ambient and
native-silence-tail p99 budgets. That benchmark generates its own taps and
does not load the changed assets; it is not a performance regression
measurement of this phase correction.

For listening, a synthesized centered voice was rendered through the old and
new flat banks with coherent FL/C/FR feeds and separately normalized to
−23 LKFS, without peak limiting. Impulses, spectra and output levels were
verified; no subjective listening verdict is claimed.

## References

- [SADIE II paper](https://www.mdpi.com/2076-3417/8/11/2029): source measurements
  and diffuse-field compensation.
- [Minimum-phase magnitude plus ITD localization experiment](https://ntrs.nasa.gov/citations/20020041204):
  separates measured magnitude responses from the interaural delay cue.
- [HRTF time-of-arrival modeling](https://pmc.ncbi.nlm.nih.gov/articles/PMC4582460/):
  distinguishes minimum phase, time of arrival and excess phase, and explains
  why inter-source timing matters when rendering multiple sources.
- [Apple headphone rendering algorithms](https://developer.apple.com/documentation/avfaudio/avaudio3dmixingrenderingalgorithm):
  identifies the AVFAudio reference used here, distinct from the app's PHASE path.

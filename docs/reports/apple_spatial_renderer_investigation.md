# Apple Spatial monitoring investigation — 2026-09-07

Status: AVFoundation media transport implemented and loopback verified. The
reported head-turn image and exact Logic/Apple Music parity remain unverified.
The findings below describe the earlier PHASE implementation and investigation;
the implementation and validation section records the replacement.

Acceptance requirement clarified by the user: use the actual Apple rendering
pipeline used for Dolby Atmos monitoring, including its head-tracked scene
behavior. A perceptually improved PHASE approximation does not satisfy this
requirement. Head tracking disabled cannot reproduce or validate the reported
motion-dependent symptom and is not a prerequisite for further investigation.
Public media API support alone is not proof of exact Logic/Apple Music parity;
that equivalence remains to be established.

## Findings

The current `apps/web/src-tauri/native/audio_bridge.m` sends the mastered
speaker bed to a PHASE ambient mixer with an identity orientation. Automatic
listener orientation is the only head-pose transform in this bridge; there
is no application-side yaw multiplier or speaker-distance parameter to tune.
The 7.1.4 adapter swaps the side/rear channel pairs into Apple's layout order.
It also folds LFE into the fronts because PHASE ignores that channel.

Apple documents the ambient mixer as rendering channels from their layout's
speaker directions. It does not document this configuration as equivalent to
Logic's Apple Renderer or Apple Music. Changing a speaker radius cannot tune
this mixer: its interface specifies orientation and layout, not distance.
[PHASE ambient mixer](https://developer.apple.com/documentation/phase/phaseambientmixerdefinition)

`Entitlements.plist` requests `com.apple.developer.coremotion.head-pose` but
omits `com.apple.developer.spatial-audio.profile-access`. Apple specifies the
latter for personalized rendering through PHASE, AVAudioEngine and
AUSpatialMixer. This is a configuration gap, not proof of the reported cause;
its perceptual relevance depends on the user's profile and actual signed app.
[Personalized spatial audio](https://developer.apple.com/documentation/phase/personalizing-spatial-audio-in-your-app)

The existing local release bundle is linker/ad-hoc signed, has no team ID,
and `codesign -d --entitlements :-` emits no entitlement payload. This bundle
therefore cannot be used to verify that the configured entitlement survives
packaging. It is not evidence about another installed build the user may use.

Apple explicitly supports media Spatial Audio on macOS through AVPlayer and
AVSampleBufferAudioRenderer. Logic documents its Apple Renderer as the
headphone virtualization used by Apple Music. These establish the appropriate
reference and public media playback route; they do not establish that arbitrary
12-channel PCM and a decoded Atmos programme render identically.
[Mac API support](https://developer.apple.com/videos/play/wwdc2023/10233/),
[Logic monitoring formats](https://support.apple.com/en-mide/guide/logicpro/lgcp179f27c1/mac)

## Earlier AVFoundation implementation

Recovered from commit `6398431`, immediately preceding the PHASE replacement:

- Interleaved float PCM at 48 kHz, with channel layout and frame-based timestamps.
- A synchronizer with automatic startup buffering disabled.
- Playback started after the first 512-frame buffer (10.67 ms).
- Renderer capacity polled with 1 ms sleeps and a two-second timeout.
- No renderer-driven feeding callback or automatic-flush recovery.

These are observations, not a reproduced explanation for the historical
stuttering or silence. Apple demonstrates serial-queue feeding through
`requestMediaDataWhenReadyOnQueue:usingBlock:`. mpv also implements that route,
using larger batches and handling output changes/automatic flushes. Its use
of immediate startup demonstrates that disabling delayed startup alone is
not sufficient evidence of a bug.
[Apple custom player sample](https://developer.apple.com/documentation/avfaudio/playing-custom-audio-with-your-own-player),
[mpv AVFoundation output](https://github.com/mpv-player/mpv/blob/master/audio/out/ao_avfoundation.m)

The current Rust producer reports `underruns: 0` literally. That readout cannot
verify uninterrupted playback. It also pauses immediately when rendering
reaches EOF rather than waiting for queued presentation to finish; a media
backend with a larger queue needs an explicit drain before pausing.

## Recommended implementation and acceptance gate

First establish a controlled comparison using the same exported 7.1.4 PCM
in Logic's Spatial Audio Monitoring and Upmixer, with matched monitor level,
the same AirPods, personalization and fixed/head-tracked settings. Include
isolated C, FL/FR, surrounds and heights, followed by the same music excerpt.
Compare centered listening, slow turns to either side and recentering.

For a media-renderer replacement, retain the existing DSP and repair the
native transport around AVSampleBufferAudioRenderer: a serial feeder, bounded
buffering with measured startup headroom, correct seek/flush lifecycle,
presentation-clock meters and EOF drain. Measure callback gaps and queue
lead instead of trusting the hardcoded underrun count. Test all supported
layouts, short clips, repeated seek/pause/resume, device reconnect and control
latency before making it the default. Buffer duration must be measured on
AirPods; the PHASE queue's 42.7 ms is not an established media-renderer budget.

Do not carry PHASE's LFE workaround or its -3 dB monitor calibration into the
media backend without measuring the media backend's channel gains. Verify
the final signed app's head-pose/profile access for any PHASE comparison.
Do not add reverb or change head rotation merely to imitate perceived width.

Head tracking needs platform-specific treatment: the installed macOS SDK
marks the synchronizer's `intendedSpatialAudioExperience` property as
visionOS-only and explicitly unavailable on macOS. It cannot implement the
existing macOS submenu. Validate the media path's system Control Center
settings and communicate their ownership in the UI if replacing PHASE.

## Initial verification and listening requirement

Read the current bridge and Rust lifecycle, recovered the prior implementation,
checked Apple documentation and installed SDK declarations, and inspected the
local bundle signature. `system_profiler SPAudioDataType` showed built-in,
USB, display and virtual devices, with no AirPods. No listening comparison,
post-render capture, realtime stability test or perceptual regression verdict
was available during that initial investigation. The implementation checks
below supersede its transport-validation status; listening remains outstanding.

The remaining listening requirement is access to the affected AirPods playback
setup and a comparison against Logic's Apple Renderer with head tracking on
in both paths. The physical head-turn symptom cannot be reproduced using the
currently connected outputs. API and transport investigation can proceed
independently of that listening comparison.

## Implementation and validation

The Apple Spatial preview now uses AVSampleBufferAudioRenderer with a serial
renderer-driven feeder. It uses frame-exact 48 kHz timestamps, rate 1 with
varispeed (no pitch-preserving time stretching), 8192-frame startup prefill and
a 16384-frame presentation lead cap. These are provisional buffer budgets,
not measured minimum AirPods latency. Short clips explicitly start at EOF;
queued media drains before stopping. Nonblocking producer capacity checks
keep pause/seek commands responsive. Stalled clocks, renderer failure,
automatic flushes and late producer buffers surface as errors. Output-reset
recovery requires restarting preview; it does not silently discard/re-time PCM.

Removed PHASE's LFE fold and -3 dB monitor trim; the Apple media renderer now
receives the original LFE channel and nominal programme gain before transport
volume. A system-settings hint replaces the app head-tracking checkbox;
existing saved head-tracking fields are retained for request compatibility.

`apps/web/scripts/test-native-audio.sh --capture` launches a named test app
that explicitly requests macOS audio-input permission. Audio is routed and
recorded through BlackHole 2ch without changing system device selection. A
separate AudioQueue reference checks capture first, so a permission failure
cannot be mistaken for silent AVFoundation playback. The recorder must
re-enqueue its input buffers after stopping the reference recording.

Observed output after granting permission:

- 144137 frames (3.003 seconds): clock elapsed 3.037 seconds; captured active
  audio 3.01 seconds, frequency 439.20 Hz for a 440 Hz source, no internal
  silent gaps of 10 ms or more. Producer workload includes 15 ms jitter bursts.
- Start at frame 96000 with pause/resume: elapsed 3.080 seconds after excluding
  the 180 ms pause; position is stationary while paused and ends at the exact
  final sample frame.
- A 127-frame clip at frame 480000 starts and drains without prefill deadlock
  (51 ms wall time including output startup).
- Six multichannel layouts retain exact interleaved PCM samples, including
  LFE and the 7.1.4 side/rear reorder; timestamps and partial-buffer durations
  are validated independently.

The loopback establishes non-silent, correctly paced media output on this
machine, not AirPods HRTF, head tracking, or long-session stability. Realtime
integration uses the same native bridge tested here, but this test supplies
synthetic PCM rather than a full project DSP workload.

Additional checks passed for paused-queue capacity, loop continuation during
EOF drain and explicit automatic-flush errors. Frontend tests: 434 passed;
native Rust tests: 6 passed. Web production and macOS Tauri app builds passed.
The desktop app is under
`apps/web/src-tauri/target/release/bundle/macos/Upmixer.app`.

## Startup latency follow-up

Reduced startup prefill from 8192 to 2048 frames (16 to 4 blocks; 170.7 to
42.7 ms of programme audio). The 16384-frame running queue remains unchanged.
This reduces the work required before starting without reducing steady-state
buffering protection. It does not remove device or Apple spatializer latency.

In sequential BlackHole capture runs using the same synthetic workload, the
first active 10 ms window moved from frame 3360 to frame 2400 of the recording
(approximately 70 to 50 ms after capture began). This is a single comparison
including test setup and capture startup, not an AirPods play-click benchmark.
Both runs captured 3.01 seconds at 439.20 Hz for a 440 Hz input with no internal
10 ms gaps. Pause/resume, nonzero start, 127-frame EOF, queue bounds and output
reset checks passed. Web tests (434), native Rust tests (6), and web production
build also passed.

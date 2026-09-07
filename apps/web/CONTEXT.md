# Web

The web context lets a user edit a project and hear its spatial result.

## Language

**Preview**: The interactive audible representation of current project settings.
_Avoid_: mockup

**Layout-adaptive preset**: A spatial starting point that redistributes stems for the current track and selected speaker layout.
_Avoid_: fixed preset

**Anchor stem**: Lead vocal, vocal, bass, kick, or snare material that holds the front image.
_Avoid_: surround candidate

**Secondary stem**: Support or flexible musical material that can expand from the front to side, rear, and height zones as a layout becomes richer.
_Avoid_: ambience stem

**Front-centered stem**: A separated stem whose direct image remains midway between left and right to avoid directional artifacts.
_Avoid_: fixed-side stem

**Preset-owned ambient treatment**: The rear/height sends, height crossover, ambience trim, and height texture assigned by a spatial preset; reapplying a preset replaces these values. Every profile contributes a height layer when the zone is available, with Immersive and Live more pronounced.
_Avoid_: persistent ambient default

**Available spatial zone**: A front, rear, or height region represented by the selected speaker layout; a preset assigns no treatment to an unavailable zone.
_Avoid_: latent zone

**Height texture**: A high-passed direct-residual layer sent to height speakers. Presets assign it to every named stem, with strength and cutoff tailored independently; it activates only for rich, height-capable layouts.
_Avoid_: bright-stem-only enhancer

**Composer**: The workspace for editing a project mix and delivery settings.
_Avoid_: dashboard

**Production-ready spatial mix**: An editable multichannel mix that a producer or engineer can use as a final deliverable or as a credible starting point for further production.
_Avoid_: demo, mockup

**Guided workflow**: A project-editing path that starts with a layout-adaptive preset and exposes only the corrections needed to reach a credible spatial mix.
_Avoid_: DAW workflow, full mixing console

**Delivery output**: A standards-compliant multichannel WAV or ADM-BWF export with preview and downmix checks; it does not claim Dolby Atmos platform or distributor acceptance.
_Avoid_: Atmos master, platform-ready master

**Source mix**: A finished stereo or multichannel music mix supplied for spatial conversion or refinement. Mono sources are outside the current product scope.
_Avoid_: raw recording, session

**Source spatial intent**: The front, surround, height, center, and LFE placement already present in a multichannel source mix; it is the default starting point for refinement.
_Avoid_: extra stereo channels

**Project**: A container for independently edited and delivered tracks; it does not impose album-wide creative cohesion.
_Avoid_: album mix

**Technical check**: A visible assessment of a delivery output's format integrity, loudness, true peak, or downmix. Format-integrity failures block export; the other findings warn.
_Avoid_: aesthetic score

**Stem repair**: Subtle corrective treatment of a separated stem through corrective EQ, de-essing, resonance control, or gentle dynamics. It excludes creative effects and a general channel strip.
_Avoid_: stem production

**Mastering chain**: The existing mix-level reference matching, spectral EQ, compression, bass control, loudness normalization, and true-peak control applied to a delivery output.
_Avoid_: stem repair

**Movement pattern**: A preset-controlled spatial motion assigned to a stem, with per-stem enablement, depth, rate, and start/stop controls. It is not freehand or keyframed automation.
_Avoid_: timeline automation

**Motion parity**: The invariant that previewed movement renders equivalently into multichannel WAV and time-varying ADM-BWF object metadata.
_Avoid_: preview-only effect

**Intent-based preset**: A small, explainable layout-adaptive preset library organized by musical spatial intent rather than genre labels or a broad catalog.
_Avoid_: genre preset

**Preset recommendation**: An automatically applied intent-based preset accompanied by a plain-language rationale; the producer can replace it.
_Avoid_: mandatory preset choice

**Automatic stem plan**: The separation plan selected by the guided workflow and then presented for practical stem inclusion or exclusion.
_Avoid_: model configuration

**Advanced automation**: The CLI and manifest-driven workflows retained for reproducible or batch processing, outside the normal Project workspace.
_Avoid_: primary workspace

**Personal workspace**: A cloud workspace owned by one producer that contains their Projects, source assets, generated stems, and Delivery outputs. It may later be owned by an organization without changing Project ownership semantics.
_Avoid_: team workspace

**Project asset retention**: Source assets and generated artifacts remain available for editing until the producer explicitly deletes the Project, with visible storage usage.
_Avoid_: delivery-only storage

**Producer account**: The authenticated identity that owns a Personal workspace and is authorized to access every Project and its assets.
_Avoid_: anonymous project owner

**Managed delivery job**: A durable, asynchronous processing job with visible queue, progress, failure, and retry state.
_Avoid_: synchronous export request

**Recoverable deletion**: A producer-requested Project deletion that remains restorable for 30 days before its assets are permanently purged.
_Avoid_: immediate deletion

**Private audio**: Producer audio and generated artifacts used only to provide the service, never for model training without separate explicit opt-in.
_Avoid_: product telemetry

**Supported delivery layout**: Any output speaker layout the application exposes; each receives equal preset, preview, downmix, and export validation.
_Avoid_: best-effort layout

**Preset evidence**: The automated technical checks and curated listening/reference-set review required before an Intent-based preset is released.
_Avoid_: automated aesthetic score

**Reference deployment**: A supported self-hostable cloud installation operated by its deployer, rather than a shared service operated by this project.
_Avoid_: managed SaaS

**Project collaborator**: A Producer account granted owner, editor, or viewer access to one Project. Collaboration includes live presence and ordinary field updates with revision-conflict detection.
_Avoid_: anonymous collaborator

**Restore point**: An automatically retained Project mix state created at meaningful edits and before a Delivery job; it can be restored but does not create branches.
_Avoid_: version-control branch

**Delivery snapshot**: The immutable Project revision rendered by a Delivery job, retained as a Restore point even while collaborators continue editing.
_Avoid_: latest-state export

**Project membership**: Direct owner, editor, or viewer access granted to a specific Project. Organization-wide membership is deferred.
_Avoid_: inherited workspace access

**Instrument display**: A visual meter or spatial display that communicates audio state.
_Avoid_: chart

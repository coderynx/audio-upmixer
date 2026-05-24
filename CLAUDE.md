# CLAUDE.md

This file provides guidance to Claude Code (claude.ai/code) when working with code in this repository.

## Commands

```bash
# Run all tests (must stay at 394, zero regressions)
python3 -m pytest -q

# Run a single test file
python3 -m pytest tests/test_pipeline.py -q

# Run a single test by name
python3 -m pytest -q -k "test_name"

# Install in editable mode with dev deps
pip install -e ".[dev]"

# Install stem separation (CPU)
pip install -e ".[dev,separation-cpu]"

# CLI
upmixer input.wav output.wav --format 7.1.4 --mode stem
upmixer --manifest examples/atmos_music.yaml
upmixer --profile-info
upmixer --manifest-keys
```

## Architecture

Two top-level pipelines share a common mastering chain:

- **`UpmixPipeline`** (`upmixer/pipeline.py`) — realtime/file mode. For stereo/mono input: coherence-based STFT → decompose → route → master. For multichannel input: `MultichannelUpmixer` pass-through + channel derivation.
- **`StemUpmixPipeline`** (`upmixer/separation/stem_pipeline.py`) — stem mode. Separates audio into instrument stems via `audio-separator` (Demucs/RoFormer), analyzes each stem, routes spatially, then masters.

Both pipelines end at **`MasteringChain`** (`upmixer/mastering.py`), which applies in order: spectral EQ → bus compression → bass control → BS.1770-4 loudness normalization → True Peak ceiling → tanh soft-limiter.

### Key modules

| Path | Role |
|------|------|
| `upmixer/config.py` | `UpmixConfig` dataclass — all tunable params |
| `upmixer/profiles.py` | `DeliveryProfile` + `PROFILES` dict (`atmos-music`, `atmos-bluray`) |
| `upmixer/formats.py` | `FORMAT_MAP`, `INPUT_FORMAT_MAP`, `ChannelLabel` enum |
| `upmixer/manifest.py` | `load_manifest()` / `apply_manifest()` — YAML/JSON job files |
| `upmixer/result.py` | `UpmixResult` dataclass |
| `upmixer/analysis/` | `CoherenceEstimator`, `StreamingSTFT`, transient + harmonicity detection |
| `upmixer/decomposition/` | `SoftMatrixDecomposer` — direct/ambient separation in STFT domain |
| `upmixer/routing/` | `ChannelRouter` (realtime), `lfe.py` |
| `upmixer/separation/` | `StemSeparator`, `StemRouter`, `stem_analyzer`, `stem_eq`, `stem_rebalance`, `content_mixer` |
| `upmixer/io/` | `AudioReader`, `AudioWriter`, `AdmBwfWriter` (ITU-R BS.2076-2 ADM-BWF) |
| `upmixer/mastering_comp.py`, `mastering_bass.py`, `mastering_eq.py` | Public re-export shims — do not remove |

### Parameter priority

```
CLI flags > manifest values > profile defaults > UpmixConfig defaults
```

### Realtime pipeline data flow

```
AudioReader → stereo L/R → StreamingProcessor (per-hop loop):
  StreamingSTFT.analyze_frame()
  → CoherenceEstimator.estimate_frame()
  → SoftMatrixDecomposer.decompose_frame()   (direct / ambient bins)
  → ChannelRouter.route_frame()              (assign to output channels)
  → StreamingSTFT.synthesize_frame()
→ normalize_energy() → MasteringChain → AudioWriter / AdmBwfWriter
```

### Stem pipeline data flow

```
AudioReader → zone pairs (front/surround/back/height) → StemSeparator (per zone)
→ analyze_stems() → StemRouter.route() → mix_stems()
→ MasteringChain → AudioWriter / AdmBwfWriter
```

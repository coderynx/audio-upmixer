---
description: Comment policy for all Python files in upmixer/. Apply whenever writing or editing Python code in this project.
---

# Comment Policy

Strip all inline `#` comments from production Python files (`upmixer/`).

**Keep only:**
- `# noqa: X` — linter suppression (functional, not explanatory)
- `# type: ignore[X]` — untyped optional deps (`yaml`, `soundfile`)
- Hack/load-bearing comments: explain non-obvious constraints, model-specific quirks, or workarounds that would surprise a future reader

**Hack comment examples (keep these patterns):**
- ADM-BWF spec constraints: `# Dolby Atmos Music Delivery Specification v2022.07: ADM-BWF requires 48 kHz.`
- Model tag quirks: `# NOTE: verify exact tag strings if model output filenames differ`
- Pickling constraints: `# Use config.__dict__ for pickling (UpmixConfig is a plain dataclass).`
- Registry pattern notes: `# routing: and mastering: blocks are populated at import time by domain modules`

**Never add:**
- Section headers (`# ── Output format ──`, `# ── Mastering ──`)
- Explanatory prose (`# Collect all stems`, `# Apply profile first`)
- TODO/FIXME (resolve or delete)
- Commented-out code blocks

**Docstrings:** Preserve intact for all public modules, classes, and functions. Never shorten or remove.

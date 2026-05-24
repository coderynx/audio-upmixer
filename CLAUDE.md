# upmixer — Coding Rules

## Comments

No inline comments. No section headers. No explanatory prose in code.

**Exceptions (keep these):**
- `# noqa: X` — linter suppression
- `# type: ignore[X]` — on optional untyped deps (`yaml`, `soundfile`)
- Hack/load-bearing comments: non-obvious constraints, model-specific quirks, or workarounds that would confuse a future reader. Examples currently in the codebase:
  - `separator.py`: `# NOTE:` blocks about model output tag naming — specific to upstream model behavior
  - `batch.py`: pickling comment before `asdict` use in `ProcessPoolExecutor`
  - `manifest.py`: `# stem_model removed` and registration comment
  - `pipeline.py`: ADM-BWF 48 kHz spec comment

Docstrings are preserved intact for all public modules, classes, and functions.

## Imports

Every import must be referenced in the same file. No unused imports.

Exception: `# noqa: F401` shims in backward-compat files (`mastering_comp.py`, `mastering_bass.py`, `mastering_eq.py`) — intentional re-exports.

## Dead Modules

Every `.py` file must be imported somewhere in production code, or serve a documented public-API purpose (e.g. backward-compat shims). Unreachable modules are deleted.

## Function Signatures

No parameters added unless referenced in the function body.

## Backward-Compat Shims

`upmixer/mastering_comp.py`, `upmixer/mastering_bass.py`, `upmixer/mastering_eq.py` are intentional public API re-exports. Do not remove them.

## Tests

Test suites are verified with `python3 -m pytest -q` before and after any change. All 394 tests must pass.

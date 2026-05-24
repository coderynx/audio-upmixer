---
description: Dead code and module hygiene. Apply when adding new files, refactoring, or doing cleanup passes.
---

# Dead Code Policy

## Modules

Every `.py` file in `upmixer/` must be imported somewhere in production code, or have a documented public-API purpose with `# noqa: F401` shims.

Before creating a new module, confirm it will be imported. A file that nothing imports is dead on arrival.

**Known intentional modules (not imported transitively but kept for public API):**
- `upmixer/mastering_comp.py` — re-exports `MasteringCompressor`
- `upmixer/mastering_bass.py` — re-exports `MasteringBassProcessor`
- `upmixer/mastering_eq.py` — re-exports `MasteringEQ`

## Functions and Classes

Remove unused functions, classes, methods, and constants. If unsure, grep across `upmixer/` and `tests/` before deleting.

Do not add helper functions "for future use". Implement when needed.

## Function Signatures

No parameters added to functions unless referenced in the function body. Remove parameters made unused by refactoring, and update all call sites.

## Dead Branches

Delete `if/else` branches that can never be reached under current logic. Do not leave dead fallbacks "just in case".

---
description: Import hygiene rules for upmixer/. Apply when adding, removing, or reviewing imports in any Python file.
---

# Import Rules

Every import must be referenced in the same file. No unused imports.

**Exception:** Backward-compat shim files use `# noqa: F401` on re-exports — this is intentional and correct.

**Shim files (do not touch their imports):**
- `upmixer/mastering_comp.py`
- `upmixer/mastering_bass.py`
- `upmixer/mastering_eq.py`

When removing a feature or function, remove its import too. When adding a helper import, verify it is used before committing.

Test files follow the same rule: unused imports in `tests/` must be removed.

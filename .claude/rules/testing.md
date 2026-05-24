---
description: Testing requirements for upmixer. Apply before and after any refactoring, feature addition, or cleanup.
---

# Testing Rules

Baseline: **394 tests, all pass.** Any change must leave this green.

Run tests with:
```bash
python3 -m pytest -q
```

**Before any significant change:** run the full suite and note the count.

**After any change:** run again. Count must match. Zero regressions.

Test fixtures in `tests/conftest.py` must be referenced by at least one test. Remove unused fixtures (check with grep before deleting — fixture names may appear indirectly via parametrize or indirect).

Do not add test-only helpers to production `upmixer/` code.

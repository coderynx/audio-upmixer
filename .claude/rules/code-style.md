---
description: General code style for upmixer Python code. Apply when writing any new code or reviewing existing code.
---

# Code Style

## No Over-Engineering

Don't add features, abstractions, or error handling for hypothetical future requirements.

- Three similar lines beats premature abstraction
- No half-finished implementations
- No feature flags or backwards-compat shims unless replacing an existing public API

## Error Handling

Only validate at system boundaries (user input, external APIs, file I/O). Trust internal code and framework guarantees. Do not add fallbacks for scenarios that cannot happen.

## Naming

Well-named identifiers document themselves. No comments explaining what code does — only why, when non-obvious.

## No Backwards-Compat Noise

When removing code: delete it. No `_unused_var`, `# removed`, or re-export stubs unless replacing a documented public API.

When renaming: update all call sites. Do not keep old name as alias unless there is an explicit public-API contract.

## Type Annotations

Use standard Python type hints. For optional untyped deps, use `# type: ignore[import-untyped]`.

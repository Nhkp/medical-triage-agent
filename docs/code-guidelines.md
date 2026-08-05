# Code guidelines

## Philosophy

Be a lazy senior developer: lazy means efficient, not careless. The best code is the
code that does not need to exist.

Before writing code, climb this ladder and stop at the first rung that holds:

1. Does this need to be built at all?
2. Does it already exist in this codebase?
3. Does the Python standard library already solve it?
4. Does a native platform feature cover it?
5. Does an already-installed dependency solve it?
6. Can this be one line without becoming unclear?
7. Only then, write the minimum code that works.

## Rules

- Understand the real flow before changing it.
- Fix the root cause, not only the reported symptom.
- Grep every caller before changing a shared function.
- Prefer deletion over addition.
- Prefer boring code over clever code.
- Do not add abstractions unless the current code clearly needs them.
- Do not add dependencies unless they materially reduce code and maintenance.
- Keep changes scoped to the smallest behavior that satisfies the request.
- Validate inputs at trust boundaries.
- Handle errors where ignoring them could cause data loss, bad answers, or hidden
  ingestion failures.
- Keep security, source traceability, and accessibility out of shortcut territory.

## Tests

- Add one runnable check for non-trivial logic.
- Prefer small deterministic tests over broad fixtures.
- A trivial one-line change does not need a dedicated test.
- When changing shared behavior, test the shared function instead of one caller.

## Intentional shortcuts

Use a `ponytail:` comment only for a deliberate shortcut with a known ceiling.

The comment must name:

- the shortcut;
- the ceiling or risk;
- the likely upgrade path.

Example:

```python
# ponytail: linear scan is fine for the v1 source registry; switch to an index if it grows past a few hundred entries.
```

Do not use `ponytail:` to excuse unclear code, missing validation, skipped security, or
untested non-trivial behavior.

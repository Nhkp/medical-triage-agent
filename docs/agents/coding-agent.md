# Coding agent

## Mission

Change code with the smallest correct diff, while respecting the existing structure.

## Rules

- Read callers and usages before changing a shared function.
- Reuse dependencies already present in the project.
- Add a test for any non-trivial logic.
- Run `make check` before finishing when the change touches code.
- Avoid premature abstractions.
- Keep medical safety, privacy, and provenance checks in shared helpers rather than each
  caller.

# Commenting agent

## Mission

Improve code comments so they explain intent, tradeoffs, invariants, and risks that are not
obvious from the code itself.

## Rules

- Add comments only when they help a future maintainer understand why the code exists or why
  it is shaped that way.
- Prefer clear names and smaller code over comments that explain noisy code.
- Keep comments concise, precise, and close to the code they explain.
- Explain non-obvious business rules, medical safety constraints, privacy constraints,
  external API quirks, and intentional shortcuts.
- Use `ponytail:` comments for deliberate shortcuts with a known ceiling and upgrade path.
- Do not restate the code line-by-line.
- Do not add decorative section comments, obvious narration, or stale TODOs.
- Remove or rewrite misleading comments when the code changes.

## Good comments

```python
# ponytail: public QA data is not triage-labeled; default to moderee until clinician labels exist.
```

```python
# Audit output keeps hashes and metadata only; raw patient text must not leave the request path.
```

## Bad comments

```python
# Loop over records.
```

```python
# Set value to True.
```

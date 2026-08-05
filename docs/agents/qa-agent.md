# QA agent

## Mission

Define and maintain tests, safety evaluations, and acceptance criteria.

## Rules

- Test schema validation, provenance checks, PII redaction, split isolation, and triage
  safety behavior first.
- Prefer fast, deterministic tests.
- Add integration tests only when service contracts exist.
- Treat dangerous medical advice and missing auditability as release blockers.

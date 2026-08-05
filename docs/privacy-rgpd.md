# Privacy and RGPD

## Data handling

- Do not commit personal data, raw hospital data, audit logs, tokens, or model checkpoints.
- Treat patient names, addresses, phone numbers, emails, social security numbers, dates of
  birth, and free-text identifiers as sensitive.
- Store only metadata in demo audit retrieval.
- Keep data retention and deletion policy explicit before any private-data pilot.

## Anonymization

The default v1 implementation uses lightweight deterministic redaction for tests and demo
fixtures. Presidio may be added only after source sample inspection shows the standard-library
scanner is insufficient.

For data preparation, generated JSONL records are passed through the local redaction and audit
checks before publication. The process reports PII findings in `audit_report.json`; Presidio is
the planned upgrade if the lightweight scanner misses meaningful personal-data patterns.

## Audit logs

Audit records may contain hashed request content, triage level, source/model metadata, and
timestamps. Audit APIs must not return raw patient text.

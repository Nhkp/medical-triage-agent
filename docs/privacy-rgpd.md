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

## Repository credential review — 2026-10-07

The review inspected 650 named Git objects across all locally available refs for long
Hugging Face token literals, credential assignments, and private-key headers. No matching
secret was found in that history. The current working files are checked separately before
handoff. Pattern checks do not prove that every possible credential format is absent or
that a remote-only ref is clean. `.env`, generated data, and presentation exports are
ignored by Git; `.env.example` keeps credential values empty.

The local HF/API token values were exposed during an earlier terminal inspection.
The account owner must revoke and replace `HF_TOKEN` and `TRIAGE_API_TOKEN`, update local
`.env` and GitHub Actions secrets, then redeploy to refresh the named Modal secrets.
Do not reuse the exposed values in GitHub. Rotation has not been performed by this agent.
No history rewrite is indicated by the current findings.

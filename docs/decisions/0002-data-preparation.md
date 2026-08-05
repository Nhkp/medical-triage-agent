# 0002 - Data preparation process

## Status

Accepted.

## Decision

The local data-preparation milestone builds SFT and DPO JSONL artifacts from verified public
Hugging Face sources using `make data-ready`. The generated artifacts stay under
`data/processed/training` and are ignored by git.

## Source mix

- SFT starts with MediQAl and FrenchMedMCQA French records, then fills the remaining target
  count with English MedQuad records.
- DPO uses UltraMedical-Preference preference pairs.
- Public QA and preference sources are not clinician-validated CHSA triage labels.

## Privacy and review

The v1 pipeline uses deterministic redaction and audit checks, then produces a clinician
review queue for records that need medical validation. Presidio remains the planned upgrade if
PII audit findings show the lightweight scanner is insufficient.

## Completion evidence

On `2026-08-05T17:59:36Z`, `make data-ready` produced the local step-1 dataset under
`data/processed/training`.

- Accepted records: `5,000` SFT and `1,000` DPO.
- Languages: `3,406` English and `2,594` French records.
- Sources: MediQAl `2,000`, FrenchMedMCQA `594`, MedQuad `2,406`,
  UltraMedical-Preference `1,000`.
- Rejected source rows: MediQAl `763`, FrenchMedMCQA `1`, MedQuad `53`,
  UltraMedical-Preference `2`.
- Audit result: passed, with `0` PII findings, `0` duplicate findings, and `0` missing
  provenance findings.
- Clinical review queue: `5,000` records.

The generated dataset completes the technical data-preparation milestone. It still must not be
described as clinician-validated CHSA triage data.

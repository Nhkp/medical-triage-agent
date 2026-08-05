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

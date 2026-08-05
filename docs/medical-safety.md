# Medical safety

## Scope

The assistant supports initial emergency triage for clinical staff. It does not diagnose,
prescribe, discharge, or replace clinician judgment.

## Required behavior

- Include a safety disclaimer in triage responses.
- Escalate red-flag symptoms to immediate human review.
- Explain uncertainty plainly.
- Prefer conservative triage when severe symptoms are present.
- Refuse dangerous treatment instructions and redirect to clinical staff.

## Red flags for v1

Chest pain, severe breathing difficulty, stroke signs, loss of consciousness, severe
bleeding, major trauma, suicidal intent, anaphylaxis signs, seizure, and severe burns must
produce `urgence_maximale`.

## Release blockers

- Any response that tells a patient not to seek care for red-flag symptoms.
- Missing audit ID on triage output.
- Missing safety disclaimer.
- Raw patient text exposed through audit retrieval.

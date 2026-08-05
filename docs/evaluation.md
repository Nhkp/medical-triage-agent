# Evaluation

## Acceptance checks

- Schema validation passes for SFT and DPO JSONL.
- All records keep source IDs, license metadata, and transform history.
- No duplicate IDs or content hashes in a split.
- No overlap between train, validation, test, and clinical evaluation IDs.
- Red-flag symptoms produce `urgence_maximale`.
- Triage responses include explanation, disclaimer, and audit ID.

## Model evaluation

Track these metrics after SFT and DPO:

- clinical safety pass rate;
- dangerous-advice refusal rate;
- hallucination rate on unknown or ambiguous symptoms;
- bilingual French/English response quality;
- latency p50 and p95 for realistic prompts;
- traceability completeness.

The POC cannot be presented as production-ready unless a clinician validates evaluation
examples and thresholds.

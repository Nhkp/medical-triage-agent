# CHSA medical triage POC report

## Executive summary

TODO: summarize the final POC results, clinical value, and limits.

## Dataset preparation

Current status: completed locally for the POC technical milestone.

- MediQAl is verified from Hugging Face metadata as public, ungated, French medical QA,
  license `cc-by-4.0`.
- UltraMedical-Preference is verified from Hugging Face metadata as public, ungated, English
  medical preference data, license `mit`.
- FrenchMedMCQA is manually verified by the project owner as `apache-2.0`.
- MedQuAD is manually verified by the project owner as `apache-2.0`.

Generated local artifacts:

- Generation timestamp: `2026-08-05T17:59:36Z`.
- Output folder: `data/processed/training`.
- SFT records: `5,000`.
- DPO records: `1,000`.
- Language counts: `3,406` English records and `2,594` French records.
- Source counts: MediQAl `2,000`, FrenchMedMCQA `594`, MedQuad `2,406`,
  UltraMedical-Preference `1,000`.
- Rejected source rows: MediQAl `763`, FrenchMedMCQA `1`, MedQuad `53`,
  UltraMedical-Preference `2`.
- Audit status: passed with `0` PII findings, `0` duplicate findings, and `0` missing
  provenance findings.

Split counts:

| kind | train | validation | test | clinical_eval |
| --- | ---: | ---: | ---: | ---: |
| SFT | 4,001 | 490 | 357 | 152 |
| DPO | 791 | 111 | 65 | 33 |

Content hashes:

| file group | train | validation | test | clinical_eval |
| --- | --- | --- | --- | --- |
| SFT | `1f252e086b0b57b31dda5d43a9180f7c22433c602b75deac83e5db501a179a89` | `f891e59a39debb72a389d0ebd619a008c80f40e6a59314701fc66160d1ffa404` | `0b3d3db6af4f662a222d245a1b9602e7ac8ee01e3032a38833ddbe18977bb76b` | `97af7891b655db90bdc0c0e8484420d430dd714e3fd345f91a85e8b1230a5f9e` |
| DPO | `f5f2be54f29ec4ace656e384a542f08fbb880cecacd9739f13b223322822d16d` | `e66744a5153cc9e6d0601443430afc41455a699c94e4af01356300e411c157e2` | `4638e5ed031c574354a9319c27ec89dcd36483e6d490817d86ec3ac2bbc7313b` | `a4bed2fb0ee561d77edb2ce94835f0f654cf6d6772bba4b9b7d1d02638b8bb83` |

Clinical review queue:

- `5,000` records are queued for clinician review because public medical QA sources are not
  CHSA triage-labeled data.
- This queue is evidence of validation debt, not completed clinician sign-off.

Hugging Face publication:

- Private dataset publication completed on Hugging Face:
  `https://huggingface.co/datasets/Lokhidor/medical-triage-dataset`.
- Hub commit: `170e68c354e78fac153b732d6fa8a0ce6e269fa3`.
- Published files: SFT/DPO train, validation, test, clinical-evaluation JSONL splits,
  `manifest.json`, and the dataset card `README.md`.
- Local `audit_report.json` and `clinical_review_queue.jsonl` were not uploaded.
- Repository visibility was verified as private after upload.

Local data-preparation command:

```bash
make data-ready
```

Generated artifacts are written under `data/processed/training` and are intentionally not
committed to git.

## Training

Current status:

- Kaggle-oriented QLoRA configuration exists for SFT, DPO, and optional GRPO.
- Training scripts support CPU-safe dry runs, YAML configs, CLI overrides, adapter outputs,
  and checkpoint resume arguments.
- Full SFT/DPO runs remain TODO until generated data and GPU runtime are available.

Smoke commands:

```bash
make train-sft-smoke
make train-dpo-smoke
make train-grpo-smoke
```

## Evaluation

Current local safety evaluation command:

```bash
uv run python -m medical_triage_agent evaluate-safety
```

TODO: add model-backed safety, hallucination, bilingual quality, latency, and traceability
metrics after smoke training.

Deterministic model-evaluation entrypoint:

```bash
make eval-models
```

## Deployment

Current API supports a rule-based fallback and optional vLLM-compatible chat completions via
`VLLM_BASE_URL`, `VLLM_MODEL_ID`, `VLLM_TIMEOUT_SECONDS`, and `API_KEY`.

## Roadmap

TODO: document go/no-go criteria and production requirements.

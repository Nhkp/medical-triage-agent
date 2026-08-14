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

Current status: completed for the Step 2 technical milestone.

- The SFT and DPO full runs were executed from the Colab/T4 training notebook.
- Training used the Kaggle/Colab-oriented 4-bit QLoRA path with Qwen3-1.7B-Base, LoRA
  adapters, fixed YAML configs, and the generated SFT/DPO JSONL splits.
- CPU-safe smoke commands remain available for startup validation, but the recorded metrics
  below come from full 1-epoch GPU runs.

Published adapter repositories:

- SFT adapter: <https://huggingface.co/Lokhidor/medical-triage-qwen3-sft-lora>
- DPO adapter: <https://huggingface.co/Lokhidor/medical-triage-qwen3-dpo-lora>

SFT run summary:

| metric | value |
| --- | ---: |
| epochs | `1` |
| train runtime | `3,660s` (~61 min) |
| train samples/second | `1.093` |
| train steps/second | `0.137` |
| train loss | `1.445` |
| eval loss | `1.294` |
| eval mean token accuracy | `0.7176` |
| eval runtime | `102.4s` |
| eval samples/second | `4.785` |

DPO run summary:

| metric | value |
| --- | ---: |
| epochs | `1` |
| train runtime | `1,314s` (~22 min) |
| train samples/second | `0.595` |
| train steps/second | `0.075` |
| train loss | `0.5962` |
| eval loss | `0.5695` |
| eval mean token accuracy | `0.7074` |
| eval rewards accuracy | `0.7091` |
| eval rewards margin | `0.4726` |
| eval rewards chosen | `0.02491` |
| eval rewards rejected | `-0.4477` |
| eval runtime | `82.8s` |

Interpretation:

- The SFT run shows the model learned the supervised response format over the prepared
  dataset, with validation token accuracy around `0.72`.
- The DPO run shows the preference objective separating chosen from rejected answers on the
  validation split, with positive reward margin and roughly `0.71` reward accuracy.
- These are technical training indicators only. They do not prove clinical safety,
  diagnostic validity, or CHSA protocol compliance.

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

Current model-backed evaluation status:

- Training losses and preference metrics are recorded from the Colab full runs.
- Base vs SFT vs DPO deterministic generation evaluation remains to be run against the
  published adapters.
- Clinical safety, hallucination, bilingual quality, latency, and traceability metrics must be
  regenerated against the served DPO adapter before the final go/no-go decision.

Deterministic model-evaluation entrypoint:

```bash
make eval-models
```

## Deployment

Current API supports LLM-assisted triage suggestions through a vLLM-compatible chat endpoint via
`VLLM_BASE_URL`, `VLLM_MODEL_ID`, `VLLM_TIMEOUT_SECONDS`, and `API_KEY`. The LLM returns a
structured priority suggestion, explanation, and confidence; the FastAPI wrapper keeps final
authority by applying a conservative rule-based safety floor before returning the final priority.
This is not autonomous triage and still requires clinician review.

Step 3 local deployment status:

- Docker Compose defines a `vllm` OpenAI-compatible model server and a FastAPI CHSA wrapper.
- `make serve-api` runs the wrapper alone with rule-based fallback or an external vLLM URL.
- Local FastAPI serving without vLLM has been exercised successfully.
- `make serve-local` starts the local GPU-oriented Compose demo.
- `make eval-robustness` checks empty payload handling, red-flag escalation, bilingual inputs,
  and metadata-only audit behavior.
- `make eval-latency` records p50/p95 latency and response-size indicators under
  `outputs/evaluations`.
- `make step3-ready` runs the full local gate plus robustness and latency checks.

Current limitation: vLLM plus FastAPI still needs to be tested on Colab/T4 or another GPU
runtime with the published DPO adapter. Local non-vLLM serving and in-process fallback metrics
are useful smoke evidence, but they are not the final model-backed latency/robustness result.

## Roadmap

Go/no-go before any pilot exposure:

- Final SFT/DPO adapter repositories are available and versioned.
- vLLM endpoint passes health, robustness, and latency checks with the selected adapter.
- Audit retrieval remains metadata-only with no raw patient text.
- Clinical reviewer signs off on evaluation prompts, thresholds, and observed failure modes.
- Secrets, audit logs, model caches, and generated outputs remain outside git.

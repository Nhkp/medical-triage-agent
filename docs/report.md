# CHSA medical triage POC report

## Executive summary

TODO: summarize the final POC results, clinical value, and limits.

## Main experiment: 8k dataset

The main school submission uses `8,000` SFT records and `2,000` DPO pairs. This extends
the approximately 5,000 SFT examples requested in `input/step_1.md`; the initial 5k
experiment remains documented below with its own results.

Evidence: `data/processed/training/manifest.json` and `audit_report.json`, generated at
`2026-08-20T10:48:52Z`. These local artifacts are ignored by git.

- Languages: `5,455` English and `4,545` French records, counting SFT and DPO together.
- Sources: MediQAl `3,951`, FrenchMedMCQA `594`, MedQuad `3,455`,
  UltraMedical-Preference `2,000`.
- Rejected rows: MediQAl `1,018`, FrenchMedMCQA `1`, MedQuad `54`,
  UltraMedical-Preference `8`.
- Automated audit passed: `0` PII, duplicate, and missing-provenance findings.
- Review queue: `8,000` records requiring professional review before clinical use.

| kind | train | validation | test | clinical_eval |
| --- | ---: | ---: | ---: | ---: |
| SFT | 6,361 | 809 | 589 | 241 |
| DPO | 1,588 | 215 | 137 | 60 |

SHA-256 content hashes from the 8k manifest:

| file group | train | validation | test | clinical_eval |
| --- | --- | --- | --- | --- |
| SFT | `2e53258d7932e872dfab53607d7835782197c9c822b72e5f85475c8f2c7ffb37` | `d40444ba13d1ed46f0446945cc115eb567d46d7ef7a9cca8ce4dbb9826ce1646` | `3b000fe46a10b872daf47dc5656a2cab16c156e44ea0cf448d9cdfee7f3acb11` | `d329f2c88fa0776ac02b0a8fc199eefa2e8da107a557e1258c711867f79311bb` |
| DPO | `86b2b252b66767f01c130679fdeb20a59464c10e1096eb1f297ddef00a510b07` | `15c011716b29bc6410f1175d6d15d64ae84fdcf98ca4b87370a325ca2b25ed56` | `2f2f65ab2332fe5f76eab18a709673d329d6db20bb8ad9da3b1679bc2cc4c135` | `ccbbb08bd1734918507de882816ac7ae71c9059d6c2f3db6353d5df85836381d` |

8k Hub references configured in the Makefile and Modal app:

- Dataset: `Lokhidor/medical-triage-dataset-8k`.
- SFT: `Lokhidor/medical-triage-qwen3-sft-lora-8k`.
- DPO: `Lokhidor/medical-triage-qwen3-dpo-lora-8k`.
- The Modal configuration records adapter revision
  `9f1c83f91064a8b464bc9e8b87f91231c617b2e7`; this is a configured reference,
  not evidence that the live Hub publication was verified during this review.
- Local adapters and checkpoint logs exist under `outputs/experiments/dataset-8k`.
  The available local evidence does not establish current Hub availability or visibility.

Reproduce the published-data workflows with `make data-pull-8k`, `make data-audit-8k`,
and `make train-8k`; use the equivalent `*-5k` commands for the historical experiment.
`make data-ready` retains its historical 5k/1k defaults and can overwrite the local
`data/processed/training` folder; it does not reproduce the current 8k snapshot by default.

## Historical experiment: 5k dataset preparation

Current status: completed locally for the POC technical milestone.

- MediQAl is verified from Hugging Face metadata as public, ungated, French medical QA,
  license `cc-by-4.0`.
- UltraMedical-Preference is verified from Hugging Face metadata as public, ungated, English
  medical preference data, license `mit`.
- FrenchMedMCQA is manually verified by the project owner as `apache-2.0`.
- MedQuAD is manually verified by the project owner as `apache-2.0`.

Historical artifacts recorded in the original report (the current local folder is 8k):

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

Historical Hugging Face publication recorded in the original report (not reverified live):

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

## Main experiment: 8k training evidence

These validation metrics come from the final one-epoch local checkpoint logs, not from
the historical Colab 5k run or a new model-comparison evaluation.

| metric | SFT, step 398 | DPO, step 99 |
| --- | ---: | ---: |
| eval loss | `2.615474` | `0.558291` |
| eval mean token accuracy | `0.711297` | `0.707596` |
| eval rewards accuracy | — | `0.724299` |
| eval rewards margin | — | `0.531351` |

Evidence: `outputs/experiments/dataset-8k/sft/checkpoint-398/trainer_state.json` and
`outputs/experiments/dataset-8k/dpo/checkpoint-99/trainer_state.json`. Adapter directories
are `outputs/experiments/dataset-8k/sft` and `outputs/experiments/dataset-8k/dpo`;
these paths do not prove that the final adapters match a published Hub revision.
The five-step smoke checkpoints are excluded. These indicators do not establish
clinical safety or comparability across different datasets and training settings.

## Historical experiment: 5k training

Historical status: completed for the initial 5k Step 2 technical milestone. The following
metrics belong to that run and its original, unsuffixed adapter repositories.

- The SFT and DPO full runs were executed from the Colab/T4 training notebook.
- Training used the Kaggle/Colab-oriented 4-bit QLoRA path with Qwen3-1.7B-Base, LoRA
  adapters, fixed YAML configs, and the generated SFT/DPO JSONL splits.
- CPU-safe smoke commands remain available for startup validation, but the recorded metrics
  below come from full 1-epoch GPU runs.

Historical adapter publication references recorded in the original report:

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

- Historical 5k Colab metrics and main 8k checkpoint metrics are reported separately above.
- The initial Base/SFT/DPO generation campaign exposed serving failures; a corrected
  comparison against the same pinned adapters remains to be run.
- Clinical safety, hallucination, bilingual quality, latency, and traceability metrics must be
  regenerated against the served DPO adapter before the final go/no-go decision.

Deterministic model-evaluation entrypoint:

```bash
make eval-models
```

Model calibration comparison:

- The Kaggle notebook now prepares a pinned Base/SFT-8k/DPO-8k campaign with 48 paired
  synthetic cases, two warmups and three measured passes per model.
- The evaluator checks served identity and separately reports usable model suggestions
  before arbitration and system outcomes after backend rules. The prompt supplies rule
  priority, so these are POC-context indicators rather than independent clinical reasoning.
- Dated campaign exports include provenance, per-language metrics, errors/fallbacks,
  nearest-rank p50/p95 latency, response stability and a pedagogical review grid.
- The local normalized overlap check passed against both current 8k training splits.
- The initial campaign `20261008T061839Z-8k` returned three reports (144 requests/model)
  on two Tesla T4 GPUs, code revision `c217273add3036b3d0588ad93178101f46ff55b1`.
  Its serving configuration is technically inadequate for model selection; corrected
  measurements and 144 first-pass annotations remain pending.
- See `docs/evaluation.md` for setup, resume/export steps, rubric and conditional selection.
  Training loss metrics and in-process fallback timing do not replace these campaign results.

### Initial 8k campaign and serving remediation

The immutable manifest and three local JSON reports for `20261008T061839Z-8k` are the
source of the following historical measurements. “Accepted” includes safety validation;
HTTP latency includes unusable model outputs and is not accepted-model performance.

| Model | Measured requests | Accepted suggestions | Backend red-flag recall | HTTP p50 / p95 |
| --- | --- | --- | --- | --- |
| Base | 144 | 2/144 (1.39%) | 90% | 3.64 / 3.72 s |
| SFT-8k | 144 | 0/144 | 90% | 9.85 / 10.84 s |
| DPO-8k | 144 | 0/144 | 90% | 9.67 / 9.82 s |

All three used seed 42, temperature zero and a 110-token cap. Server logs recorded ignored
`structured_outputs`; SFT/DPO also fell back from their incompatible tokenizer configuration.
The adapters were loaded, but these failures prevent attributing a reliable improvement to
fine-tuning. The two rule misses were French anaphylaxis and major trauma. Backend protection
must be distinguished from model safety. Missing finish reason/token counts prevent estimating
generation-length failures; a storage preview flag does not establish generation completeness.
The pedagogical review remains pending, so no hallucination or clinical-validation claim is made.

The correction retains vLLM 0.10.2, Transformers 4.56.2 and the exact Base/SFT/DPO revisions
listed in `docs/evaluation.md`. `guided_json` is sent from the first request with a common
256-token cap. Audit/report metadata separate complete JSON/schema validity, safety acceptance,
raw priority, accepted pre-arbitration priority and final backend priority. Compatible adapter
copies preserve all weights and prove vocabulary/token/prompt equivalence before serving.
A CPU check with Transformers 4.56.2 verified both local tokenizers (151,669 vocabulary entries)
and all 48 prompt encodings. This is a local compatibility check, not GPU performance evidence.
The shared rules now recognize “traumatisme majeur” and “anaphylaxie”, including accented forms
of existing French signals; fixture labels are unchanged.

A separate repaired CSV under `outputs/reviews/20261008T061839Z-8k/` restores input `language`
and reserves `language_quality` for annotation without changing any initial campaign file.
Each corrected model must pass six technical probes before two warmups and 144 measured calls.
Safety rejections remain observations; technical failures block full measurement. A new dated
campaign and its returned archive are required for comparison, review and model selection.
Professional clinical validation remains outside the school scope.

## Deployment

Current API supports LLM-assisted triage suggestions through a vLLM-compatible chat endpoint via
`VLLM_BASE_URL`, `VLLM_MODEL_ID`, `VLLM_TIMEOUT_SECONDS`, `VLLM_STRUCTURED_OUTPUT`,
`VLLM_MAX_TOKENS`, and `API_KEY`. The LLM returns a
structured priority suggestion, explanation, and confidence; the FastAPI wrapper keeps final
authority by applying a conservative rule-based safety floor before returning the final priority.
This is not autonomous triage and still requires clinician review.

The `deploy-demo` GitHub Actions workflow is manually triggered. It runs the base quality
gate, Docker build and Compose validation before deploying the existing Modal app, then
polls `/health` for at most fifteen minutes. See `docs/modal-deployment.md` for secrets
and URL configuration. A successful live Actions run and endpoint probe have not yet
been recorded for this workflow; implementation alone is not deployment evidence.

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

## School submission and professional clinical validation

Technical validation has been performed through dataset audits and local safety/API checks.
Professional clinical validation has not been performed. Public medical QA/preference data,
automated tests, and a review queue do not constitute clinician sign-off.

This is an explicit gap against the request for clinically validated DPO pairs in
`input/step_1.md`. For this school submission, it is a documented limitation and a future
clinical-pilot gate, rather than an additional implementation task requiring a clinical panel.
This does not establish that the grading requirement is fully satisfied.

Pedagogical review grid (criteria, not a claim that professional review occurred):

| criterion | evidence to inspect |
| --- | --- |
| Provenance | source IDs, license, transform history and split isolation |
| French/English consistency | equivalent symptoms retain coherent intent and priority |
| Red flags | severe symptoms escalate to immediate human attention |
| Uncertainty | limitations and confidence remain visible |
| Human decision | no autonomous diagnosis, prescription or discharge |

Final model-backed comparisons and GPU latency/robustness measurements remain separate
pending work. Do not substitute local fallback results for those measurements.

## Roadmap

Go/no-go before any pilot exposure:

- Final SFT/DPO adapter repositories are available and versioned.
- vLLM endpoint passes health, robustness, and latency checks with the selected adapter.
- Audit retrieval remains metadata-only with no raw patient text.
- Clinical reviewer signs off on evaluation prompts, thresholds, and observed failure modes.
- Secrets, audit logs, model caches, and generated outputs remain outside git.

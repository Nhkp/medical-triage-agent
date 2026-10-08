# Medical Triage Agent

## Abstract

This repository implements a CHSA proof of concept for a bilingual medical triage assistant.
The current system is a reproducible scaffold for public-data preparation, privacy checks,
local safety evaluation, SFT/DPO training scripts, and a thin FastAPI serving layer. It is not
a production medical device and does not replace clinician judgment.

## Introduction

Emergency triage requires fast synthesis of symptoms, context, and red-flag signals. This POC
explores whether an LLM can support clinical staff during that first assessment while keeping
provenance, privacy, auditability, and human review visible throughout the workflow.

The project starts from public medical QA and preference datasets because no private CHSA
clinical labels are available in the repository. Those sources are useful for a technical POC,
but they are not clinician-validated CHSA triage labels.

## Objective

The target assistant should help clinical staff collect patient-declared symptoms, identify
escalation signals, and produce a triage-oriented explanation with an audit identifier. It must
not produce autonomous diagnoses, treatment decisions, or patient-facing final recommendations.

## Methodology

The data-preparation pipeline registers public sources, checks license and provenance metadata,
normalizes records into stable SFT and DPO JSONL schemas, applies lightweight PII screening,
creates deterministic train/validation/test/clinical-evaluation splits, and writes generated
artifacts outside git.

The model plan follows a staged alignment workflow: Qwen3-1.7B-Base is adapted with 4-bit
QLoRA SFT, then aligned with DPO preference data. Kaggle free GPUs are the low-VRAM execution
path for the first experiments, while Hugging Face remains the publication target for private
datasets and LoRA adapters. Full training artifacts are not claimed until run logs and model
repositories exist.

Serving is designed around a vLLM OpenAI-compatible backend behind a thin FastAPI wrapper. The
wrapper owns domain validation, safety disclaimers, conservative priority arbitration,
metadata-only audit traces, and the stable `/health`, `/triage`, and `/audit/{id}` API surface.

## Architecture

The global flow is documented in [`docs/architecture.md`](docs/architecture.md): verified
public medical sources feed the data-preparation pipeline, SFT produces an instruction-tuned
adapter/model, DPO produces the aligned medical-triage model, and vLLM plus FastAPI expose the
demonstration API with audit metadata and safety evaluation.

## Current Results

- `make check` is the single local and CI quality gate for linting, formatting checks, typing,
  tests, and source-registry validation.
- The source registry includes MediQAl, FrenchMedMCQA, MedQuad, and UltraMedical-Preference
  with license and provenance tracking.
- `make data-ready` can build local SFT/DPO split artifacts, a manifest, an audit report, a
  clinical review queue, and a generated dataset card under `data/processed/training`.
- `make data-audit` validates schema, provenance, duplicate content, split isolation, and
  obvious PII findings for generated data.
- The API scaffold exposes `/health`, `/triage`, and `/audit/{id}` with metadata-only audit
  responses, optional vLLM priority suggestions, and backend safety arbitration.
- SFT, DPO, and optional GRPO scripts have YAML configs and CPU-safe dry-run smoke commands
  before full GPU jobs.

## Reproducibility

Install the base development environment and run the quality gate:

```bash
uv sync
make check
```

Build and inspect local training-data artifacts:

```bash
make data-ready
make data-audit
make data-summary
```

The main experiment contains `8,000` SFT records and `2,000` DPO pairs, extending the
approximately 5,000 SFT examples requested for the school project. The local manifest
(`2026-08-20T10:48:52Z`) records `5,455` English and `4,545` French examples. Sources,
splits, SHA-256 hashes, and separate 5k/8k training metrics are recorded in
[`docs/report.md`](docs/report.md). Generated artifacts are ignored by git.

The 5k/1k experiment remains historical. `make data-ready` retains those defaults and can
overwrite the current local 8k artifacts. Use the separate published-data workflows:

```bash
make data-pull-8k
make data-audit-8k
make train-8k
# Historical experiment: data-pull-5k, data-audit-5k, train-5k.
```

Hub IDs are configured for both experiments; current 8k publication availability and
visibility have not been verified from the available local evidence.

Remove generated training-data artifacts:

```bash
make data-clean
```

Install optional serving dependencies:

```bash
uv sync --extra serving
```

Install optional training dependencies:

```bash
uv sync --extra training
```

Run the local safety evaluation:

```bash
uv run python -m medical_triage_agent evaluate-safety
```

Validate the local Step 3 serving path without a GPU:

```bash
make step3-ready
```

Launch the FastAPI wrapper only:

```bash
make serve-api
```

Use the optional Streamlit API tester in a second terminal:

```bash
make serve-ui
```

The tester defaults to `http://127.0.0.1:8080`; change the sidebar URL to your
Kaggle/ngrok FastAPI URL when testing a remote session.

Launch the Docker Compose vLLM + FastAPI demo on a GPU host:

```bash
make serve-local
```

Validate Kaggle training and model-evaluation startup without loading weights:

```bash
make train-sft-smoke
make train-dpo-smoke
make train-grpo-smoke
make eval-models
```

## Model comparison on Kaggle

The existing [comparison notebook](notebooks/kaggle_model_comparison.ipynb) evaluates
Base/SFT-8k/DPO-8k sequentially on 48 synthetic bilingual cases, with pinned snapshots,
three measured passes and separate model/backend indicators. Run
`make eval-model-comparison-dry-run` locally; use Kaggle for GPU execution. See
[the evaluation protocol](docs/evaluation.md) for credentials, campaign exports and the
pedagogical review grid. The initial campaign `20261008T061839Z-8k` produced 144
requests per model but exposed ignored JSON constraints and tokenizer fallback; it does
not support a model recommendation. The corrected notebook retains the same revisions,
uses `guided_json`, 256 tokens and six technical probes per model before measurement.
Corrected GPU measurements and reviewed conclusions remain pending. Initial exports stay
unchanged; the repaired review is written separately under `outputs/reviews/`.

## Demonstration deployment

The `deploy-demo` GitHub Actions workflow runs checks, builds Docker, validates Compose,
deploys to Modal on manual launch, and polls `/health` for up to fifteen minutes.
Configure the four GitHub secrets and the `MODAL_URL` variable described in
[`docs/modal-deployment.md`](docs/modal-deployment.md). `make deploy` remains available locally.
A successful live workflow and endpoint check are still required as deployment evidence.

## Limitations

This repository does not contain private hospital data, clinician-validated CHSA triage labels,
trained model checkpoints, production audit logs, or deployment credentials. Generated datasets,
model artifacts, Hugging Face caches, and audit outputs must stay outside git.

Technical validation has been performed through local audits and safety/API tests;
professional clinical validation has not been performed. The review queue is preparation
for future professional review. This is a documented limitation of the school submission,
including an explicit gap against the requested clinically validated DPO pairs, and a gate
before any clinical pilot. The pedagogical review grid is in `docs/report.md`. Any demo output must remain human-reviewed and must preserve
uncertainty, escalation rules, and safety disclaimers.

## Repository Map

- `src/medical_triage_agent`: dataset contracts, ingestion, audits, safety evaluation,
  training entrypoints, and API code.
- `tests`: unit and integration tests for schemas, audits, CLI commands, API behavior, and
  documentation contracts.
- `docs`: guardrails, source policy, privacy/RGPD policy, evaluation policy, architecture,
  agent roles, workflow notes, and report material.
- `scripts`: local quality-gate helpers.
- `data/processed/training`: generated local dataset artifacts, intentionally ignored by git.

## References

- [`AGENTS.md`](AGENTS.md): repository-wide agent and safety instructions.
- [`docs/source-policy.md`](docs/source-policy.md): accepted source and provenance rules.
- [`docs/data-sources.md`](docs/data-sources.md): public dataset registry.
- [`docs/privacy-rgpd.md`](docs/privacy-rgpd.md): privacy and RGPD constraints.
- [`docs/medical-safety.md`](docs/medical-safety.md): medical safety boundaries.
- [`docs/evaluation.md`](docs/evaluation.md): evaluation policy and acceptance targets.
- [`docs/kaggle-workflow.md`](docs/kaggle-workflow.md): free-GPU QLoRA workflow.
- [`docs/colab-workflow.md`](docs/colab-workflow.md): Colab T4 QLoRA workflow.
- [`docs/report.md`](docs/report.md): longer technical report scaffold.

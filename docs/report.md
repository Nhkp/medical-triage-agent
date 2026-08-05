# CHSA medical triage POC report

## Executive summary

TODO: summarize the final POC results, clinical value, and limits.

## Dataset preparation

Current status:

- MediQAl is verified from Hugging Face metadata as public, ungated, French medical QA,
  license `cc-by-4.0`.
- UltraMedical-Preference is verified from Hugging Face metadata as public, ungated, English
  medical preference data, license `mit`.
- FrenchMedMCQA is manually verified by the project owner as `apache-2.0`.
- MedQuAD is manually verified by the project owner as `apache-2.0`.

TODO: add generated split counts, manifest hashes, anonymization findings, and Hub dataset URL.

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

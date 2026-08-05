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

TODO: run `MAX_STEPS=5` SFT/DPO smoke jobs after `HF_TOKEN`, `HF_DATASET_REPO`,
`HF_DPO_DATASET_REPO`, `HF_SFT_MODEL_REPO`, and `HF_DPO_MODEL_REPO` are configured.

## Evaluation

Current local safety evaluation command:

```bash
uv run python -m medical_triage_agent evaluate-safety
```

TODO: add model-backed safety, hallucination, bilingual quality, latency, and traceability
metrics after smoke training.

## Deployment

Current API supports a rule-based fallback and optional vLLM-compatible chat completions via
`VLLM_BASE_URL`, `VLLM_MODEL_ID`, `VLLM_TIMEOUT_SECONDS`, and `API_KEY`.

## Roadmap

TODO: document go/no-go criteria and production requirements.

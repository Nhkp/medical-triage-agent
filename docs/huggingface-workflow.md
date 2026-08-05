# Hugging Face workflow

## Dataset publication

1. Verify source status in `docs/data-sources.md`.
2. Generate normalized JSONL outside git.
3. Run `uv run python -m medical_triage_agent audit-jsonl sft <path>` or `dpo <path>`.
4. Publish anonymized train, validation, test, and clinical evaluation splits to a private
   Hugging Face dataset repo.
5. Add a dataset card with source IDs, licenses, transforms, and known limits.

## Training jobs

Use Hugging Face Jobs with `HF_TOKEN` passed as a secret. Job environments are ephemeral, so
scripts must push checkpoints and final adapters to the Hub.

Recommended smoke run:

```bash
uv run scripts/train_sft.py --config configs/sft_kaggle.yaml --max-steps 5 --dry-run
uv run scripts/train_dpo.py --config configs/dpo_kaggle.yaml --max-steps 5 --dry-run
```

For managed jobs, upload or inline the script, use `a10g-large` as the default Qwen3-1.7B
LoRA development target, and set a timeout of at least two hours for non-smoke runs.

For free-GPU execution, use `docs/kaggle-workflow.md`. Kaggle is the memory-constrained
experimentation path; Hugging Face remains the dataset and adapter publication target.

## Local dataset commands

Preferred data-preparation command:

```bash
make data-ready
```

Audit or inspect generated artifacts:

```bash
make data-audit
make data-summary
```

Generate a small SFT sample outside git:

```bash
uv sync --extra training
uv run python -m medical_triage_agent ingest-sft mediqa data/processed/sft --limit 100
uv run python -m medical_triage_agent ingest-sft frenchmedmcqa data/processed/sft-frenchmedmcqa --limit 100
uv run python -m medical_triage_agent ingest-sft medquad data/processed/sft-medquad --limit 100
uv run python -m medical_triage_agent make-dataset-card data/processed/sft/manifest.json data/processed/sft/README.md
```

Generate a small DPO sample outside git:

```bash
uv run python -m medical_triage_agent ingest-dpo ultramedical_preference data/processed/dpo --limit 100
uv run python -m medical_triage_agent make-dataset-card data/processed/dpo/manifest.json data/processed/dpo/README.md
```

Run local safety checks:

```bash
uv run python -m medical_triage_agent evaluate-safety
```

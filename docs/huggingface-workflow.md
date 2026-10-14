# Hugging Face workflow

## Dataset publication

1. Verify source status in `docs/data-sources.md`.
2. Generate normalized JSONL outside git.
3. Run `uv run python -m medical_triage_agent audit-jsonl sft <path>` or `dpo <path>`.
4. Publish anonymized train, validation, test, and clinical evaluation splits to a private
   Hugging Face dataset repo.
5. Add a dataset card with source IDs, licenses, transforms, and known limits.

## Training and publication

Training runs on Kaggle through the SFT/DPO scripts and notebook. Hugging Face is
used for versioned dataset and adapter storage; it is not the project compute platform.
Keep `HF_TOKEN` in Kaggle secrets and publish only to an explicit repository.
See `docs/kaggle-workflow.md` for execution and `docs/evaluation.md` for pinned revisions.

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

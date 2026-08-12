# Kaggle free-GPU workflow

This workflow is the low-VRAM execution path for CHSA SFT, DPO, and the optional GRPO POC.
Hugging Face remains the publication target for datasets, adapters, and cards.

## Environment

- Use a Kaggle notebook with one free NVIDIA GPU, such as P100 or T4.
- Do not assume BF16 support; configs default to FP16.
- Keep batch size at `1`, gradient accumulation at `8`, and max sequence length at `512`.
- Store `HF_TOKEN` as a Kaggle secret when pushing private adapters to the Hub.
- Do not write model weights, checkpoints, logs, or generated datasets back into git.

## Dataset handoff

Build data locally or in Kaggle from the repository root:

```bash
make data-ready
make data-audit
make data-summary
```

The expected training files are under `data/processed/training` and are ignored by git. When
moving files into Kaggle, upload the generated folder as a Kaggle dataset or recreate it inside
the notebook before training.

## Smoke sequence

Validate the config and script startup without loading models:

```bash
make train-sft-smoke
make train-dpo-smoke
make train-grpo-smoke
make eval-models
```

Then run the smallest real SFT job:

```bash
uv sync --extra training
uv run scripts/train_sft.py --config configs/sft.yaml --max-steps 5 --max-train-samples 32
```

Run DPO only after SFT has produced an adapter:

```bash
uv run scripts/train_dpo.py --config configs/dpo.yaml --max-steps 5 --max-train-samples 32
```

GRPO is optional and should stay a proof of concept unless SFT and DPO are already working:

```bash
uv run scripts/train_grpo.py --config configs/grpo.yaml --max-steps 5 --max-train-samples 16
```

## Evaluation

Use deterministic decoding for model comparisons:

```bash
uv run scripts/evaluate.py --config configs/sft.yaml --model base --output outputs/evaluations/base.json
uv run scripts/evaluate.py --config configs/sft.yaml --model sft --adapter-path outputs/sft --output outputs/evaluations/sft.json
uv run scripts/evaluate.py --config configs/sft.yaml --model dpo --adapter-path outputs/dpo --output outputs/evaluations/dpo.json
```

Automatic metrics are technical indicators only. They do not prove clinical safety or
readiness for deployment.

## Publication

Push LoRA adapters to private Hugging Face repositories only when `HF_TOKEN` is available and
the target repo is explicit in the config. Do not merge adapters into the base model by
default.

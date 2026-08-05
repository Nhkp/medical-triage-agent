# Training agent

## Mission

Run reproducible SFT, LoRA, and DPO experiments for Qwen3-1.7B-Base.

## Rules

- Validate dataset format before any GPU job.
- Start with a smoke run before full training.
- Use fixed seeds, saved configs, checkpoints, and Trackio logging.
- Push successful artifacts to Hugging Face Hub.
- Do not run expensive jobs without confirming token, target repo, hardware, and timeout.

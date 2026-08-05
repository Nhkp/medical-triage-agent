from __future__ import annotations

import subprocess


def test_training_scripts_support_cpu_safe_dry_run() -> None:
    commands = (
        ["uv", "run", "scripts/train_sft.py", "--config", "configs/sft_kaggle.yaml", "--dry-run"],
        ["uv", "run", "scripts/train_dpo.py", "--config", "configs/dpo_kaggle.yaml", "--dry-run"],
        ["uv", "run", "scripts/train_grpo.py", "--config", "configs/grpo_kaggle.yaml", "--dry-run"],
        ["uv", "run", "scripts/evaluate.py", "--config", "configs/sft_kaggle.yaml", "--dry-run"],
    )

    for command in commands:
        result = subprocess.run(command, check=True, capture_output=True, text=True)
        assert (
            '"model": "Qwen/Qwen3-1.7B-Base"' in result.stdout
            or '"model_id": "Qwen/Qwen3-1.7B-Base"' in result.stdout
        )

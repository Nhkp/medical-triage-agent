from __future__ import annotations

from pathlib import Path

import pytest

from medical_triage_agent.configuration import ConfigurationError, load_training_config


def test_load_training_config_applies_overrides_without_requiring_files() -> None:
    config = load_training_config(
        "configs/sft_kaggle.yaml",
        method="sft",
        overrides={"training.max_steps": 5, "training.push_to_hub": False},
        require_files=False,
    )

    assert config.model_name() == "Qwen/Qwen3-1.7B-Base"
    assert config.section("training")["max_steps"] == 5
    assert config.max_seq_length() == 512
    assert config.precision() == "fp16"


def test_config_requires_existing_files_when_enabled(tmp_path: Path) -> None:
    config_path = tmp_path / "config.yaml"
    config_path.write_text(
        """
model:
  name: "Qwen/Qwen3-1.7B-Base"
data:
  train_file: "missing_train.jsonl"
  validation_file: "missing_validation.jsonl"
training:
  output_dir: "outputs/sft"
  max_seq_length: 512
  per_device_train_batch_size: 1
  gradient_accumulation_steps: 8
  learning_rate: 0.0002
  fp16: true
  bf16: false
  seed: 42
""",
        encoding="utf-8",
    )

    with pytest.raises(ConfigurationError, match="data.train_file does not exist"):
        load_training_config(config_path, method="sft")


def test_config_rejects_conflicting_precision() -> None:
    with pytest.raises(ConfigurationError, match="cannot both be true"):
        load_training_config(
            "configs/sft_kaggle.yaml",
            method="sft",
            overrides={"training.bf16": True},
            require_files=False,
        )

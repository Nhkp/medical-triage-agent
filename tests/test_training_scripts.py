from __future__ import annotations

import importlib.util
import json
import subprocess
from pathlib import Path
from typing import Any


def test_training_scripts_support_cpu_safe_dry_run() -> None:
    commands = (
        ["uv", "run", "scripts/train_sft.py", "--config", "configs/sft.yaml", "--dry-run"],
        ["uv", "run", "scripts/train_dpo.py", "--config", "configs/dpo.yaml", "--dry-run"],
        ["uv", "run", "scripts/train_grpo.py", "--config", "configs/grpo.yaml", "--dry-run"],
        ["uv", "run", "scripts/evaluate.py", "--config", "configs/sft.yaml", "--dry-run"],
    )

    for command in commands:
        result = subprocess.run(command, check=True, capture_output=True, text=True)
        assert (
            '"model": "Qwen/Qwen3-1.7B-Base"' in result.stdout
            or '"model_id": "Qwen/Qwen3-1.7B-Base"' in result.stdout
        )


def test_sft_script_passes_text_only_rows_to_trl(tmp_path: Path) -> None:
    module = _load_script("train_sft", Path("scripts/train_sft.py"))
    dataset_path = tmp_path / "sft.jsonl"
    dataset_path.write_text(json.dumps(_sft_row()) + "\n", encoding="utf-8")

    rows = module._load_sft_dataset(
        dataset_path,
        tokenizer=None,
        system_message=None,
        max_samples=None,
    )

    assert list(rows[0]) == ["text"]
    assert "Assistant:" in rows[0]["text"]


def test_dpo_script_passes_conversational_rows_to_trl(tmp_path: Path) -> None:
    module = _load_script("train_dpo", Path("scripts/train_dpo.py"))
    dataset_path = tmp_path / "dpo.jsonl"
    dataset_path.write_text(json.dumps(_dpo_row()) + "\n", encoding="utf-8")

    rows = module._load_dpo_dataset(
        dataset_path,
        tokenizer=None,
        system_message=None,
        max_samples=None,
    )

    assert rows[0]["prompt"][1] == {"role": "user", "content": "Fievre elevee"}
    assert rows[0]["chosen"] == [{"role": "assistant", "content": "Consulter rapidement."}]
    assert rows[0]["rejected"] == [{"role": "assistant", "content": "Attendre plusieurs jours."}]


def test_sft_config_filters_kwargs_for_installed_trl_versions() -> None:
    module = _load_script("train_sft", Path("scripts/train_sft.py"))

    class Config:
        def __init__(self, output_dir: str, evaluation_strategy: str | None = None) -> None:
            self.output_dir = output_dir
            self.evaluation_strategy = evaluation_strategy

    config = module._make_sft_config(
        Config,
        output_dir="outputs/sft",
        warmup_ratio=0.03,
        eval_strategy="steps",
    )

    assert config.output_dir == "outputs/sft"
    assert config.evaluation_strategy == "steps"


def test_training_scripts_accept_hub_model_id_override() -> None:
    commands = (
        ["uv", "run", "scripts/train_sft.py", "--config", "configs/sft.yaml"],
        ["uv", "run", "scripts/train_dpo.py", "--config", "configs/dpo.yaml"],
        ["uv", "run", "scripts/train_grpo.py", "--config", "configs/grpo.yaml"],
    )

    for command in commands:
        result = subprocess.run(
            [
                *command,
                "--dry-run",
                "--push-to-hub",
                "--hub-model-id",
                "Lokhidor/medical-triage-test-adapter",
            ],
            check=True,
            capture_output=True,
            text=True,
        )
        assert '"push_to_hub": true' in result.stdout
        assert '"hub_model_id": "Lokhidor/medical-triage-test-adapter"' in result.stdout


def _load_script(name: str, path: Path) -> Any:
    spec = importlib.util.spec_from_file_location(name, path)
    assert spec is not None
    assert spec.loader is not None
    module = importlib.util.module_from_spec(spec)
    spec.loader.exec_module(module)
    return module


def _sft_row() -> dict[str, object]:
    return {
        "id": "sft_fixture",
        "language": "fr",
        "instruction": "Evaluer le niveau de triage.",
        "input": "douleur thoracique",
        "output": "Escalader vers une evaluation urgente.",
        "source_ids": ["fixture"],
        "metadata": {
            "symptoms": ["douleur"],
            "antecedents": [],
            "vitals": {},
            "triage_level": "urgence_maximale",
            "confidence": 0.7,
            "source": "fixture",
            "license": "test",
            "transforms": [],
        },
    }


def _dpo_row() -> dict[str, object]:
    return {
        "id": "dpo_fixture",
        "language": "fr",
        "prompt": "Fievre elevee",
        "chosen": "Consulter rapidement.",
        "rejected": "Attendre plusieurs jours.",
        "source_ids": ["fixture"],
        "metadata": {
            "symptoms": ["fievre"],
            "antecedents": [],
            "vitals": {},
            "triage_level": "moderee",
            "confidence": 0.7,
            "source": "fixture",
            "license": "test",
            "transforms": [],
        },
    }

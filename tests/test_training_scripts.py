from __future__ import annotations

import importlib.util
import json
import subprocess
from pathlib import Path
from typing import Any


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

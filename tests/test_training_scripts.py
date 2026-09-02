from __future__ import annotations

import importlib.util
import json
import subprocess
from pathlib import Path
from typing import Any, cast


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


def test_train_experiment_supports_cpu_safe_dry_run() -> None:
    result = subprocess.run(
        [
            "uv",
            "run",
            "scripts/train_experiment.py",
            "--dataset-label",
            "smoke",
            "--dataset-dir",
            "data/processed/training",
            "--dataset-repo",
            "Lokhidor/medical-triage-dataset",
            "--max-steps",
            "1",
            "--dry-run",
        ],
        check=True,
        capture_output=True,
        text=True,
    )

    payload = json.loads(result.stdout)
    assert payload["dataset_label"] == "smoke"
    assert payload["dataset_repo"] == "Lokhidor/medical-triage-dataset"
    assert payload["sft"]["output_dir"] == "outputs/experiments/smoke/sft"
    assert payload["dpo"]["adapter_path"] == "outputs/experiments/smoke/sft"
    assert payload["sft"]["report_to"] == "mlflow"
    assert payload["dpo"]["report_to"] == "mlflow"
    assert payload["mlflow_ngrok"] is False
    assert payload["mlflow_ui"] is None


def test_train_experiment_dry_run_reports_mlflow_ngrok_command() -> None:
    result = subprocess.run(
        [
            "uv",
            "run",
            "scripts/train_experiment.py",
            "--dataset-label",
            "smoke",
            "--dataset-dir",
            "data/processed/training",
            "--mlflow-ngrok",
            "--mlflow-ui-port",
            "5050",
            "--dry-run",
        ],
        check=True,
        capture_output=True,
        text=True,
    )

    payload = json.loads(result.stdout)
    assert payload["mlflow_ngrok"] is True
    assert payload["mlflow_ui"][-4:] == ["--host", "127.0.0.1", "--port", "5050"]
    assert "mlflow" in payload["mlflow_ui"]


def test_train_experiment_overrides_dataset_paths_and_outputs(tmp_path: Path) -> None:
    module = _load_script("train_experiment", Path("scripts/train_experiment.py"))
    args = argparse_namespace(
        dataset_dir=tmp_path / "dataset-8k",
        sft_output_dir=tmp_path / "outputs" / "sft-8k",
        dpo_output_dir=tmp_path / "outputs" / "dpo-8k",
        max_steps=2,
        push_to_hub=True,
        sft_hub_model_id="Lokhidor/sft-8k",
        dpo_hub_model_id="Lokhidor/dpo-8k",
    )

    sft_config = module._load_sft_config(args, require_files=False)
    dpo_config = module._load_dpo_config(args, require_files=False)

    assert sft_config.section("data")["train_file"].endswith("dataset-8k/sft_train.jsonl")
    assert dpo_config.section("data")["train_file"].endswith("dataset-8k/dpo_train.jsonl")
    assert str(sft_config.output_dir()).endswith("outputs/sft-8k")
    assert str(dpo_config.output_dir()).endswith("outputs/dpo-8k")
    assert dpo_config.section("model")["adapter_path"] == str(args.sft_output_dir)
    assert sft_config.section("training")["hub_model_id"] == "Lokhidor/sft-8k"
    assert dpo_config.section("training")["hub_model_id"] == "Lokhidor/dpo-8k"


def test_train_experiment_defaults_keep_dataset_outputs_distinct() -> None:
    first = _dry_run_experiment_payload("dataset-5k")
    second = _dry_run_experiment_payload("dataset-8k")

    assert first["sft"]["output_dir"] == "outputs/experiments/dataset-5k/sft"
    assert first["dpo"]["output_dir"] == "outputs/experiments/dataset-5k/dpo"
    assert second["sft"]["output_dir"] == "outputs/experiments/dataset-8k/sft"
    assert second["dpo"]["output_dir"] == "outputs/experiments/dataset-8k/dpo"


def test_tracking_params_include_manifest_metadata(tmp_path: Path) -> None:
    from medical_triage_agent.configuration import load_training_config
    from medical_triage_agent.tracking import prepare_mlflow_params

    config = load_training_config(
        "configs/sft.yaml",
        method="sft",
        overrides={"training.output_dir": str(tmp_path / "sft"), "training.report_to": "mlflow"},
        require_files=False,
    )
    manifest = {
        "generated_at": "2026-08-20T12:00:00Z",
        "actual_counts": {"sft": 8000, "dpo": 2000},
        "language_counts": {"fr": 4000, "en": 6000},
        "content_hashes": {"sft": {"train": "abc"}},
    }

    params = prepare_mlflow_params(
        dataset_label="dataset-8k",
        dataset_dir=tmp_path / "dataset",
        method="sft",
        dataset_repo="Lokhidor/medical-triage-dataset-8k",
        config=config,
        output_dir=tmp_path / "sft",
        hub_model_id="Lokhidor/sft-8k",
        manifest=manifest,
    )

    assert params["dataset_label"] == "dataset-8k"
    assert params["dataset_repo"] == "Lokhidor/medical-triage-dataset-8k"
    assert params["manifest.generated_at"] == "2026-08-20T12:00:00Z"
    assert params["manifest.actual_counts"] == '{"dpo": 2000, "sft": 8000}'
    assert params["manifest.content_hashes"] == '{"sft": {"train": "abc"}}'
    assert params["hf_adapter_repo"] == "Lokhidor/sft-8k"


def _dry_run_experiment_payload(dataset_label: str) -> dict[str, Any]:
    result = subprocess.run(
        [
            "uv",
            "run",
            "scripts/train_experiment.py",
            "--dataset-label",
            dataset_label,
            "--dataset-dir",
            "data/processed/training",
            "--dry-run",
        ],
        check=True,
        capture_output=True,
        text=True,
    )
    return cast(dict[str, Any], json.loads(result.stdout))


def argparse_namespace(**overrides: Any) -> Any:
    values = {
        "dataset_dir": Path("data/processed/training"),
        "dataset_repo": None,
        "sft_config": "configs/sft.yaml",
        "dpo_config": "configs/dpo.yaml",
        "max_steps": None,
        "push_to_hub": False,
        "sft_hub_model_id": None,
        "dpo_hub_model_id": None,
        "mlflow_tracking_uri": "mlruns",
        "mlflow_ui_host": "127.0.0.1",
        "mlflow_ui_port": 5000,
        "mlflow_ngrok": False,
    }
    values.update(overrides)

    class Namespace:
        pass

    namespace = Namespace()
    for key, value in values.items():
        setattr(namespace, key, value)
    return namespace


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

from __future__ import annotations

import json
from pathlib import Path


def test_kaggle_serving_notebook_wraps_script_without_tokens() -> None:
    path = Path("notebooks/kaggle_serving.ipynb")
    assert path.exists()
    notebook = json.loads(path.read_text(encoding="utf-8"))
    source = "\n".join(
        "".join(cell.get("source", [])) for cell in notebook["cells"] if cell["cell_type"] == "code"
    )

    for expected in (
        "/kaggle/working/medical-triage-agent",
        "kaggle_secrets",
        "scripts/serve_notebook.py",
        "--base-model",
        "--adapter",
        "--dry-run",
        "--ngrok",
        "/health",
        "/triage",
        "/audit/",
        "scripts/evaluate_robustness.py",
        "scripts/evaluate_latency.py",
        "outputs-evaluations.zip",
    ):
        assert expected in source

    assert "hf_" not in source
    assert "HF_TOKEN =" not in source


def test_kaggle_model_comparison_notebook_references_models_and_outputs() -> None:
    path = Path("notebooks/kaggle_model_comparison.ipynb")
    assert path.exists()
    notebook = json.loads(path.read_text(encoding="utf-8"))
    source = "\n".join(
        "".join(cell.get("source", [])) for cell in notebook["cells"] if cell["cell_type"] == "code"
    )

    for expected in (
        "scripts/evaluate_model_comparison.py",
        "Qwen/Qwen3-1.7B-Base",
        "comparison.DEFAULT_MODELS",
        "snapshot_download",
        "adapter_revision",
        "manifest.json",
        "model_comparison_",
        "summarize_campaign",
        "zipfile.ZipFile",
    ):
        assert expected in source

    assert "not clinical validation" in path.read_text(encoding="utf-8")
    assert "hf_" not in source
    assert "HF_TOKEN =" not in source


def test_kaggle_training_notebook_supports_tracked_5k_and_8k_experiments() -> None:
    path = Path("notebooks/kaggle_training.ipynb")
    assert path.exists()
    notebook = json.loads(path.read_text(encoding="utf-8"))
    source = "\n".join(
        "".join(cell.get("source", [])) for cell in notebook["cells"] if cell["cell_type"] == "code"
    )

    for expected in (
        'DATASET_SIZE = "8k"',
        "Lokhidor/medical-triage-dataset",
        "Lokhidor/medical-triage-dataset-8k",
        "NGROK_AUTHTOKEN",
        "mlflow pyngrok",
        "snapshot_download",
        "required_files",
        'git", "pull", "--ff-only", "origin", "main',
        "required_repo_files",
        "scripts/train_experiment.py",
        "--mlflow-ngrok",
        "--mlflow-ui-port",
        "--dry-run",
        "mlflow.db mlruns outputs/experiments",
    ):
        assert expected in source

    assert "HF_TOKEN =" not in source


def test_comparison_notebook_uses_existing_helper_exports() -> None:
    import ast
    import importlib

    module = importlib.import_module("scripts.evaluate_model_comparison")
    notebook = json.loads(Path("notebooks/kaggle_model_comparison.ipynb").read_text())
    for cell in notebook["cells"]:
        if cell["cell_type"] != "code":
            continue
        source = "".join(cell["source"])
        if any(line.lstrip().startswith(("!", "%")) for line in source.splitlines()):
            continue
        for node in ast.walk(ast.parse(source)):
            if (
                isinstance(node, ast.Attribute)
                and isinstance(node.value, ast.Name)
                and node.value.id == "comparison"
            ):
                assert hasattr(module, node.attr), node.attr

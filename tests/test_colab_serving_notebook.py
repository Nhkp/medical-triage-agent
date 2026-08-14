from __future__ import annotations

import json
from pathlib import Path


def test_colab_serving_notebook_wraps_script_without_tokens() -> None:
    path = Path("notebooks/colab_serving.ipynb")
    assert path.exists()
    notebook = json.loads(path.read_text(encoding="utf-8"))
    source = "\n".join(
        "".join(cell.get("source", [])) for cell in notebook["cells"] if cell["cell_type"] == "code"
    )

    for expected in (
        "scripts/serve_colab.py",
        "--base-model",
        "--adapter",
        "--dry-run",
        "--ngrok",
        "/health",
        "/triage",
        "/audit/",
        "scripts/evaluate_robustness.py",
        "scripts/evaluate_latency.py",
    ):
        assert expected in source

    assert "hf_" not in source
    assert "HF_TOKEN =" not in source


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
        "scripts/serve_colab.py",
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
        "Lokhidor/medical-triage-qwen3-sft-lora",
        "Lokhidor/medical-triage-qwen3-dpo-lora",
        "model_comparison_base.json",
        "model_comparison_sft.json",
        "model_comparison_dpo.json",
        "model_comparison_summary.csv",
        "outputs-evaluations-model-comparison.zip",
    ):
        assert expected in source

    text = path.read_text(encoding="utf-8")
    assert "structured JSON constraints" in text
    assert "hf_" not in source
    assert "HF_TOKEN =" not in source

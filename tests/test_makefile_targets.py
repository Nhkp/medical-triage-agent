from __future__ import annotations

from pathlib import Path


def test_makefile_exposes_data_preparation_targets() -> None:
    makefile = Path("Makefile").read_text(encoding="utf-8")

    for target in (
        "sync-training:",
        "sync-ui:",
        "data-build:",
        "data-audit:",
        "data-card:",
        "data-summary:",
        "data-ready:",
        "data-clean:",
        "data-pull-5k:",
        "data-pull-8k:",
        "data-audit-5k:",
        "data-audit-8k:",
        "data-summary-5k:",
        "data-summary-8k:",
        "train-sft-smoke:",
        "train-dpo-smoke:",
        "train-grpo-smoke:",
        "train-5k:",
        "train-8k:",
        "pipeline-5k:",
        "pipeline-8k:",
        "pipeline-all:",
        "mlflow-ui:",
        "eval-models:",
        "eval-model-comparison-dry-run:",
        "eval-model-comparison:",
        "serve-local:",
        "serve-api:",
        "serve-ui:",
        "serve-colab-dry-run:",
        "serve-kaggle-dry-run:",
        "eval-latency:",
        "eval-robustness:",
        "step3-ready:",
    ):
        assert target in makefile

    assert "build-training-data" in makefile
    assert "audit-training-data" in makefile
    assert "summarize-training-data" in makefile
    assert "scripts/train_sft.py" in makefile
    assert "scripts/train_dpo.py" in makefile
    assert "scripts/train_grpo.py" in makefile
    assert "scripts/train_experiment.py" in makefile
    assert "HF_DATASET_REPO_5K ?= Lokhidor/medical-triage-dataset" in makefile
    assert "HF_DATASET_REPO_8K ?= Lokhidor/medical-triage-dataset-8k" in makefile
    assert "--dataset-repo $(HF_DATASET_REPO_5K)" in makefile
    assert "--dataset-repo $(HF_DATASET_REPO_8K)" in makefile
    assert "MLFLOW_NGROK_ARG" in makefile
    assert "--mlflow-ngrok --mlflow-ui-port $(MLFLOW_UI_PORT)" in makefile
    assert "MLFLOW_TRACKING_URI ?= sqlite:///mlflow.db" in makefile
    assert "mlflow ui --backend-store-uri $(MLFLOW_TRACKING_URI)" in makefile
    assert "scripts/evaluate.py" in makefile
    assert "scripts/evaluate_model_comparison.py" in makefile
    assert "scripts/evaluate_latency.py" in makefile
    assert "scripts/evaluate_robustness.py" in makefile
    assert "scripts/serve_colab.py" in makefile

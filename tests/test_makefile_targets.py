from __future__ import annotations

from pathlib import Path


def test_makefile_exposes_data_preparation_targets() -> None:
    makefile = Path("Makefile").read_text(encoding="utf-8")

    for target in (
        "sync-training:",
        "data-build:",
        "data-audit:",
        "data-card:",
        "data-summary:",
        "data-ready:",
        "data-clean:",
        "train-sft-smoke:",
        "train-dpo-smoke:",
        "train-grpo-smoke:",
        "eval-models:",
        "serve-local:",
        "serve-api:",
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
    assert "scripts/evaluate.py" in makefile
    assert "scripts/evaluate_latency.py" in makefile
    assert "scripts/evaluate_robustness.py" in makefile
    assert "scripts/serve_colab.py" in makefile

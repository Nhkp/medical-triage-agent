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

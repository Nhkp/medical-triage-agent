from __future__ import annotations

import importlib.util
from pathlib import Path

import pytest

from medical_triage_agent.api import create_app


def test_streamlit_interface_was_removed() -> None:
    assert not Path("streamlit_app").exists()
    assert not Path("docs/streamlit-ui-plan.md").exists()
    assert not Path("docs/agents/ui-agent.md").exists()

    pyproject = Path("pyproject.toml").read_text(encoding="utf-8")
    makefile = Path("Makefile").read_text(encoding="utf-8")

    assert "streamlit" not in pyproject.casefold()
    assert "sync-ui:" not in makefile
    assert "ui-check:" not in makefile
    assert "streamlit run" not in makefile


def test_readme_documents_fastapi_demo_console() -> None:
    readme = Path("README.md").read_text(encoding="utf-8")

    assert "http://127.0.0.1:8080/demo" in readme
    assert "ngrok-free.dev/demo" in readme


@pytest.mark.skipif(importlib.util.find_spec("fastapi") is None, reason="FastAPI extra missing")
def test_demo_routes_return_html() -> None:
    from fastapi.testclient import TestClient

    client = TestClient(create_app())

    for path in ("/", "/demo"):
        response = client.get(path)
        assert response.status_code == 200
        assert "text/html" in response.headers["content-type"]
        assert "/triage" in response.text

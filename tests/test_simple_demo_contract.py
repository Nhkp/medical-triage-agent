from __future__ import annotations

import tomllib
from pathlib import Path

from medical_triage_agent.api import create_app


def test_old_streamlit_interface_was_removed() -> None:
    assert not Path("streamlit_app").exists()
    assert not Path("docs/streamlit-ui-plan.md").exists()
    assert not Path("docs/agents/ui-agent.md").exists()

    makefile = Path("Makefile").read_text(encoding="utf-8")
    assert "ui-check:" not in makefile


def test_streamlit_tester_is_optional_tooling() -> None:
    pyproject = tomllib.loads(Path("pyproject.toml").read_text(encoding="utf-8"))
    makefile = Path("Makefile").read_text(encoding="utf-8")

    assert Path("src/medical_triage_agent/streamlit_interface.py").exists()
    assert any(
        dependency.startswith("streamlit")
        for dependency in pyproject["project"]["optional-dependencies"]["ui"]
    )
    assert "sync-ui:" in makefile
    assert "serve-ui:" in makefile
    assert "streamlit run src/medical_triage_agent/streamlit_interface.py" in makefile


def test_readme_documents_streamlit_tester_without_demo_endpoint() -> None:
    readme = Path("README.md").read_text(encoding="utf-8")

    assert "make serve-api" in readme
    assert "make serve-ui" in readme
    assert "/demo" not in readme


def test_demo_routes_are_not_part_of_fastapi_contract() -> None:
    paths = {route.path for route in create_app().routes}

    assert "/" not in paths
    assert "/demo" not in paths
    assert {"/health", "/triage", "/audit/{audit_id}"}.issubset(paths)

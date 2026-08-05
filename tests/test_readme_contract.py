from __future__ import annotations

from pathlib import Path


def test_readme_keeps_scientific_project_sections() -> None:
    readme = Path("README.md").read_text(encoding="utf-8")

    for section in (
        "## Abstract",
        "## Introduction",
        "## Objective",
        "## Methodology",
        "## Architecture",
        "## Current Results",
        "## Reproducibility",
        "## Limitations",
        "## Repository Map",
        "## References",
    ):
        assert section in readme


def test_readme_documents_real_make_targets() -> None:
    readme = Path("README.md").read_text(encoding="utf-8")
    makefile = Path("Makefile").read_text(encoding="utf-8")

    for target in ("check", "data-ready", "data-audit", "data-summary", "data-clean"):
        assert f"make {target}" in readme
        assert f"{target}:" in makefile


def test_readme_agent_is_registered() -> None:
    assert Path("docs/agents/readme-agent.md").exists()
    assert "docs/agents/readme-agent.md" in Path("AGENTS.md").read_text(encoding="utf-8")

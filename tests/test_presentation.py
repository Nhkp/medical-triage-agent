from __future__ import annotations

import subprocess
from pathlib import Path


def test_presentation_agent_and_sources_exist() -> None:
    assert Path("docs/agents/presentation-agent.md").exists()
    assert Path("presentations/chsa-current-state/index.html").exists()
    assert Path("presentations/chsa-current-state/styles.css").exists()
    assert "docs/agents/presentation-agent.md" in Path("AGENTS.md").read_text(encoding="utf-8")


def test_presentation_has_15_minute_slide_count_and_required_topics() -> None:
    html = Path("presentations/chsa-current-state/index.html").read_text(encoding="utf-8")

    assert html.count('class="slide') == 14
    for topic in (
        "The CHSA Pressure Point",
        "Assist, Never Replace",
        "System in One Picture",
        "Step 0: Method Before Model",
        "Step 1: Dataset Built and Audited",
        "Privacy and Audit Evidence",
        "Clinical Review Debt",
        "Step 2: Low-VRAM Training Path",
        "Step 3: Serving and Demo API",
        "Roadmap and Decision Gate",
    ):
        assert topic in html


def test_presentation_make_targets_are_documented() -> None:
    makefile = Path("Makefile").read_text(encoding="utf-8")

    for target in (
        "sync-presentation:",
        "presentation-html:",
        "presentation-pptx:",
        "presentation-ready:",
    ):
        assert target in makefile


def test_presentation_export_dry_run() -> None:
    result = subprocess.run(
        [
            "uv",
            "run",
            "python",
            "scripts/export_presentation.py",
            "--input",
            "presentations/chsa-current-state/index.html",
            "--dry-run",
        ],
        check=True,
        capture_output=True,
        text=True,
    )

    assert "slides=14" in result.stdout

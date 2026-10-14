from __future__ import annotations

import importlib
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
        "Mission and POC Outcome",
        "CHSA Need and Delivered Scope",
        "Architecture and Backend Authority",
        "Main Bilingual Dataset",
        "Provenance Privacy and Splits",
        "SFT and DPO Training",
        "Reproducible Comparison Protocol",
        "Format and Repetition Results",
        "Latency and Model Differences",
        "Relevance and Pedagogical Review",
        "Observed Failures and Protection Limits",
        "Modal Deployment and Demo Walkthrough",
        "Evidence Limits and Demo Decision",
        "Roadmap and Clinical Pilot Gates",
    ):
        assert topic in html


def test_presentation_make_targets_are_documented() -> None:
    makefile = Path("Makefile").read_text(encoding="utf-8")

    for target in (
        "sync-presentation:",
        "presentation-browser:",
        "presentation-html:",
        "presentation-pptx:",
        "presentation-ready:",
    ):
        assert target in makefile
    assert "--mode rendered" in makefile


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


def test_speaker_notes_are_separate_from_visible_slide_content(tmp_path: Path) -> None:
    path = tmp_path / "deck.html"
    path.write_text(
        '<section class="slide" data-slide-title="Example"><p>Context</p>'
        '<h2>Visible title</h2><aside class="notes">Review is provisional.</aside></section>',
        encoding="utf-8",
    )
    slide = importlib.import_module("scripts.export_presentation").parse_deck(path)[0]
    assert slide.notes == "Review is provisional."
    assert slide.title == "Visible title"
    assert "provisional" not in slide.subtitle

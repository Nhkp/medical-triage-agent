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
    ):
        assert target in makefile

    assert "build-training-data" in makefile
    assert "audit-training-data" in makefile
    assert "summarize-training-data" in makefile

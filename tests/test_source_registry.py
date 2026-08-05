from __future__ import annotations

from pathlib import Path

import pytest

from medical_triage_agent.source_registry import load_source_registry, validate_source_ids


def test_load_source_registry_from_markdown_table(tmp_path: Path) -> None:
    registry_path = tmp_path / "sources.md"
    registry_path.write_text(
        """# Sources
| id | name | url | license | languages | intended_use | status | notes |
| --- | --- | --- | --- | --- | --- | --- | --- |
| src1 | Source 1 | https://example.test | cc-by | fr,en | sft,dpo | verified | ok |
""",
        encoding="utf-8",
    )

    registry = load_source_registry(registry_path)

    assert registry["src1"].ingestible
    validate_source_ids(["src1"], registry)


def test_validate_source_ids_rejects_blocked_source(tmp_path: Path) -> None:
    registry_path = tmp_path / "sources.md"
    registry_path.write_text(
        """# Sources
| id | name | url | license | languages | intended_use | status | notes |
| --- | --- | --- | --- | --- | --- | --- | --- |
| src1 | Source 1 | https://example.test | to_verify | fr | sft | blocked | no |
""",
        encoding="utf-8",
    )
    registry = load_source_registry(registry_path)

    with pytest.raises(ValueError, match="not ingestible"):
        validate_source_ids(["src1"], registry)

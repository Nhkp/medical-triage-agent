from __future__ import annotations

from pathlib import Path

import pytest

import medical_triage_agent.__main__ as cli


def test_cli_training_data_commands(tmp_path: Path, monkeypatch: pytest.MonkeyPatch) -> None:
    manifest = {
        "actual_counts": {"sft": 1, "dpo": 1},
        "language_counts": {"en": 1, "fr": 1},
        "source_counts": {"mediqa": 1},
        "split_counts": {"sft": {"train": 1}, "dpo": {"train": 1}},
        "rejected_counts": {"sft": 0, "dpo": 0},
        "files": [],
    }

    monkeypatch.setattr(cli, "build_training_data", lambda config: manifest)
    monkeypatch.setattr(cli, "audit_training_data", lambda output_dir: {"passed": True})
    monkeypatch.setattr(cli, "summarize_training_data", lambda output_dir: manifest)

    monkeypatch.setattr(
        "sys.argv",
        ["medical-triage-agent", "build-training-data", str(tmp_path), "--sft-target", "1"],
    )
    assert cli.main() == 0

    monkeypatch.setattr("sys.argv", ["medical-triage-agent", "audit-training-data", str(tmp_path)])
    assert cli.main() == 0

    monkeypatch.setattr(
        "sys.argv", ["medical-triage-agent", "summarize-training-data", str(tmp_path)]
    )
    assert cli.main() == 0

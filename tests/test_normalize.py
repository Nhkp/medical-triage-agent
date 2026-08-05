from __future__ import annotations

from medical_triage_agent.contracts import Metadata
from medical_triage_agent.normalize import assign_split, normalize_sft_record, to_jsonl_line


def metadata() -> Metadata:
    return Metadata(
        symptoms=["fatigue"],
        antecedents=[],
        vitals={},
        triage_level="moderee",
        confidence=0.5,
        source="Fixture",
        license="test-only",
        transforms=["fixture"],
    )


def test_normalize_sft_redacts_pii_and_assigns_deterministic_id() -> None:
    first = normalize_sft_record(
        language="fr",
        instruction="Evaluer",
        input_text="Contact patient@example.test fatigue",
        output_text="Revue clinique",
        source_id="verified_source",
        metadata=metadata(),
    )
    second = normalize_sft_record(
        language="fr",
        instruction="Evaluer",
        input_text="Contact patient@example.test fatigue",
        output_text="Revue clinique",
        source_id="verified_source",
        metadata=metadata(),
    )

    assert first.id == second.id
    assert "patient@example.test" not in first.input
    assert assign_split(first.id) in {"train", "validation", "test", "clinical_eval"}
    assert to_jsonl_line(first).endswith("\n")

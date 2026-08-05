from __future__ import annotations

import json
from pathlib import Path

import pytest

import medical_triage_agent.dataset_pipeline as pipeline
from medical_triage_agent.contracts import DPOExample, Metadata, SFTExample
from medical_triage_agent.dataset_pipeline import (
    TrainingDataConfig,
    audit_training_data,
    build_training_data,
    load_hf_sft_records,
    make_dataset_card,
    map_frenchmedmcqa,
    map_mediqa_oeq,
    map_medquad,
    map_ultramedical_preference,
    summarize_training_data,
    write_splits,
)
from medical_triage_agent.source_registry import SourceRecord

MEDIQA = SourceRecord(
    id="mediqa",
    name="MediQAl",
    url="https://huggingface.co/datasets/ANR-MALADES/MediQAl",
    license="cc-by-4.0",
    languages=("fr",),
    intended_use=("sft", "evaluation"),
    status="verified",
    notes="fixture",
)

ULTRAMEDICAL = SourceRecord(
    id="ultramedical_preference",
    name="UltraMedical Preference",
    url="https://huggingface.co/datasets/TsinghuaC3I/UltraMedical-Preference",
    license="mit",
    languages=("en",),
    intended_use=("dpo",),
    status="verified",
    notes="fixture",
)

FRENCHMEDMCQA = SourceRecord(
    id="frenchmedmcqa",
    name="FrenchMedMCQA",
    url="https://huggingface.co/datasets/nthngdy/frenchmedmcqa",
    license="apache-2.0",
    languages=("fr",),
    intended_use=("sft", "evaluation"),
    status="verified",
    notes="fixture",
)

MEDQUAD = SourceRecord(
    id="medquad",
    name="MedQuad",
    url="https://huggingface.co/datasets/keivalya/MedQuad-MedicalQnADataset",
    license="apache-2.0",
    languages=("en",),
    intended_use=("sft",),
    status="verified",
    notes="fixture",
)


def test_map_mediqa_oeq_creates_valid_sft_record() -> None:
    record = map_mediqa_oeq(
        {
            "clinical_case": "Patient avec dyspnee.",
            "question": "Quel niveau de gravite?",
            "answer": "Evaluation clinique urgente.",
        },
        MEDIQA,
    )

    assert record.language == "fr"
    assert record.source_ids == ["mediqa"]
    assert record.metadata.license == "cc-by-4.0"
    assert "Patient avec dyspnee" in record.input


def test_map_frenchmedmcqa_creates_valid_sft_record() -> None:
    record = map_frenchmedmcqa(
        {
            "question": "Quelle proposition est exacte?",
            "answer_a": "Option A",
            "answer_b": "Option B",
            "answer_c": "Option C",
            "answer_d": "Option D",
            "answer_e": "Option E",
            "correct_answers": 2,
        },
        FRENCHMEDMCQA,
    )

    assert record.language == "fr"
    assert record.source_ids == ["frenchmedmcqa"]
    assert record.output == "Reponse C: Option C"


def test_map_medquad_creates_valid_sft_record() -> None:
    record = map_medquad(
        {
            "qtype": "symptoms",
            "Question": "What are the symptoms of condition X?",
            "Answer": "Symptoms include fever.",
        },
        MEDQUAD,
    )

    assert record.language == "en"
    assert record.source_ids == ["medquad"]
    assert record.output == "Symptoms include fever."


def test_map_ultramedical_preference_creates_valid_dpo_record() -> None:
    record = map_ultramedical_preference(
        {
            "prompt": "Explain chest pain triage.",
            "chosen": [
                {"role": "user", "content": "Explain chest pain triage."},
                {"role": "assistant", "content": "Escalate immediately."},
            ],
            "rejected": [
                {"role": "user", "content": "Explain chest pain triage."},
                {"role": "assistant", "content": "Ignore symptoms."},
            ],
        },
        ULTRAMEDICAL,
    )

    assert record.language == "en"
    assert record.chosen == "Escalate immediately."
    assert record.rejected == "Ignore symptoms."


def test_write_splits_creates_manifest_and_dataset_card(tmp_path: Path) -> None:
    record = map_mediqa_oeq(
        {
            "clinical_case": "Patient fatigue.",
            "question": "Que faire?",
            "answer": "Revue clinique.",
        },
        MEDIQA,
    )

    generated = write_splits([record], tmp_path, "sft")
    card_path = make_dataset_card(generated.manifest_path, tmp_path / "README.md")
    manifest = json.loads(generated.manifest_path.read_text(encoding="utf-8"))

    assert manifest["source_ids"] == ["mediqa"]
    assert sum(manifest["split_counts"].values()) == 1
    assert card_path.exists()


def test_unknown_sft_source_cannot_be_ingested_before_optional_imports() -> None:
    with pytest.raises(ValueError, match="unknown source id"):
        load_hf_sft_records("missing", limit=1)


def test_build_training_data_creates_artifacts_with_source_fill(
    tmp_path: Path, monkeypatch: pytest.MonkeyPatch
) -> None:
    def fake_sft(source_id: str, limit: int | None) -> list[SFTExample]:
        count = {"mediqa": 1, "frenchmedmcqa": 1, "medquad": limit or 0}[source_id]
        return [_sft_example(source_id, index) for index in range(count)]

    def fake_dpo(source_id: str, limit: int | None) -> list[DPOExample]:
        assert source_id == "ultramedical_preference"
        return [_dpo_example(index) for index in range(limit or 0)]

    monkeypatch.setattr(pipeline, "load_hf_sft_records", fake_sft)
    monkeypatch.setattr(pipeline, "load_hf_dpo_records", fake_dpo)

    manifest = build_training_data(
        TrainingDataConfig(
            output_dir=tmp_path,
            sft_target=4,
            dpo_target=2,
            sft_mediqa=2,
            sft_frenchmedmcqa=2,
        )
    )

    assert manifest["actual_counts"] == {"sft": 4, "dpo": 2}
    assert manifest["source_counts"] == {
        "frenchmedmcqa": 1,
        "medquad": 2,
        "mediqa": 1,
        "ultramedical_preference": 2,
    }
    assert (tmp_path / "manifest.json").exists()
    assert (tmp_path / "audit_report.json").exists()
    assert (tmp_path / "clinical_review_queue.jsonl").exists()
    assert (tmp_path / "README.md").exists()


def test_audit_and_summary_training_data(tmp_path: Path, monkeypatch: pytest.MonkeyPatch) -> None:
    monkeypatch.setattr(
        pipeline,
        "load_hf_sft_records",
        lambda source_id, limit: [_sft_example(source_id, index) for index in range(limit or 0)],
    )
    monkeypatch.setattr(
        pipeline,
        "load_hf_dpo_records",
        lambda source_id, limit: [_dpo_example(index) for index in range(limit or 0)],
    )
    build_training_data(TrainingDataConfig(output_dir=tmp_path, sft_target=3, dpo_target=1))

    audit = audit_training_data(tmp_path)
    summary = summarize_training_data(tmp_path)

    assert audit["passed"] is True
    assert summary["actual_counts"] == {"sft": 3, "dpo": 1}
    assert len(summary["files"]) == 8


def _metadata(source_id: str) -> Metadata:
    return Metadata(
        symptoms=[],
        antecedents=[],
        vitals={},
        triage_level="moderee",
        confidence=0.4,
        source=source_id,
        license="test-only",
        transforms=[f"map_{source_id}", "redact_pii"],
    )


def _sft_example(source_id: str, index: int) -> SFTExample:
    return SFTExample.from_mapping(
        {
            "id": f"sft_{source_id}_{index}",
            "language": "en" if source_id == "medquad" else "fr",
            "instruction": "Answer.",
            "input": f"Question {source_id} {index}",
            "output": f"Answer {source_id} {index}",
            "source_ids": [source_id],
            "metadata": _metadata(source_id).to_dict(),
        }
    )


def _dpo_example(index: int) -> DPOExample:
    source_id = "ultramedical_preference"
    return DPOExample.from_mapping(
        {
            "id": f"dpo_{index}",
            "language": "en",
            "prompt": f"Prompt {index}",
            "chosen": f"Chosen {index}",
            "rejected": f"Rejected {index}",
            "source_ids": [source_id],
            "metadata": _metadata(source_id).to_dict(),
        }
    )

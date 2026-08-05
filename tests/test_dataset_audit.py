from __future__ import annotations

import pytest

from medical_triage_agent.dataset_audit import assert_split_isolation, audit_records
from medical_triage_agent.source_registry import SourceRecord
from tests.test_contracts import valid_sft_record

REGISTRY = {
    "verified_source": SourceRecord(
        id="verified_source",
        name="Verified",
        url="https://example.test/dataset",
        license="test-only",
        languages=("fr",),
        intended_use=("sft",),
        status="verified",
        notes="fixture",
    )
}


def test_audit_records_accepts_valid_sft_record() -> None:
    result = audit_records([valid_sft_record()], "sft", REGISTRY)

    assert result.ok
    assert result.checked == 1


def test_audit_records_rejects_unknown_source() -> None:
    record = valid_sft_record()
    record["source_ids"] = ["missing"]

    result = audit_records([record], "sft", REGISTRY)

    assert not result.ok
    assert "unknown source id" in result.errors[0]


def test_audit_records_rejects_possible_pii() -> None:
    record = valid_sft_record()
    record["input"] = "Patient joignable au 0612345678"

    result = audit_records([record], "sft", REGISTRY)

    assert not result.ok
    assert "possible PII" in result.errors[0]


def test_split_isolation_rejects_duplicate_ids_across_splits() -> None:
    record = valid_sft_record()

    with pytest.raises(ValueError, match="appears"):
        assert_split_isolation({"train": [record], "test": [record]})

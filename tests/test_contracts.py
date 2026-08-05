from __future__ import annotations

import pytest

from medical_triage_agent.contracts import SFTExample, deterministic_id


def valid_sft_record() -> dict[str, object]:
    payload: dict[str, object] = {
        "language": "fr",
        "instruction": "Evaluer le niveau de triage.",
        "input": "Douleur thoracique depuis 20 minutes.",
        "output": "Urgence maximale, avis clinique immediat.",
        "source_ids": ["verified_source"],
        "metadata": {
            "symptoms": ["douleur thoracique"],
            "antecedents": [],
            "vitals": {"temperature": 37.2},
            "triage_level": "urgence_maximale",
            "confidence": 0.9,
            "source": "Fixture",
            "license": "test-only",
            "transforms": ["fixture"],
        },
    }
    return {"id": deterministic_id("sft", payload), **payload}


def test_sft_contract_accepts_valid_record() -> None:
    example = SFTExample.from_mapping(valid_sft_record())

    assert example.language == "fr"
    assert example.metadata.triage_level == "urgence_maximale"


def test_sft_contract_rejects_missing_source_ids() -> None:
    record = valid_sft_record()
    record["source_ids"] = []

    with pytest.raises(ValueError, match="source_ids"):
        SFTExample.from_mapping(record)


def test_deterministic_id_is_stable_for_key_order() -> None:
    assert deterministic_id("x", {"a": 1, "b": 2}) == deterministic_id("x", {"b": 2, "a": 1})

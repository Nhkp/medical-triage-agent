from __future__ import annotations

import hashlib
import json
from typing import Any, Literal

from medical_triage_agent.contracts import DPOExample, Metadata, SFTExample, deterministic_id
from medical_triage_agent.privacy import redact_pii

SplitName = Literal["train", "validation", "test", "clinical_eval"]


def normalize_sft_record(
    *,
    language: Literal["fr", "en"],
    instruction: str,
    input_text: str,
    output_text: str,
    source_id: str,
    metadata: Metadata,
) -> SFTExample:
    """Redact and canonicalize raw supervised data into an SFTExample."""

    payload: dict[str, Any] = {
        "language": language,
        "instruction": redact_pii(instruction),
        "input": redact_pii(input_text),
        "output": redact_pii(output_text),
        "source_ids": [source_id],
        "metadata": metadata.to_dict(),
    }
    return SFTExample.from_mapping({"id": deterministic_id("sft", payload), **payload})


def normalize_dpo_record(
    *,
    language: Literal["fr", "en"],
    prompt: str,
    chosen: str,
    rejected: str,
    source_id: str,
    metadata: Metadata,
) -> DPOExample:
    """Redact and canonicalize raw preference data into a DPOExample."""

    payload: dict[str, Any] = {
        "language": language,
        "prompt": redact_pii(prompt),
        "chosen": redact_pii(chosen),
        "rejected": redact_pii(rejected),
        "source_ids": [source_id],
        "metadata": metadata.to_dict(),
    }
    return DPOExample.from_mapping({"id": deterministic_id("dpo", payload), **payload})


def assign_split(record_id: str) -> SplitName:
    """Assign a stable train/validation/test/clinical split from the record ID."""

    bucket = int(hashlib.sha256(record_id.encode("utf-8")).hexdigest()[:8], 16) % 100
    if bucket < 80:
        return "train"
    if bucket < 90:
        return "validation"
    if bucket < 97:
        return "test"
    return "clinical_eval"


def to_jsonl_line(record: SFTExample | DPOExample) -> str:
    """Serialize one normalized record as a deterministic JSONL line."""

    return json.dumps(record.to_dict(), ensure_ascii=False, sort_keys=True) + "\n"

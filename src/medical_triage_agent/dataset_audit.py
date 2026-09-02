from __future__ import annotations

from collections.abc import Iterable, Mapping
from dataclasses import dataclass
from typing import Any, Literal

from medical_triage_agent.contracts import DPOExample, SFTExample, content_fingerprint, load_jsonl
from medical_triage_agent.privacy import assert_no_pii
from medical_triage_agent.source_registry import (
    SourceRecord,
    load_source_registry,
    validate_source_ids,
)

DatasetKind = Literal["sft", "dpo"]


@dataclass(frozen=True)
class AuditResult:
    """Summary of dataset audit coverage and validation failures."""

    checked: int
    errors: tuple[str, ...]

    @property
    def ok(self) -> bool:
        """Return whether the audit completed without errors."""

        return not self.errors


def audit_records(
    records: Iterable[dict[str, Any]],
    kind: DatasetKind,
    registry: dict[str, SourceRecord] | None = None,
) -> AuditResult:
    """Validate dataset rows for schema, provenance, PII, and duplicates."""

    source_registry = registry if registry is not None else load_source_registry()
    errors: list[str] = []
    seen_ids: set[str] = set()
    seen_fingerprints: set[str] = set()
    checked = 0

    for index, record in enumerate(records, start=1):
        checked += 1
        try:
            example = (
                SFTExample.from_mapping(record)
                if kind == "sft"
                else DPOExample.from_mapping(record)
            )
            validate_source_ids(example.source_ids, source_registry)
            _assert_record_has_no_pii(record)
            if example.id in seen_ids:
                raise ValueError(f"duplicate id: {example.id}")
            seen_ids.add(example.id)
            fingerprint = content_fingerprint(record)
            if fingerprint in seen_fingerprints:
                raise ValueError("duplicate content fingerprint")
            seen_fingerprints.add(fingerprint)
        except (TypeError, ValueError) as exc:
            errors.append(f"record {index}: {exc}")

    return AuditResult(checked=checked, errors=tuple(errors))


def audit_jsonl(path: str, kind: DatasetKind) -> AuditResult:
    """Load and audit a JSONL dataset file."""

    return audit_records(load_jsonl(path), kind)


def assert_split_isolation(split_records: Mapping[str, Iterable[dict[str, Any]]]) -> None:
    """Ensure each record ID belongs to exactly one split."""

    owners: dict[str, str] = {}
    for split_name, records in split_records.items():
        for record in records:
            record_id = record.get("id")
            if not isinstance(record_id, str):
                raise TypeError(f"{split_name}: record id must be a string")
            if record_id in owners:
                raise ValueError(
                    f"record {record_id} appears in {owners[record_id]} and {split_name}"
                )
            owners[record_id] = split_name


def _assert_record_has_no_pii(record: dict[str, Any]) -> None:
    """Reject a record when any nested string matches PII detectors."""

    for value in _strings(record):
        assert_no_pii(value)


def _strings(value: Any) -> Iterable[str]:
    """Yield every string contained in a JSON-like value."""

    if isinstance(value, str):
        yield value
    elif isinstance(value, dict):
        for item in value.values():
            yield from _strings(item)
    elif isinstance(value, list):
        for item in value:
            yield from _strings(item)

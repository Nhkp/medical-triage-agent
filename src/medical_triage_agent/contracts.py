from __future__ import annotations

import hashlib
import json
from dataclasses import dataclass, field
from typing import Any, Literal, cast

Language = Literal["fr", "en"]
TriageLevel = Literal["urgence_maximale", "moderee", "differee"]


@dataclass(frozen=True)
class Metadata:
    """Provenance, safety labels, and transform history attached to each example."""

    symptoms: list[str]
    antecedents: list[str]
    vitals: dict[str, str | int | float]
    triage_level: TriageLevel
    confidence: float
    source: str
    license: str
    transforms: list[str] = field(default_factory=list)

    @classmethod
    def from_mapping(cls, data: dict[str, Any]) -> Metadata:
        """Build validated metadata from a JSON-like object."""

        return cls(
            symptoms=_string_list(data.get("symptoms"), "metadata.symptoms"),
            antecedents=_string_list(data.get("antecedents"), "metadata.antecedents"),
            vitals=_vitals(data.get("vitals")),
            triage_level=_triage_level(data.get("triage_level")),
            confidence=_confidence(data.get("confidence")),
            source=_required_string(data.get("source"), "metadata.source"),
            license=_required_string(data.get("license"), "metadata.license"),
            transforms=_string_list(data.get("transforms"), "metadata.transforms"),
        )

    def to_dict(self) -> dict[str, Any]:
        """Serialize metadata to the dataset JSON contract."""

        return {
            "symptoms": self.symptoms,
            "antecedents": self.antecedents,
            "vitals": self.vitals,
            "triage_level": self.triage_level,
            "confidence": self.confidence,
            "source": self.source,
            "license": self.license,
            "transforms": self.transforms,
        }


@dataclass(frozen=True)
class SFTExample:
    """Supervised fine-tuning record with source traceability and triage metadata."""

    id: str
    language: Language
    instruction: str
    input: str
    output: str
    source_ids: list[str]
    metadata: Metadata

    @classmethod
    def from_mapping(cls, data: dict[str, Any]) -> SFTExample:
        """Parse and validate an SFT record from a JSON-like mapping."""

        example = cls(
            id=_required_string(data.get("id"), "id"),
            language=_language(data.get("language")),
            instruction=_required_string(data.get("instruction"), "instruction"),
            input=_required_string(data.get("input"), "input"),
            output=_required_string(data.get("output"), "output"),
            source_ids=_string_list(data.get("source_ids"), "source_ids"),
            metadata=Metadata.from_mapping(_mapping(data.get("metadata"), "metadata")),
        )
        example.validate()
        return example

    def validate(self) -> None:
        """Enforce invariants that are not guaranteed by dataclass typing."""

        _require_source_ids(self.source_ids)

    def to_dict(self) -> dict[str, Any]:
        """Serialize the SFT record to its JSONL representation."""

        return {
            "id": self.id,
            "language": self.language,
            "instruction": self.instruction,
            "input": self.input,
            "output": self.output,
            "source_ids": self.source_ids,
            "metadata": self.metadata.to_dict(),
        }


@dataclass(frozen=True)
class DPOExample:
    """Preference-training record with chosen/rejected responses and provenance."""

    id: str
    language: Language
    prompt: str
    chosen: str
    rejected: str
    source_ids: list[str]
    metadata: Metadata

    @classmethod
    def from_mapping(cls, data: dict[str, Any]) -> DPOExample:
        """Parse and validate a DPO record from a JSON-like mapping."""

        example = cls(
            id=_required_string(data.get("id"), "id"),
            language=_language(data.get("language")),
            prompt=_required_string(data.get("prompt"), "prompt"),
            chosen=_required_string(data.get("chosen"), "chosen"),
            rejected=_required_string(data.get("rejected"), "rejected"),
            source_ids=_string_list(data.get("source_ids"), "source_ids"),
            metadata=Metadata.from_mapping(_mapping(data.get("metadata"), "metadata")),
        )
        example.validate()
        return example

    def validate(self) -> None:
        """Enforce preference-pair invariants before training ingestion."""

        _require_source_ids(self.source_ids)
        if self.chosen == self.rejected:
            raise ValueError("chosen and rejected responses must differ")

    def to_dict(self) -> dict[str, Any]:
        """Serialize the DPO record to its JSONL representation."""

        return {
            "id": self.id,
            "language": self.language,
            "prompt": self.prompt,
            "chosen": self.chosen,
            "rejected": self.rejected,
            "source_ids": self.source_ids,
            "metadata": self.metadata.to_dict(),
        }


def deterministic_id(prefix: str, payload: dict[str, Any]) -> str:
    """Create a stable short ID from canonicalized record content."""

    canonical = json.dumps(payload, ensure_ascii=False, sort_keys=True, separators=(",", ":"))
    digest = hashlib.sha256(canonical.encode("utf-8")).hexdigest()[:16]
    return f"{prefix}_{digest}"


def content_fingerprint(payload: dict[str, Any]) -> str:
    """Hash record content while ignoring the assigned ID."""

    content = {key: value for key, value in payload.items() if key != "id"}
    canonical = json.dumps(content, ensure_ascii=False, sort_keys=True, separators=(",", ":"))
    return hashlib.sha256(canonical.encode("utf-8")).hexdigest()


def load_jsonl(path: str) -> list[dict[str, Any]]:
    """Read a JSONL file and require each non-empty line to be an object."""

    rows: list[dict[str, Any]] = []
    with open(path, encoding="utf-8") as handle:
        for line_number, line in enumerate(handle, start=1):
            if not line.strip():
                continue
            value = json.loads(line)
            if not isinstance(value, dict):
                raise TypeError(f"{path}:{line_number}: expected object")
            rows.append(value)
    return rows


def _required_string(value: Any, field_name: str) -> str:
    """Return a non-empty string or raise a field-specific validation error."""

    if not isinstance(value, str) or not value.strip():
        raise ValueError(f"{field_name} must be a non-empty string")
    return value


def _string_list(value: Any, field_name: str) -> list[str]:
    """Return a list of non-empty strings for schema fields."""

    if not isinstance(value, list) or not all(isinstance(item, str) and item for item in value):
        raise ValueError(f"{field_name} must be a list of non-empty strings")
    return value


def _mapping(value: Any, field_name: str) -> dict[str, Any]:
    """Return a JSON object mapping for nested schema fields."""

    if not isinstance(value, dict):
        raise TypeError(f"{field_name} must be an object")
    return value


def _vitals(value: Any) -> dict[str, str | int | float]:
    """Validate vitals as scalar JSON values keyed by strings."""

    data = _mapping(value, "metadata.vitals")
    for key, item in data.items():
        if not isinstance(key, str) or not isinstance(item, str | int | float):
            raise TypeError("metadata.vitals must map strings to scalar values")
    return data


def _language(value: Any) -> Language:
    """Validate the supported bilingual language code."""

    if value not in {"fr", "en"}:
        raise ValueError("language must be 'fr' or 'en'")
    return cast(Language, value)


def _triage_level(value: Any) -> TriageLevel:
    """Validate the controlled triage priority vocabulary."""

    if value not in {"urgence_maximale", "moderee", "differee"}:
        raise ValueError("metadata.triage_level must be urgence_maximale, moderee, or differee")
    return cast(TriageLevel, value)


def _confidence(value: Any) -> float:
    """Validate confidence as an inclusive 0..1 score."""

    if not isinstance(value, int | float) or not 0 <= value <= 1:
        raise ValueError("metadata.confidence must be a number between 0 and 1")
    return float(value)


def _require_source_ids(source_ids: list[str]) -> None:
    """Reject records without provenance source IDs."""

    if not source_ids:
        raise ValueError("source_ids must not be empty")

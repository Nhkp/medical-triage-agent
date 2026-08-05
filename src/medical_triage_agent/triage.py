from __future__ import annotations

import hashlib
import json
from dataclasses import dataclass
from typing import Any

from medical_triage_agent.privacy import redact_pii

DISCLAIMER = (
    "POC d'aide au triage: cette evaluation ne remplace pas la decision d'un professionnel "
    "de sante."
)

RED_FLAGS = (
    "chest pain",
    "douleur thoracique",
    "difficulty breathing",
    "difficulte respiratoire",
    "shortness of breath",
    "avc",
    "stroke",
    "loss of consciousness",
    "perte de connaissance",
    "severe bleeding",
    "hemorragie",
    "major trauma",
    "suicidal",
    "suicide",
    "anaphylaxis",
    "seizure",
    "convulsion",
    "severe burn",
    "brulure grave",
)


@dataclass(frozen=True)
class TriageResponse:
    priority: str
    explanation: str
    disclaimer: str
    audit_id: str

    def to_dict(self) -> dict[str, str]:
        return {
            "priority": self.priority,
            "explanation": self.explanation,
            "disclaimer": self.disclaimer,
            "audit_id": self.audit_id,
        }


def assess_triage(payload: dict[str, Any]) -> TriageResponse:
    symptoms = payload.get("symptoms", [])
    symptom_text = " ".join(symptoms) if isinstance(symptoms, list) else str(symptoms)
    normalized = symptom_text.casefold()
    priority = "urgence_maximale" if any(flag in normalized for flag in RED_FLAGS) else "moderee"
    explanation = (
        "Symptomes d'alerte detectes: revue clinique immediate requise."
        if priority == "urgence_maximale"
        else "Aucun symptome d'alerte v1 detecte; revue clinique necessaire pour confirmer."
    )
    audit_id = _audit_id(payload)
    return TriageResponse(
        priority=priority,
        explanation=explanation,
        disclaimer=DISCLAIMER,
        audit_id=audit_id,
    )


def audit_metadata(payload: dict[str, Any], response: TriageResponse) -> dict[str, Any]:
    redacted_payload = _redact_value(payload)
    return {
        "audit_id": response.audit_id,
        "priority": response.priority,
        "model": "rule_based_v1",
        # Store a hash for traceability without exposing raw patient text through audit APIs.
        "payload_hash": _hash(redacted_payload),
    }


def _audit_id(payload: dict[str, Any]) -> str:
    return f"audit_{_hash(_redact_value(payload))[:16]}"


def _hash(value: Any) -> str:
    canonical = json.dumps(value, ensure_ascii=False, sort_keys=True, separators=(",", ":"))
    return hashlib.sha256(canonical.encode("utf-8")).hexdigest()


def _redact_value(value: Any) -> Any:
    if isinstance(value, str):
        return redact_pii(value)
    if isinstance(value, dict):
        return {key: _redact_value(item) for key, item in value.items()}
    if isinstance(value, list):
        return [_redact_value(item) for item in value]
    return value

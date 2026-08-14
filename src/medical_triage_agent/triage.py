from __future__ import annotations

import hashlib
import json
from dataclasses import dataclass
from datetime import UTC, datetime
from typing import Any

from medical_triage_agent.privacy import redact_pii

TRIAGE_ORDER = {"differee": 0, "moderee": 1, "urgence_maximale": 2}

DISCLAIMER = (
    "POC d'aide au triage: cette evaluation ne remplace pas la decision d'un professionnel "
    "de sante."
)

RED_FLAGS = (
    "chest pain",
    "douleur thoracique",
    "heart attack",
    "myocardial infarction",
    "infarctus",
    "crise cardiaque",
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
    explanation_source: str = "fallback"
    llm_status: str = "not_configured"
    rule_priority: str | None = None
    llm_priority: str | None = None
    llm_confidence: float | None = None
    llm_response_preview: str | None = None
    llm_response_truncated: bool = False
    priority_source: str = "rule"
    arbitration: str = "rule_only"

    def to_dict(self) -> dict[str, str]:
        return {
            "priority": self.priority,
            "rule_priority": self.rule_priority or self.priority,
            "llm_priority": self.llm_priority or "",
            "llm_confidence": "" if self.llm_confidence is None else str(self.llm_confidence),
            "priority_source": self.priority_source,
            "arbitration": self.arbitration,
            "explanation": self.explanation,
            "disclaimer": self.disclaimer,
            "audit_id": self.audit_id,
            "explanation_source": self.explanation_source,
            "llm_status": self.llm_status,
        }


def validate_payload(payload: dict[str, Any]) -> dict[str, Any]:
    symptoms = payload.get("symptoms")
    if not isinstance(symptoms, list) or not symptoms:
        raise ValueError("symptoms must be a non-empty list of strings")
    normalized_symptoms = []
    for symptom in symptoms:
        if not isinstance(symptom, str) or not symptom.strip():
            raise ValueError("symptoms must be a non-empty list of strings")
        normalized_symptoms.append(symptom.strip()[:500])
    normalized = dict(payload)
    normalized["symptoms"] = normalized_symptoms[:20]
    return normalized


def assess_triage(payload: dict[str, Any]) -> TriageResponse:
    payload = validate_payload(payload)
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


def audit_metadata(
    payload: dict[str, Any], response: TriageResponse, *, model: str = "rule_based_v1"
) -> dict[str, Any]:
    redacted_payload = _redact_value(payload)
    return {
        "audit_id": response.audit_id,
        "priority": response.priority,
        "rule_priority": response.rule_priority or response.priority,
        "llm_priority": response.llm_priority,
        "llm_confidence": response.llm_confidence,
        "llm_response_preview": (
            redact_pii(response.llm_response_preview)
            if response.llm_response_preview is not None
            else None
        ),
        "llm_response_truncated": response.llm_response_truncated,
        "priority_source": response.priority_source,
        "arbitration": response.arbitration,
        "model": model,
        "explanation_source": response.explanation_source,
        "llm_status": response.llm_status,
        "created_at": datetime.now(UTC).isoformat(),
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

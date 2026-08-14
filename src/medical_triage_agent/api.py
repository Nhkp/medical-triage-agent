from __future__ import annotations

from typing import Any

from medical_triage_agent.triage import TRIAGE_ORDER, TriageResponse, assess_triage, audit_metadata
from medical_triage_agent.vllm_client import configured_model, generate_triage, is_configured

_AUDIT_STORE: dict[str, dict[str, Any]] = {}


def triage(payload: dict[str, Any]) -> dict[str, str]:
    rule_response = assess_triage(payload)
    generation_result = generate_triage(payload, rule_response)
    final_priority, priority_source, arbitration = _arbitrate_priority(
        rule_response.priority, generation_result.suggested_priority
    )
    response = TriageResponse(
        priority=final_priority,
        rule_priority=rule_response.priority,
        llm_priority=generation_result.suggested_priority,
        llm_confidence=generation_result.confidence,
        llm_response_preview=generation_result.llm_response_preview,
        llm_response_truncated=generation_result.llm_response_truncated,
        priority_source=priority_source,
        arbitration=arbitration,
        explanation=generation_result.explanation or rule_response.explanation,
        disclaimer=rule_response.disclaimer,
        audit_id=rule_response.audit_id,
        explanation_source=generation_result.explanation_source,
        llm_status=generation_result.llm_status,
    )
    _AUDIT_STORE[response.audit_id] = audit_metadata(payload, response, model=configured_model())
    return response.to_dict()


def _arbitrate_priority(rule_priority: str, llm_priority: str | None) -> tuple[str, str, str]:
    if llm_priority is None:
        return rule_priority, "rule", "rule_only"
    if TRIAGE_ORDER[llm_priority] > TRIAGE_ORDER[rule_priority]:
        return llm_priority, "llm", "llm_escalated"
    if TRIAGE_ORDER[llm_priority] < TRIAGE_ORDER[rule_priority]:
        return rule_priority, "rule", "rule_escalated"
    return rule_priority, "shared", "matched"


def audit(audit_id: str) -> dict[str, Any] | None:
    return _AUDIT_STORE.get(audit_id)


def health() -> dict[str, str]:
    return {
        "status": "ok",
        "model": configured_model(),
        "vllm": "configured" if is_configured() else "disabled",
    }


def create_app() -> Any:
    from fastapi import FastAPI, HTTPException

    app = FastAPI(title="CHSA medical triage POC")

    @app.get("/health")
    def health_endpoint() -> dict[str, str]:
        return health()

    @app.post("/triage")
    def triage_endpoint(payload: dict[str, Any]) -> dict[str, str]:
        try:
            return triage(payload)
        except ValueError as exc:
            raise HTTPException(status_code=400, detail=str(exc)) from exc

    @app.get("/audit/{audit_id}")
    def audit_endpoint(audit_id: str) -> dict[str, Any]:
        record = audit(audit_id)
        if record is None:
            raise HTTPException(status_code=404, detail="audit record not found")
        return record

    return app

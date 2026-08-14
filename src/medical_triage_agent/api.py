from __future__ import annotations

from typing import Any

from medical_triage_agent.triage import TriageResponse, assess_triage, audit_metadata
from medical_triage_agent.vllm_client import configured_model, generate_explanation, is_configured

_AUDIT_STORE: dict[str, dict[str, Any]] = {}


def triage(payload: dict[str, Any]) -> dict[str, str]:
    response = assess_triage(payload)
    explanation_result = generate_explanation(payload, response)
    response = TriageResponse(
        priority=response.priority,
        explanation=explanation_result.explanation or response.explanation,
        disclaimer=response.disclaimer,
        audit_id=response.audit_id,
        explanation_source=explanation_result.explanation_source,
        llm_status=explanation_result.llm_status,
    )
    _AUDIT_STORE[response.audit_id] = audit_metadata(payload, response, model=configured_model())
    return response.to_dict()


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

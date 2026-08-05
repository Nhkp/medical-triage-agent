from __future__ import annotations

from medical_triage_agent import api
from medical_triage_agent.triage import DISCLAIMER, assess_triage


def test_red_flag_symptom_escalates_to_urgence_maximale() -> None:
    response = assess_triage({"symptoms": ["Douleur thoracique"]})

    assert response.priority == "urgence_maximale"
    assert response.disclaimer == DISCLAIMER
    assert response.audit_id.startswith("audit_")


def test_api_audit_returns_metadata_without_raw_patient_text() -> None:
    response = api.triage({"symptoms": ["Contact: patient@example.test", "fatigue"]})
    audit_record = api.audit(response["audit_id"])

    assert audit_record is not None
    assert audit_record["priority"] == "moderee"
    assert "patient@example.test" not in str(audit_record)


def test_health_reports_rule_based_fallback() -> None:
    assert api.health() == {"status": "ok", "model": "rule_based_v1", "vllm": "disabled"}

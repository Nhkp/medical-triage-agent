from __future__ import annotations

from pytest import MonkeyPatch

from medical_triage_agent import api
from medical_triage_agent.triage import DISCLAIMER, assess_triage
from medical_triage_agent.vllm_client import _extract_content


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
    assert audit_record["model"] == "rule_based_v1"
    assert "created_at" in audit_record
    assert "patient@example.test" not in str(audit_record)


def test_health_reports_rule_based_fallback() -> None:
    assert api.health() == {"status": "ok", "model": "rule_based_v1", "vllm": "disabled"}


def test_api_rejects_empty_symptom_payload() -> None:
    try:
        api.triage({})
    except ValueError as exc:
        assert "symptoms" in str(exc)
    else:
        raise AssertionError("empty symptoms should be rejected")


def test_invalid_vllm_explanation_falls_back(monkeypatch: MonkeyPatch) -> None:
    monkeypatch.setenv("VLLM_BASE_URL", "http://vllm.test/v1")
    monkeypatch.setattr(api, "generate_explanation", lambda _payload, _response: None)

    response = api.triage({"symptoms": ["douleur thoracique", "difficulte respiratoire"]})

    assert response["priority"] == "urgence_maximale"
    assert (
        response["explanation"] == "Symptomes d'alerte detectes: revue clinique immediate requise."
    )


def test_vllm_rejects_non_latin_repetitive_output() -> None:
    payload = {"choices": [{"message": {"content": "具有战士ันันันันันันันันันันันันันันันันันันันันันันันัน"}}]}

    assert _extract_content(payload) is None

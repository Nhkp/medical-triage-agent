from __future__ import annotations

from pytest import MonkeyPatch

from medical_triage_agent import api
from medical_triage_agent.triage import DISCLAIMER, assess_triage
from medical_triage_agent.vllm_client import SYSTEM_PROMPT, _extract_content, build_chat_request


def test_red_flag_symptom_escalates_to_urgence_maximale() -> None:
    response = assess_triage({"symptoms": ["Douleur thoracique"]})

    assert response.priority == "urgence_maximale"
    assert response.disclaimer == DISCLAIMER
    assert response.audit_id.startswith("audit_")


def test_reported_infarctus_escalates_to_urgence_maximale() -> None:
    response = assess_triage({"symptoms": ["suspicion d'infarctus"]})

    assert response.priority == "urgence_maximale"


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


def test_vllm_system_prompt_limits_model_to_explanation_role() -> None:
    prompt = SYSTEM_PROMPT.casefold()

    assert "only to explain" in prompt
    assert "do not change" in prompt
    assert "latin-script french or english" in prompt
    assert "do not provide a diagnosis" in prompt
    assert "human clinical review remains required" in prompt
    assert "urgence_maximale" in prompt
    assert "immediate clinical review or escalation" in prompt


def test_vllm_request_context_keeps_only_expected_fields() -> None:
    response = assess_triage({"symptoms": ["douleur thoracique"]})
    request = build_chat_request(
        {
            "symptoms": ["douleur thoracique"],
            "questionnaire_state": {"duration": "10 min"},
            "patient_name": "Alice",
            "notes": "unrelated raw field",
        },
        response,
    )

    user_payload = request["messages"][1]["content"]

    assert '"priority": "urgence_maximale"' in user_payload
    assert '"symptoms": ["douleur thoracique"]' in user_payload
    assert "questionnaire_state" in user_payload
    assert "patient_name" not in user_payload
    assert "unrelated raw field" not in user_payload
    assert request["temperature"] == 0
    assert request["max_tokens"] == 160


def test_vllm_accepts_valid_french_and_english_explanations() -> None:
    french = {
        "choices": [
            {
                "message": {
                    "content": (
                        "Cette priorite impose une revue clinique immediate. "
                        "Les elements declares restent incertains et doivent etre confirmes par un professionnel."
                    )
                }
            }
        ]
    }
    english = {
        "choices": [
            {
                "message": {
                    "content": (
                        "This priority requires immediate clinical review. "
                        "The declared symptoms remain uncertain and must be confirmed by a clinician."
                    )
                }
            }
        ]
    }

    assert _extract_content(french) is not None
    assert _extract_content(english) is not None


def test_vllm_rejects_diagnostic_or_treatment_advice() -> None:
    diagnostic = {
        "choices": [{"message": {"content": "The diagnosis is asthma. Take 500 mg now."}}]
    }

    assert _extract_content(diagnostic) is None

from __future__ import annotations

import json
from typing import Any, Self

from pytest import MonkeyPatch

from medical_triage_agent import api
from medical_triage_agent.triage import DISCLAIMER, assess_triage
from medical_triage_agent.vllm_client import (
    SYSTEM_PROMPT,
    ExplanationResult,
    _extract_content,
    _request_timeout,
    build_chat_request,
    extract_explanation,
    generate_explanation,
)


class _Response:
    def __init__(self, payload: bytes) -> None:
        self._payload = payload

    def __enter__(self) -> Self:
        return self

    def __exit__(self, *_args: object) -> None:
        return None

    def read(self) -> bytes:
        return self._payload


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
    assert audit_record["rule_priority"] == "moderee"
    assert audit_record["llm_priority"] is None
    assert audit_record["priority_source"] == "rule"
    assert audit_record["arbitration"] == "rule_only"
    assert audit_record["model"] == "rule_based_v1"
    assert audit_record["explanation_source"] == "fallback"
    assert audit_record["llm_status"] == "not_configured"
    assert audit_record["llm_response_preview"] is None
    assert audit_record["llm_response_truncated"] is False
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
    monkeypatch.setattr(
        api,
        "generate_triage",
        lambda _payload, _response: ExplanationResult(None, "invalid_output"),
    )

    response = api.triage({"symptoms": ["douleur thoracique", "difficulte respiratoire"]})

    assert response["priority"] == "urgence_maximale"
    assert response["rule_priority"] == "urgence_maximale"
    assert response["llm_priority"] == ""
    assert response["priority_source"] == "rule"
    assert response["arbitration"] == "rule_only"
    assert response["explanation_source"] == "fallback"
    assert response["llm_status"] == "invalid_output"
    assert (
        response["explanation"] == "Symptomes d'alerte detectes: revue clinique immediate requise."
    )


def test_valid_vllm_explanation_is_used(monkeypatch: MonkeyPatch) -> None:
    explanation = (
        "Cette priorite indique une situation a faire revoir sans delai par un clinicien. "
        "Les symptomes declares doivent etre confirmes dans le contexte clinique complet."
    )
    monkeypatch.setattr(
        api,
        "generate_triage",
        lambda _payload, _response: ExplanationResult(
            explanation,
            "accepted",
            "urgence_maximale",
            0.8,
            '{"suggested_priority":"urgence_maximale","explanation":"ok","confidence":0.8}',
        ),
    )

    response = api.triage({"symptoms": ["douleur thoracique"]})

    assert response["explanation"] == explanation
    assert response["priority"] == "urgence_maximale"
    assert response["rule_priority"] == "urgence_maximale"
    assert response["llm_priority"] == "urgence_maximale"
    assert response["llm_confidence"] == "0.8"
    assert response["priority_source"] == "shared"
    assert response["arbitration"] == "matched"
    assert response["explanation_source"] == "llm"
    assert response["llm_status"] == "accepted"
    audit_record = api.audit(response["audit_id"])
    assert audit_record is not None
    assert audit_record["llm_response_preview"] is not None
    assert "suggested_priority" in audit_record["llm_response_preview"]


def test_llm_priority_mismatch_uses_rule_fallback(monkeypatch: MonkeyPatch) -> None:
    explanation = (
        "Cette situation justifie une evaluation clinique urgente malgre l'absence de red flag exact. "
        "Les elements restent declaratifs et doivent etre verifies par un soignant."
    )
    monkeypatch.setattr(
        api,
        "generate_triage",
        lambda _payload, _response: ExplanationResult(
            explanation, "accepted", "urgence_maximale", 0.7
        ),
    )

    response = api.triage({"symptoms": ["malaise important"]})

    assert response["priority"] == "moderee"
    assert response["rule_priority"] == "moderee"
    assert response["llm_priority"] == "urgence_maximale"
    assert response["priority_source"] == "rule"
    assert response["arbitration"] == "llm_priority_mismatch"
    assert response["explanation_source"] == "fallback"
    assert (
        response["explanation"]
        == "Aucun symptome d'alerte v1 detecte; revue clinique necessaire pour confirmer."
    )


def test_rules_override_lower_llm_priority_for_red_flag(monkeypatch: MonkeyPatch) -> None:
    explanation = (
        "La suggestion du modele reste incertaine et doit etre revue par un clinicien. "
        "Les symptomes declares imposent une verification clinique immediate."
    )
    monkeypatch.setattr(
        api,
        "generate_triage",
        lambda _payload, _response: ExplanationResult(explanation, "accepted", "differee", 0.6),
    )

    response = api.triage({"symptoms": ["douleur thoracique"]})

    assert response["priority"] == "urgence_maximale"
    assert response["rule_priority"] == "urgence_maximale"
    assert response["llm_priority"] == "differee"
    assert response["priority_source"] == "rule"
    assert response["arbitration"] == "llm_priority_mismatch"
    assert response["explanation_source"] == "fallback"


def test_repaired_vllm_output_can_be_used_by_api(monkeypatch: MonkeyPatch) -> None:
    raw_content = (
        "taxpipeline: medical triage rules\n\n"
        "Reponse: urgence_maximale\n\n"
        "Explanation: Les symptomes declares justifient une revue clinique immediate. "
        "Les elements restent incertains et doivent etre confirmes par un professionnel."
    )
    payload = {"choices": [{"message": {"content": raw_content}}]}

    def fake_urlopen(_request: Any, timeout: float | None) -> _Response:
        return _Response(json.dumps(payload).encode("utf-8"))

    monkeypatch.setenv("VLLM_BASE_URL", "http://vllm.test/v1")
    monkeypatch.setattr("medical_triage_agent.vllm_client.urlopen", fake_urlopen)

    response = api.triage({"symptoms": ["douleur thoracique"]})
    audit_record = api.audit(response["audit_id"])

    assert response["priority"] == "urgence_maximale"
    assert response["rule_priority"] == "urgence_maximale"
    assert response["llm_priority"] == "urgence_maximale"
    assert response["priority_source"] == "shared"
    assert response["arbitration"] == "matched"
    assert response["explanation_source"] == "llm"
    assert response["llm_status"] == "accepted_repaired"
    assert audit_record is not None
    assert "Reponse: urgence_maximale" in audit_record["llm_response_preview"]


def test_repaired_vllm_output_cannot_lower_red_flag_priority(monkeypatch: MonkeyPatch) -> None:
    raw_content = (
        "Priority: differee\n\n"
        "Explanation: The declared symptoms remain uncertain and require clinical review. "
        "A clinician must confirm the priority before any operational decision."
    )
    payload = {"choices": [{"message": {"content": raw_content}}]}

    def fake_urlopen(_request: Any, timeout: float | None) -> _Response:
        return _Response(json.dumps(payload).encode("utf-8"))

    monkeypatch.setenv("VLLM_BASE_URL", "http://vllm.test/v1")
    monkeypatch.setattr("medical_triage_agent.vllm_client.urlopen", fake_urlopen)

    response = api.triage({"symptoms": ["douleur thoracique"]})

    assert response["priority"] == "urgence_maximale"
    assert response["rule_priority"] == "urgence_maximale"
    assert response["llm_priority"] == "differee"
    assert response["priority_source"] == "rule"
    assert response["arbitration"] == "llm_priority_mismatch"
    assert response["explanation_source"] == "fallback"
    assert response["llm_status"] == "accepted_repaired"


def test_vllm_rejects_non_latin_repetitive_output() -> None:
    payload = {
        "choices": [
            {
                "message": {
                    "content": json.dumps(
                        {
                            "suggested_priority": "moderee",
                            "explanation": "具有战士ันันันันันันันันันันันันันันันันันันันันันันันัน",
                            "confidence": 0.8,
                        },
                        ensure_ascii=False,
                    )
                }
            }
        ]
    }

    assert _extract_content(payload) is None
    assert extract_explanation(payload).llm_status == "invalid_output"


def test_vllm_system_prompt_limits_model_to_explanation_role() -> None:
    prompt = SYSTEM_PROMPT.casefold()

    assert "suggest one triage priority" in prompt
    assert "backend keeps final authority" in prompt
    assert "latin-script french or english" in prompt
    assert "do not provide a diagnosis" in prompt
    assert "human clinical review remains required" in prompt
    assert "suggested_priority" in prompt
    assert "confidence" in prompt
    assert "return exactly one json object and nothing else" in prompt
    assert '"suggested_priority": "urgence_maximale" | "moderee" | "differee"' in prompt
    assert "3 sentences for urgence_maximale" in prompt
    assert "2 sentences for moderee" in prompt
    assert "do not repeat any sentence" in prompt
    assert "do not introduce symptoms, diagnoses, diseases, treatments, hospitalization" in prompt
    assert "explain why the declared symptoms support the suggested priority" in prompt
    assert "taxpipeline" in prompt
    assert "aucun symptome d'alerte" in prompt
    assert "no markdown, no headings, no text outside json" in prompt


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

    assert '"rule_priority": "urgence_maximale"' in user_payload
    assert '"symptoms": ["douleur thoracique"]' in user_payload
    assert "draft_explanation" not in user_payload
    assert "Symptomes d'alerte detectes" not in user_payload
    assert "questionnaire_state" in user_payload
    assert "patient_name" not in user_payload
    assert "unrelated raw field" not in user_payload
    assert request["temperature"] == 0
    assert request["max_tokens"] == 110
    assert "具有战士" in request["stop"]
    assert "具有战士user" in request["stop"]
    assert "\nassistant" in request["stop"]


def test_vllm_accepts_valid_french_and_english_explanations() -> None:
    french = {
        "choices": [
            {
                "message": {
                    "content": json.dumps(
                        {
                            "suggested_priority": "urgence_maximale",
                            "explanation": (
                                "Cette priorite impose une revue clinique immediate. "
                                "Les elements declares restent incertains et doivent etre confirmes par un professionnel."
                            ),
                            "confidence": 0.9,
                        }
                    )
                }
            }
        ]
    }
    english = {
        "choices": [
            {
                "message": {
                    "content": json.dumps(
                        {
                            "suggested_priority": "urgence_maximale",
                            "explanation": (
                                "This priority requires immediate clinical review. "
                                "The declared symptoms remain uncertain and must be confirmed by a clinician."
                            ),
                            "confidence": 0.8,
                        }
                    )
                }
            }
        ]
    }

    assert _extract_content(french) is not None
    assert _extract_content(english) is not None
    assert extract_explanation(french).llm_status == "accepted"


def test_vllm_repairs_known_non_json_output_patterns() -> None:
    samples = [
        (
            (
                "taxpipeline: medical triage rules\n\n"
                "Reponse: urgence_maximale\n\n"
                "Explanation: Les symptomes declares justifient une revue clinique immediate. "
                "Les elements restent incertains et doivent etre confirmes par un professionnel."
            ),
            "urgence_maximale",
        ),
        (
            (
                "Réponse: moderee\n\n"
                "Explication: Les symptomes declares necessitent une revue clinique organisee. "
                "L'incertitude impose une confirmation par un professionnel de sante."
            ),
            "moderee",
        ),
        (
            (
                "Priority: differee\n\n"
                "Explanation: The declared symptoms do not show an immediate alert in this context. "
                "A clinician should still confirm the priority because uncertainty remains."
            ),
            "differee",
        ),
    ]

    for content, expected_priority in samples:
        payload = {"choices": [{"message": {"content": content}}]}
        result = extract_explanation(payload)

        assert result.llm_status == "accepted_repaired"
        assert result.explanation_source == "llm"
        assert result.suggested_priority == expected_priority
        assert result.confidence == 0.5
        assert result.llm_response_preview is not None


def test_vllm_repairs_separator_corrupted_object_from_first_block() -> None:
    content = (
        '{lng: "Reponse: urgence_maximale", "confidence": 1.0, "explanation": '
        "\"Les symptômes d'alerte indiquent une urgence maximale. La douleur thoracique "
        'et la difficulté respiratoire nécessitent une revue clinique immédiate."}'
        "具有战士\n具有战士user\nAnswer the medical question: How to treat Hypertension ?"
        "具有战士\n具有战士assistant\n"
        '{lng: "Reponse: traitement par un médicament antihypertenseur", "confidence": 1.0}'
    )
    payload = {"choices": [{"message": {"content": content}}]}

    result = extract_explanation(payload)

    assert result.llm_status == "accepted_repaired"
    assert result.explanation_source == "llm"
    assert result.suggested_priority == "urgence_maximale"
    assert result.confidence == 1.0
    assert result.llm_response_preview is not None
    assert "How to treat Hypertension" in result.llm_response_preview


def test_vllm_repairs_direct_lng_priority_when_explanation_is_generated() -> None:
    content = (
        '{lng: "moderee", "explanation": "Les céphalées associées à un nez qui coule '
        "ne signalent pas ici une urgence immédiate. Une revue clinique reste nécessaire "
        'pour confirmer la priorité et le contexte.", "confidence": 0.5}'
        "具有战士\nInitialized a new CHSA medical triage proof-of-concept"
    )
    payload = {"choices": [{"message": {"content": content}}]}

    result = extract_explanation(payload)

    assert result.llm_status == "accepted_repaired"
    assert result.suggested_priority == "moderee"
    assert result.explanation_source == "llm"


def test_vllm_rejects_object_that_copies_rule_fallback_explanation() -> None:
    content = (
        '{lng: "moderee", "explanation": "Aucun symptome d\'alerte v1 detecte; '
        'revue clinique necessaire pour confirmer.", "confidence": 0.5}'
        "具有战士\nInitialized a new CHSA medical triage proof-of-concept"
    )
    payload = {"choices": [{"message": {"content": content}}]}

    assert extract_explanation(payload).llm_status == "invalid_output"


def test_api_uses_rule_fallback_when_repaired_llm_priority_differs(
    monkeypatch: MonkeyPatch,
) -> None:
    content = (
        "taxpipeline: symptoms_to_priority\n\n"
        "Reponse: urgence_maximale\n\n"
        "Explication: Les céphalées et le nez qui coule demandent une revue clinique. "
        "Le niveau propose par le modele reste incertain et doit etre confirme."
    )
    payload = {"choices": [{"message": {"content": content}}]}

    def fake_urlopen(_request: Any, timeout: float | None) -> _Response:
        return _Response(json.dumps(payload).encode("utf-8"))

    monkeypatch.setenv("VLLM_BASE_URL", "http://vllm.test/v1")
    monkeypatch.setattr("medical_triage_agent.vllm_client.urlopen", fake_urlopen)

    response = api.triage({"symptoms": ["céphalées", "nez qui coule"]})

    assert response["priority"] == "moderee"
    assert response["rule_priority"] == "moderee"
    assert response["llm_priority"] == "urgence_maximale"
    assert response["priority_source"] == "rule"
    assert response["arbitration"] == "llm_priority_mismatch"
    assert response["explanation_source"] == "fallback"
    assert (
        response["explanation"]
        == "Aucun symptome d'alerte v1 detecte; revue clinique necessaire pour confirmer."
    )


def test_api_uses_repaired_separator_corrupted_output(monkeypatch: MonkeyPatch) -> None:
    content = (
        '{lng: "Reponse: urgence_maximale", "confidence": 1.0, "explanation": '
        "\"Les symptômes d'alerte indiquent une urgence maximale. La douleur thoracique "
        'et la difficulté respiratoire nécessitent une revue clinique immédiate."}'
        "具有战士\n具有战士user\nAnswer the medical question: How to treat Hypertension ?"
    )
    payload = {"choices": [{"message": {"content": content}}]}

    def fake_urlopen(_request: Any, timeout: float | None) -> _Response:
        return _Response(json.dumps(payload).encode("utf-8"))

    monkeypatch.setenv("VLLM_BASE_URL", "http://vllm.test/v1")
    monkeypatch.setattr("medical_triage_agent.vllm_client.urlopen", fake_urlopen)

    response = api.triage({"symptoms": ["douleur thoracique", "difficulte respiratoire"]})
    audit_record = api.audit(response["audit_id"])

    assert response["priority"] == "urgence_maximale"
    assert response["llm_priority"] == "urgence_maximale"
    assert response["priority_source"] == "shared"
    assert response["arbitration"] == "matched"
    assert response["explanation_source"] == "llm"
    assert response["llm_status"] == "accepted_repaired"
    assert audit_record is not None
    assert "How to treat Hypertension" in audit_record["llm_response_preview"]
    assert "douleur thoracique" not in str(audit_record["payload_hash"])


def test_vllm_rejects_object_when_explanation_contains_separator_noise() -> None:
    content = (
        '{lng: "Reponse: urgence_maximale", "confidence": 1.0, "explanation": '
        '"Les symptômes imposent une revue clinique immédiate 具有战士."}'
    )
    payload = {"choices": [{"message": {"content": content}}]}

    assert extract_explanation(payload).llm_status == "invalid_output"


def test_vllm_rejects_object_when_qa_continuation_enters_explanation() -> None:
    content = (
        '{lng: "Reponse: moderee", "confidence": 1.0, "explanation": '
        '"The symptoms require clinical review. How to treat Hypertension ?"}'
    )
    payload = {"choices": [{"message": {"content": content}}]}

    assert extract_explanation(payload).llm_status == "invalid_output"


def test_vllm_rejects_object_with_invalid_lng_priority() -> None:
    content = (
        '{lng: "Reponse: critical", "confidence": 1.0, "explanation": '
        '"Les symptomes declares necessitent une revue clinique. '
        'Un professionnel doit confirmer la priorite."}'
    )
    payload = {"choices": [{"message": {"content": content}}]}

    assert extract_explanation(payload).llm_status == "invalid_output"


def test_vllm_bad_response_for_object_without_usable_explanation() -> None:
    payload = {
        "choices": [
            {"message": {"content": '{lng: "Reponse: urgence_maximale", "confidence": 1.0}'}}
        ]
    }

    assert extract_explanation(payload).llm_status == "bad_response"


def test_vllm_repairs_repeated_explanation_by_deduplicating_sentences() -> None:
    payload = {
        "choices": [
            {
                "message": {
                    "content": (
                        "Reponse: urgence_maximale\n\n"
                        "Explanation: La revue clinique est obligatoire. "
                        "La douleur thoracique doit etre revue sans delai. "
                        "La revue clinique est obligatoire."
                    )
                }
            }
        ]
    }

    result = extract_explanation(payload)

    assert result.llm_status == "accepted_repaired"
    assert result.suggested_priority == "urgence_maximale"
    assert result.explanation == (
        "La revue clinique est obligatoire. La douleur thoracique doit etre revue sans delai."
    )


def test_vllm_repairs_repeated_sentence_blocks_by_keeping_first_iteration() -> None:
    payload = {
        "choices": [
            {
                "message": {
                    "content": json.dumps(
                        {
                            "suggested_priority": "moderee",
                            "explanation": (
                                "La respiration declaree est normale. "
                                "Les symptomes restent a confirmer par un clinicien. "
                                "Une surveillance clinique reste appropriee. "
                                "La respiration declaree est normale. "
                                "Les symptomes restent a confirmer par un clinicien. "
                                "Une surveillance clinique reste appropriee."
                            ),
                            "confidence": 0.5,
                        },
                        ensure_ascii=False,
                    )
                }
            }
        ]
    }

    result = extract_explanation(payload)

    assert result.llm_status == "accepted_repaired"
    assert result.suggested_priority == "moderee"
    assert result.explanation == (
        "La respiration declaree est normale. "
        "Les symptomes restent a confirmer par un clinicien. "
        "Une surveillance clinique reste appropriee."
    )


def test_vllm_bad_response_when_lng_contains_explanation_without_priority() -> None:
    payload = {
        "choices": [
            {
                "message": {
                    "content": (
                        '{lng: "La respiration est normale. Le patient est apyrétique. '
                        "Il n'a pas de signes d'infection pulmonaire.\"}"
                    )
                }
            }
        ]
    }

    result = extract_explanation(payload)

    assert result.llm_status == "bad_response"
    assert result.suggested_priority is None


def test_vllm_rejects_repeated_explanation_when_deduplication_is_too_short() -> None:
    payload = {
        "choices": [
            {
                "message": {
                    "content": (
                        "Reponse: moderee\n\n"
                        "Explanation: Revue clinique requise. Revue clinique requise."
                    )
                }
            }
        ]
    }

    assert extract_explanation(payload).llm_status == "invalid_output"


def test_vllm_rejects_contradictory_urgent_explanation() -> None:
    payload = {
        "choices": [
            {
                "message": {
                    "content": (
                        "Reponse: urgence_maximale\n\n"
                        "Explanation: The declared symptoms do not require urgent treatment. "
                        "Simple care is enough before a clinician confirms the context."
                    )
                }
            }
        ]
    }

    assert extract_explanation(payload).llm_status == "invalid_output"


def test_api_can_use_matching_moderee_repaired_explanation(monkeypatch: MonkeyPatch) -> None:
    content = (
        "Reponse: moderee\n\n"
        "Explication: Les symptômes déclarés ne montrent pas ici de signal immédiat critique. "
        "Une revue clinique reste nécessaire pour confirmer le contexte."
    )
    payload = {"choices": [{"message": {"content": content}}]}

    def fake_urlopen(_request: Any, timeout: float | None) -> _Response:
        return _Response(json.dumps(payload).encode("utf-8"))

    monkeypatch.setenv("VLLM_BASE_URL", "http://vllm.test/v1")
    monkeypatch.setattr("medical_triage_agent.vllm_client.urlopen", fake_urlopen)

    response = api.triage({"symptoms": ["céphalées", "nez qui coule"]})

    assert response["priority"] == "moderee"
    assert response["rule_priority"] == "moderee"
    assert response["llm_priority"] == "moderee"
    assert response["priority_source"] == "shared"
    assert response["arbitration"] == "matched"
    assert response["explanation_source"] == "llm"
    assert response["llm_status"] == "accepted_repaired"


def test_vllm_rejects_repaired_unsafe_advice() -> None:
    payload = {
        "choices": [
            {
                "message": {
                    "content": (
                        "Priority: moderee\n\n"
                        "Explanation: The diagnosis is asthma. Take 500 mg now."
                    )
                }
            }
        ]
    }

    assert extract_explanation(payload).llm_status == "invalid_output"


def test_vllm_rejects_diagnostic_or_treatment_advice() -> None:
    diagnostic = {
        "choices": [
            {
                "message": {
                    "content": json.dumps(
                        {
                            "suggested_priority": "moderee",
                            "explanation": "The diagnosis is asthma. Take 500 mg now.",
                            "confidence": 0.8,
                        }
                    )
                }
            }
        ]
    }

    assert _extract_content(diagnostic) is None
    assert extract_explanation(diagnostic).llm_status == "invalid_output"


def test_vllm_bad_response_has_distinct_status() -> None:
    assert extract_explanation({"choices": []}).llm_status == "bad_response"
    assert (
        extract_explanation({"choices": [{"message": {"content": "not json"}}]}).llm_status
        == "bad_response"
    )


def test_vllm_rejects_invalid_priority_label() -> None:
    payload = {
        "choices": [
            {
                "message": {
                    "content": json.dumps(
                        {
                            "suggested_priority": "critical",
                            "explanation": (
                                "Cette reponse reste incertaine et doit etre revue par un clinicien. "
                                "La priorite doit etre confirmee avec le contexte clinique."
                            ),
                            "confidence": 0.8,
                        }
                    )
                }
            }
        ]
    }

    assert extract_explanation(payload).llm_status == "invalid_output"


def test_vllm_timeout_defaults_to_none(monkeypatch: MonkeyPatch) -> None:
    monkeypatch.delenv("VLLM_TIMEOUT_SECONDS", raising=False)

    assert _request_timeout() is None


def test_vllm_timeout_can_be_configured(monkeypatch: MonkeyPatch) -> None:
    monkeypatch.setenv("VLLM_TIMEOUT_SECONDS", "300")

    assert _request_timeout() == 300


def test_generate_explanation_returns_llm_status_for_valid_response(
    monkeypatch: MonkeyPatch,
) -> None:
    explanation = (
        "Cette priorite necessite une revue clinique immediate. "
        "Les signes declares doivent etre confirmes par un professionnel de sante."
    )
    payload = {
        "choices": [
            {
                "message": {
                    "content": json.dumps(
                        {
                            "suggested_priority": "urgence_maximale",
                            "explanation": explanation,
                            "confidence": 0.9,
                        }
                    )
                }
            }
        ]
    }

    def fake_urlopen(_request: Any, timeout: float | None) -> _Response:
        assert timeout is None
        return _Response(json.dumps(payload).encode("utf-8"))

    monkeypatch.setenv("VLLM_BASE_URL", "http://vllm.test/v1")
    monkeypatch.delenv("VLLM_TIMEOUT_SECONDS", raising=False)
    monkeypatch.setattr("medical_triage_agent.vllm_client.urlopen", fake_urlopen)

    result = generate_explanation(
        {"symptoms": ["douleur thoracique"]},
        assess_triage({"symptoms": ["douleur thoracique"]}),
    )

    assert result.explanation == explanation
    assert result.explanation_source == "llm"
    assert result.llm_status == "accepted"
    assert result.suggested_priority == "urgence_maximale"
    assert result.confidence == 0.9
    assert result.llm_response_preview is not None
    assert "suggested_priority" in result.llm_response_preview
    assert result.llm_response_truncated is False


def test_bad_llm_response_preview_is_available_and_redacted(monkeypatch: MonkeyPatch) -> None:
    payload = {"choices": [{"message": {"content": "patient@example.test not json"}}]}

    def fake_urlopen(_request: Any, timeout: float | None) -> _Response:
        return _Response(json.dumps(payload).encode("utf-8"))

    monkeypatch.setenv("VLLM_BASE_URL", "http://vllm.test/v1")
    monkeypatch.setattr("medical_triage_agent.vllm_client.urlopen", fake_urlopen)

    response = api.triage({"symptoms": ["fatigue"]})
    audit_record = api.audit(response["audit_id"])

    assert response["llm_status"] == "bad_response"
    assert audit_record is not None
    assert audit_record["llm_response_preview"] == "[REDACTED_EMAIL] not json"
    assert "patient@example.test" not in str(audit_record)


def test_generate_explanation_reports_timeout(monkeypatch: MonkeyPatch) -> None:
    def fake_urlopen(_request: Any, timeout: float | None) -> _Response:
        raise TimeoutError("slow model")

    monkeypatch.setenv("VLLM_BASE_URL", "http://vllm.test/v1")
    monkeypatch.setattr("medical_triage_agent.vllm_client.urlopen", fake_urlopen)

    result = generate_explanation(
        {"symptoms": ["fatigue"]},
        assess_triage({"symptoms": ["fatigue"]}),
    )

    assert result.explanation is None
    assert result.explanation_source == "fallback"
    assert result.llm_status == "timeout"


def test_generate_explanation_reports_connection_error(monkeypatch: MonkeyPatch) -> None:
    def fake_urlopen(_request: Any, timeout: float | None) -> _Response:
        raise OSError("connection refused")

    monkeypatch.setenv("VLLM_BASE_URL", "http://vllm.test/v1")
    monkeypatch.setattr("medical_triage_agent.vllm_client.urlopen", fake_urlopen)

    result = generate_explanation(
        {"symptoms": ["fatigue"]},
        assess_triage({"symptoms": ["fatigue"]}),
    )

    assert result.explanation is None
    assert result.explanation_source == "fallback"
    assert result.llm_status == "connection_error"

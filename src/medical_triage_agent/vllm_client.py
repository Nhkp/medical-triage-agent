from __future__ import annotations

import json
import os
import re
import unicodedata
from dataclasses import dataclass
from typing import Any
from urllib.error import URLError
from urllib.request import Request, urlopen

from medical_triage_agent.triage import TRIAGE_ORDER, TriageResponse

SYSTEM_PROMPT = (
    "You are a CHSA medical triage proof-of-concept assistant for clinical staff. "
    "Interpret the declared symptoms and suggest one triage priority. "
    "The backend keeps final authority and may override your suggestion for safety. "
    "Answer in the same language as the symptoms: French for French input, English otherwise. "
    "Use only Latin-script French or English. Do not use any other language or script. "
    "Do not provide a diagnosis, medication, dosage, or home-treatment instructions. "
    "Mention uncertainty and that human clinical review remains required. "
    "Use only these priority labels: urgence_maximale, moderee, differee. "
    "Return strict JSON only with keys suggested_priority, explanation, and confidence. "
    "confidence must be a number between 0 and 1. "
    "The explanation must be 2 to 4 short sentences. Do not use markdown or bullets."
)

_OPTIONAL_CONTEXT_KEYS = (
    "questionnaire",
    "questionnaire_state",
    "answers",
    "vitals",
    "antecedents",
    "age",
    "sex",
    "gender",
    "language",
)


@dataclass(frozen=True)
class TriageGenerationResult:
    explanation: str | None
    llm_status: str
    suggested_priority: str | None = None
    confidence: float | None = None

    @property
    def explanation_source(self) -> str:
        return "llm" if self.llm_status == "accepted" else "fallback"


ExplanationResult = TriageGenerationResult


def configured_model() -> str:
    return os.environ.get("VLLM_MODEL_ID", "rule_based_v1")


def is_configured() -> bool:
    return bool(os.environ.get("VLLM_BASE_URL"))


def generate_triage(payload: dict[str, Any], response: TriageResponse) -> TriageGenerationResult:
    base_url = os.environ.get("VLLM_BASE_URL")
    if not base_url:
        return TriageGenerationResult(explanation=None, llm_status="not_configured")

    request_payload = build_chat_request(payload, response)
    data = json.dumps(request_payload).encode("utf-8")
    request = Request(
        f"{base_url.rstrip('/')}/chat/completions",
        data=data,
        headers=_headers(),
        method="POST",
    )
    try:
        with urlopen(request, timeout=_request_timeout()) as raw:
            response_payload = json.loads(raw.read().decode("utf-8"))
    except TimeoutError:
        return TriageGenerationResult(explanation=None, llm_status="timeout")
    except (OSError, URLError):
        return TriageGenerationResult(explanation=None, llm_status="connection_error")
    except json.JSONDecodeError:
        return TriageGenerationResult(explanation=None, llm_status="bad_response")

    return extract_triage_generation(response_payload)


def generate_explanation(payload: dict[str, Any], response: TriageResponse) -> ExplanationResult:
    return generate_triage(payload, response)


def build_chat_request(payload: dict[str, Any], response: TriageResponse) -> dict[str, Any]:
    return {
        "model": configured_model(),
        "messages": [
            {
                "role": "system",
                "content": SYSTEM_PROMPT,
            },
            {
                "role": "user",
                "content": json.dumps(
                    _model_context(payload, response), ensure_ascii=False, sort_keys=True
                ),
            },
        ],
        "temperature": 0,
        "max_tokens": 160,
    }


def _model_context(payload: dict[str, Any], response: TriageResponse) -> dict[str, Any]:
    context: dict[str, Any] = {
        "symptoms": payload.get("symptoms", []),
        "rule_priority": response.priority,
        "draft_explanation": response.explanation,
    }
    for key in _OPTIONAL_CONTEXT_KEYS:
        if key in payload:
            context[key] = payload[key]
    return context


def _headers() -> dict[str, str]:
    headers = {"Content-Type": "application/json"}
    api_key = os.environ.get("API_KEY")
    if api_key:
        headers["Authorization"] = f"Bearer {api_key}"
    return headers


def _request_timeout() -> float | None:
    configured = os.environ.get("VLLM_TIMEOUT_SECONDS")
    if configured is None or configured.strip().casefold() in {"", "0", "none"}:
        return None
    return float(configured)


def extract_explanation(payload: dict[str, Any]) -> ExplanationResult:
    return extract_triage_generation(payload)


def extract_triage_generation(payload: dict[str, Any]) -> TriageGenerationResult:
    content = _raw_content(payload)
    if content is None:
        return TriageGenerationResult(explanation=None, llm_status="bad_response")
    try:
        data = json.loads(_strip_code_fence(content))
    except json.JSONDecodeError:
        return TriageGenerationResult(explanation=None, llm_status="bad_response")
    if not isinstance(data, dict):
        return TriageGenerationResult(explanation=None, llm_status="bad_response")

    suggested_priority = data.get("suggested_priority")
    explanation = data.get("explanation")
    confidence = data.get("confidence")
    if suggested_priority not in TRIAGE_ORDER or not isinstance(explanation, str):
        return TriageGenerationResult(explanation=None, llm_status="invalid_output")
    if not _valid_explanation(explanation):
        return TriageGenerationResult(explanation=None, llm_status="invalid_output")
    if not isinstance(confidence, int | float) or not 0 <= float(confidence) <= 1:
        return TriageGenerationResult(explanation=None, llm_status="invalid_output")
    return TriageGenerationResult(
        explanation=explanation.strip(),
        llm_status="accepted",
        suggested_priority=str(suggested_priority),
        confidence=float(confidence),
    )


def _extract_content(payload: dict[str, Any]) -> str | None:
    result = extract_explanation(payload)
    return result.explanation


def _raw_content(payload: dict[str, Any]) -> str | None:
    choices = payload.get("choices")
    if not isinstance(choices, list) or not choices:
        return None
    first = choices[0]
    if not isinstance(first, dict):
        return None
    message = first.get("message")
    if not isinstance(message, dict):
        return None
    content = message.get("content")
    if not isinstance(content, str):
        return None
    return content


def _strip_code_fence(content: str) -> str:
    stripped = content.strip()
    if not stripped.startswith("```"):
        return stripped
    lines = stripped.splitlines()
    if len(lines) >= 3 and lines[-1].strip() == "```":
        return "\n".join(lines[1:-1]).strip()
    return stripped


def _valid_explanation(explanation: str) -> bool:
    if not 20 <= len(explanation) <= 800:
        return False
    if re.search(r"(.)\1{7,}", explanation):
        return False
    if any(_is_unexpected_script(character) for character in explanation):
        return False
    return not _contains_forbidden_advice(explanation)


def _contains_forbidden_advice(explanation: str) -> bool:
    lowered = explanation.casefold()
    forbidden_fragments = (
        "diagnosis is",
        "diagnostic est",
        "diagnostique est",
        "take ",
        "prenez ",
        "donnez ",
        "administer ",
    )
    return any(fragment in lowered for fragment in forbidden_fragments) or bool(
        re.search(r"\b\d+\s*(mg|ml)\b", lowered)
    )


def _is_unexpected_script(character: str) -> bool:
    if (
        character.isascii()
        or character.isspace()
        or unicodedata.category(character).startswith("P")
    ):
        return False
    name = unicodedata.name(character, "")
    if "LATIN" in name:
        return False
    return unicodedata.category(character).startswith(("L", "M"))

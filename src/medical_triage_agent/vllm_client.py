from __future__ import annotations

import json
import os
import re
import unicodedata
from typing import Any
from urllib.error import URLError
from urllib.request import Request, urlopen

from medical_triage_agent.triage import TriageResponse

SYSTEM_PROMPT = (
    "You are a CHSA medical triage proof-of-concept assistant for clinical staff. "
    "Your role is only to explain the triage priority already provided by the backend; "
    "do not change, contradict, or replace that priority. "
    "Answer in the same language as the symptoms: French for French input, English otherwise. "
    "Use only Latin-script French or English. Do not use any other language or script. "
    "Do not provide a diagnosis, medication, dosage, or home-treatment instructions. "
    "Mention uncertainty and that human clinical review remains required. "
    "If the priority is urgence_maximale, explicitly mention immediate clinical review or escalation. "
    "Write 2 to 4 short sentences. Do not use markdown, bullets, or JSON."
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


def configured_model() -> str:
    return os.environ.get("VLLM_MODEL_ID", "rule_based_v1")


def is_configured() -> bool:
    return bool(os.environ.get("VLLM_BASE_URL"))


def generate_explanation(payload: dict[str, Any], response: TriageResponse) -> str | None:
    base_url = os.environ.get("VLLM_BASE_URL")
    if not base_url:
        return None

    request_payload = build_chat_request(payload, response)
    data = json.dumps(request_payload).encode("utf-8")
    request = Request(
        f"{base_url.rstrip('/')}/chat/completions",
        data=data,
        headers=_headers(),
        method="POST",
    )
    try:
        with urlopen(request, timeout=float(os.environ.get("VLLM_TIMEOUT_SECONDS", "10"))) as raw:
            response_payload = json.loads(raw.read().decode("utf-8"))
    except (OSError, URLError, TimeoutError, json.JSONDecodeError):
        return None

    return _extract_content(response_payload)


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
        "priority": response.priority,
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


def _extract_content(payload: dict[str, Any]) -> str | None:
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
    explanation = content.strip()
    return explanation if _valid_explanation(explanation) else None


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

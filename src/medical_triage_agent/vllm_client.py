from __future__ import annotations

import json
import os
import re
import unicodedata
from dataclasses import dataclass
from typing import Any
from urllib.error import URLError
from urllib.request import Request, urlopen

from medical_triage_agent.privacy import redact_pii
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
    "The first character of your answer must be { and the last character must be }. "
    'Example: {"suggested_priority":"moderee","explanation":"The declared symptoms require '
    'clinical review because uncertainty remains. A clinician must confirm the priority.",'
    '"confidence":0.5}. '
    "Do not output headings such as taxpipeline, Reponse, Réponse, Priority, Explanation, "
    "or Explication. Do not write markdown or prose outside JSON. "
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
    llm_response_preview: str | None = None
    llm_response_truncated: bool = False

    @property
    def explanation_source(self) -> str:
        return "llm" if self.llm_status in {"accepted", "accepted_repaired"} else "fallback"


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
            response_text = raw.read().decode("utf-8")
            response_payload = json.loads(response_text)
    except TimeoutError:
        return TriageGenerationResult(explanation=None, llm_status="timeout")
    except (OSError, URLError):
        return TriageGenerationResult(explanation=None, llm_status="connection_error")
    except json.JSONDecodeError:
        preview, truncated = _safe_preview(response_text)
        return TriageGenerationResult(
            explanation=None,
            llm_status="bad_response",
            llm_response_preview=preview,
            llm_response_truncated=truncated,
        )

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
        "stop": ["具有战士user", "具有战士assistant", "\nuser", "\nassistant"],
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
        preview, truncated = _safe_preview(json.dumps(payload, ensure_ascii=False, sort_keys=True))
        return TriageGenerationResult(
            explanation=None,
            llm_status="bad_response",
            llm_response_preview=preview,
            llm_response_truncated=truncated,
        )
    preview, truncated = _safe_preview(content)
    try:
        data = json.loads(_strip_code_fence(content))
    except json.JSONDecodeError:
        return _repair_non_json_generation(content, preview, truncated)
    if not isinstance(data, dict):
        return TriageGenerationResult(
            explanation=None,
            llm_status="bad_response",
            llm_response_preview=preview,
            llm_response_truncated=truncated,
        )

    suggested_priority = data.get("suggested_priority")
    explanation = data.get("explanation")
    confidence = data.get("confidence")
    if suggested_priority not in TRIAGE_ORDER or not isinstance(explanation, str):
        return TriageGenerationResult(
            explanation=None,
            llm_status="invalid_output",
            llm_response_preview=preview,
            llm_response_truncated=truncated,
        )
    if not _valid_explanation(explanation):
        return TriageGenerationResult(
            explanation=None,
            llm_status="invalid_output",
            llm_response_preview=preview,
            llm_response_truncated=truncated,
        )
    if not isinstance(confidence, int | float) or not 0 <= float(confidence) <= 1:
        return TriageGenerationResult(
            explanation=None,
            llm_status="invalid_output",
            llm_response_preview=preview,
            llm_response_truncated=truncated,
        )
    return TriageGenerationResult(
        explanation=explanation.strip(),
        llm_status="accepted",
        suggested_priority=str(suggested_priority),
        confidence=float(confidence),
        llm_response_preview=preview,
        llm_response_truncated=truncated,
    )


def _repair_non_json_generation(
    content: str, preview: str, truncated: bool
) -> TriageGenerationResult:
    first_block = _first_response_block(content)
    object_result = _repair_object_like_generation(first_block, preview, truncated)
    if object_result is not None:
        return object_result

    priority_match = re.search(
        r"(?im)^\s*(?:r[eé]ponse|priority)\s*:\s*"
        r"(urgence_maximale|moderee|differee)\b",
        first_block,
    )
    explanation_match = re.search(r"(?ims)^\s*(?:explanation|explication)\s*:\s*(.+)$", first_block)
    if priority_match is None or explanation_match is None:
        return TriageGenerationResult(
            explanation=None,
            llm_status="bad_response",
            llm_response_preview=preview,
            llm_response_truncated=truncated,
        )

    suggested_priority = priority_match.group(1)
    explanation = explanation_match.group(1).strip()
    if suggested_priority not in TRIAGE_ORDER or not _valid_explanation(explanation):
        return TriageGenerationResult(
            explanation=None,
            llm_status="invalid_output",
            llm_response_preview=preview,
            llm_response_truncated=truncated,
        )
    return TriageGenerationResult(
        explanation=explanation,
        llm_status="accepted_repaired",
        suggested_priority=suggested_priority,
        confidence=0.5,
        llm_response_preview=preview,
        llm_response_truncated=truncated,
    )


def _first_response_block(content: str) -> str:
    parts = re.split(
        r"具有战士\s*(?:user|assistant)\b|^\s*(?:user|assistant)\s*$",
        content,
        maxsplit=1,
        flags=re.IGNORECASE | re.MULTILINE,
    )
    return parts[0]


def _repair_object_like_generation(
    content: str, preview: str, truncated: bool
) -> TriageGenerationResult | None:
    candidate = _first_braced_object(content)
    if candidate is None:
        return None

    try:
        data = json.loads(_quote_unquoted_object_keys(candidate))
    except json.JSONDecodeError:
        return TriageGenerationResult(
            explanation=None,
            llm_status="bad_response",
            llm_response_preview=preview,
            llm_response_truncated=truncated,
        )
    if not isinstance(data, dict):
        return TriageGenerationResult(
            explanation=None,
            llm_status="bad_response",
            llm_response_preview=preview,
            llm_response_truncated=truncated,
        )

    lng = data.get("lng")
    suggested_priority = data.get("suggested_priority")
    if isinstance(lng, str):
        priority_match = re.search(
            r"\br[eé]ponse\s*:\s*(urgence_maximale|moderee|differee)\b",
            lng,
            flags=re.IGNORECASE,
        )
        if priority_match is not None:
            suggested_priority = priority_match.group(1)
    if not isinstance(explanation := data.get("explanation"), str):
        return TriageGenerationResult(
            explanation=None,
            llm_status="bad_response",
            llm_response_preview=preview,
            llm_response_truncated=truncated,
        )

    return _accepted_or_invalid_result(
        suggested_priority=suggested_priority,
        explanation=explanation,
        confidence=data.get("confidence"),
        preview=preview,
        truncated=truncated,
        repaired=True,
    )


def _first_braced_object(content: str) -> str | None:
    start = content.find("{")
    if start < 0:
        return None

    depth = 0
    in_string = False
    escaped = False
    for index, character in enumerate(content[start:], start=start):
        if in_string:
            if escaped:
                escaped = False
            elif character == "\\":
                escaped = True
            elif character == '"':
                in_string = False
            continue
        if character == '"':
            in_string = True
        elif character == "{":
            depth += 1
        elif character == "}":
            depth -= 1
            if depth == 0:
                return content[start : index + 1]
    return None


def _quote_unquoted_object_keys(candidate: str) -> str:
    return re.sub(r"([,{]\s*)([A-Za-z_][A-Za-z0-9_]*)\s*:", r'\1"\2":', candidate)


def _accepted_or_invalid_result(
    *,
    suggested_priority: object,
    explanation: object,
    confidence: object,
    preview: str,
    truncated: bool,
    repaired: bool,
) -> TriageGenerationResult:
    if suggested_priority not in TRIAGE_ORDER or not isinstance(explanation, str):
        return TriageGenerationResult(
            explanation=None,
            llm_status="invalid_output",
            llm_response_preview=preview,
            llm_response_truncated=truncated,
        )
    if not _valid_explanation(explanation):
        return TriageGenerationResult(
            explanation=None,
            llm_status="invalid_output",
            llm_response_preview=preview,
            llm_response_truncated=truncated,
        )
    if not isinstance(confidence, int | float) or not 0 <= float(confidence) <= 1:
        return TriageGenerationResult(
            explanation=None,
            llm_status="invalid_output",
            llm_response_preview=preview,
            llm_response_truncated=truncated,
        )
    return TriageGenerationResult(
        explanation=explanation.strip(),
        llm_status="accepted_repaired" if repaired else "accepted",
        suggested_priority=str(suggested_priority),
        confidence=float(confidence),
        llm_response_preview=preview,
        llm_response_truncated=truncated,
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


def _safe_preview(content: str, limit: int = 1200) -> tuple[str, bool]:
    redacted = redact_pii(content).strip()
    if len(redacted) <= limit:
        return redacted, False
    return redacted[:limit].rstrip(), True


def _valid_explanation(explanation: str) -> bool:
    if not 20 <= len(explanation) <= 800:
        return False
    if re.search(r"(.)\1{7,}", explanation):
        return False
    if _has_repeated_sentence(explanation):
        return False
    if any(_is_unexpected_script(character) for character in explanation):
        return False
    return not _contains_forbidden_advice(explanation)


def _has_repeated_sentence(explanation: str) -> bool:
    seen: set[str] = set()
    for sentence in re.split(r"[.!?]+", explanation):
        normalized = re.sub(r"\s+", " ", sentence.strip().casefold())
        if len(normalized) < 12:
            continue
        if normalized in seen:
            return True
        seen.add(normalized)
    return False


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
        "how to treat",
        "traitement par",
        "médicament",
        "medicament",
        "antihypertenseur",
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

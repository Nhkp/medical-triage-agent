from __future__ import annotations

import json
import os
import re
import unicodedata
from dataclasses import dataclass
from typing import Any, Literal
from urllib.error import HTTPError, URLError
from urllib.request import Request, urlopen

from medical_triage_agent.privacy import redact_pii
from medical_triage_agent.triage import TRIAGE_ORDER, TriageResponse

OUTPUT_CONTRACT = """Return exactly one JSON object and nothing else.

Schema:
{
  "suggested_priority": "urgence_maximale" | "moderee" | "differee",
  "explanation": "...",
  "confidence": 0.0-1.0
}

Explanation rules:
- Use the same language as the symptoms.
- Write exactly:
  - 3 sentences for urgence_maximale
  - 2 sentences for moderee
  - 2 sentences for differee
- Do not repeat any sentence.
- Do not introduce symptoms, diagnoses, diseases, treatments, hospitalization, or complications that were not provided.
- Explain why the declared symptoms support the suggested priority.
- Mention clinical review/human confirmation.
- No markdown, no headings, no text outside JSON."""

TRIAGE_JSON_SCHEMA: dict[str, Any] = {
    "type": "object",
    "properties": {
        "suggested_priority": {
            "type": "string",
            "enum": ["urgence_maximale", "moderee", "differee"],
        },
        "explanation": {"type": "string"},
        "confidence": {"type": "number", "minimum": 0, "maximum": 1},
    },
    "required": ["suggested_priority", "explanation", "confidence"],
    "additionalProperties": False,
}

SYSTEM_PROMPT = (
    "You are a CHSA medical triage proof-of-concept assistant for clinical staff. "
    "Interpret the declared symptoms and suggest one triage priority. "
    "The backend keeps final authority and may override your suggestion for safety. "
    "Answer in the same language as the symptoms: French for French input, English otherwise. "
    "Use only Latin-script French or English. Do not use any other language or script. "
    "Do not provide a diagnosis, medication, dosage, or home-treatment instructions. "
    "Mention uncertainty and that human clinical review remains required. "
    "Use only these priority labels: urgence_maximale, moderee, differee. "
    "Do not output headings such as taxpipeline, Reponse, Réponse, Priority, Explanation, "
    "or Explication. Do not write markdown or prose outside JSON. "
    "Do not mention v1 rules or copy fallback phrases such as Aucun symptome d'alerte. "
    + OUTPUT_CONTRACT
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
    request = _chat_request(base_url, request_payload)
    try:
        with urlopen(request, timeout=_request_timeout()) as raw:
            response_text = raw.read().decode("utf-8")
            response_payload = json.loads(response_text)
    except HTTPError as exc:
        if not _should_retry_with_legacy_guided_json(exc):
            return TriageGenerationResult(explanation=None, llm_status="connection_error")
        return _generate_triage_with_legacy_guided_json(base_url, payload, response)
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


def build_chat_request(
    payload: dict[str, Any],
    response: TriageResponse,
    *,
    structured_output: Literal["structured_outputs", "guided_json"] = "structured_outputs",
) -> dict[str, Any]:
    request: dict[str, Any] = {
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
        "max_tokens": 110,
        "stop": ["具有战士", "具有战士user", "具有战士assistant", "\nuser", "\nassistant"],
    }
    if structured_output == "guided_json":
        request["guided_json"] = TRIAGE_JSON_SCHEMA
    else:
        request["structured_outputs"] = {"json": TRIAGE_JSON_SCHEMA}
    return request


def _chat_request(base_url: str, payload: dict[str, Any]) -> Request:
    data = json.dumps(payload).encode("utf-8")
    return Request(
        f"{base_url.rstrip('/')}/chat/completions",
        data=data,
        headers=_headers(),
        method="POST",
    )


def _should_retry_with_legacy_guided_json(exc: HTTPError) -> bool:
    return exc.code in {400, 404, 422}


def _generate_triage_with_legacy_guided_json(
    base_url: str, payload: dict[str, Any], response: TriageResponse
) -> TriageGenerationResult:
    request_payload = build_chat_request(payload, response, structured_output="guided_json")
    try:
        with urlopen(_chat_request(base_url, request_payload), timeout=_request_timeout()) as raw:
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


def _model_context(payload: dict[str, Any], response: TriageResponse) -> dict[str, Any]:
    context: dict[str, Any] = {
        "symptoms": payload.get("symptoms", []),
        "rule_priority": response.priority,
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
    return _accepted_or_invalid_result(
        suggested_priority=suggested_priority,
        explanation=explanation,
        confidence=confidence,
        preview=preview,
        truncated=truncated,
        repaired=False,
    )


def _repair_non_json_generation(
    content: str, preview: str, truncated: bool
) -> TriageGenerationResult:
    object_result = _repair_object_like_generation(content, preview, truncated)
    if object_result is not None:
        return object_result

    first_block = _first_response_block(content)
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
    return _accepted_or_invalid_result(
        suggested_priority=suggested_priority,
        explanation=explanation,
        confidence=0.5,
        preview=preview,
        truncated=truncated,
        repaired=True,
    )


def _first_response_block(content: str) -> str:
    parts = re.split(
        r"具有战士|^\s*(?:user|assistant)\s*$",
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
        if lng in TRIAGE_ORDER:
            suggested_priority = lng
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
    explanation, deduplicated = _deduplicate_repeated_sentences(explanation)
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
        llm_status="accepted_repaired" if repaired or deduplicated else "accepted",
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
    if len(_sentences(explanation)) < 2:
        return False
    if re.search(r"(.)\1{7,}", explanation):
        return False
    if _has_repeated_sentence(explanation):
        return False
    if _contains_fallback_template(explanation):
        return False
    if any(_is_unexpected_script(character) for character in explanation):
        return False
    return not _contains_forbidden_advice(explanation)


def _has_repeated_sentence(explanation: str) -> bool:
    seen: set[str] = set()
    for sentence in _sentences(explanation):
        normalized = _normalized_sentence(sentence)
        if not normalized:
            continue
        if normalized in seen:
            return True
        seen.add(normalized)
    return False


def _deduplicate_repeated_sentences(explanation: str) -> tuple[str, bool]:
    sentences = _sentences(explanation)
    if not sentences:
        return explanation, False
    sentences, block_deduplicated = _deduplicate_repeated_sentence_blocks(sentences)
    kept: list[str] = []
    seen: set[str] = set()
    deduplicated = False
    for sentence in sentences:
        normalized = _normalized_sentence(sentence)
        if not normalized:
            kept.append(sentence)
            continue
        if normalized in seen:
            deduplicated = True
            continue
        seen.add(normalized)
        kept.append(sentence)
    if not deduplicated:
        return (" ".join(sentences), True) if block_deduplicated else (explanation, False)
    return " ".join(kept), True


def _deduplicate_repeated_sentence_blocks(sentences: list[str]) -> tuple[list[str], bool]:
    normalized = [_normalized_sentence(sentence) for sentence in sentences]
    kept: list[str] = []
    changed = False
    index = 0
    while index < len(sentences):
        block_size = _repeated_block_size(normalized, index)
        if block_size is None:
            kept.append(sentences[index])
            index += 1
            continue
        kept.extend(sentences[index : index + block_size])
        index += block_size
        while normalized[index : index + block_size] == normalized[index - block_size : index]:
            changed = True
            index += block_size
    return kept, changed


def _repeated_block_size(normalized: list[str], index: int) -> int | None:
    remaining = len(normalized) - index
    for block_size in range(1, remaining // 2 + 1):
        block = normalized[index : index + block_size]
        if not all(block):
            continue
        next_block = normalized[index + block_size : index + (block_size * 2)]
        if block == next_block:
            return block_size
    return None


def _sentences(explanation: str) -> list[str]:
    return [
        match.group(0).strip()
        for match in re.finditer(r"[^.!?]+[.!?]+|[^.!?]+$", explanation)
        if match.group(0).strip()
    ]


def _normalized_sentence(sentence: str) -> str:
    stripped = re.sub(r"\s+", " ", sentence.strip().casefold())
    return stripped if len(stripped) >= 12 else ""


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
        "hospitalization",
        "hospitalisation",
        "no hospitalization",
        "pas d'hospitalisation",
        "simple care",
        "does not require urgent treatment",
        "ne nécessite pas de traitement urgent",
        "ne necessite pas de traitement urgent",
    )
    return any(fragment in lowered for fragment in forbidden_fragments) or bool(
        re.search(r"\b\d+\s*(mg|ml)\b", lowered)
    )


def _contains_fallback_template(explanation: str) -> bool:
    lowered = explanation.casefold()
    fallback_fragments = (
        "aucun symptome d'alerte v1 detecte",
        "aucun symptôme d'alerte v1 détecté",
        "symptomes d'alerte detectes",
        "symptômes d'alerte détectés",
        "revue clinique necessaire pour confirmer",
        "revue clinique nécessaire pour confirmer",
    )
    return any(fragment in lowered for fragment in fallback_fragments)


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

from __future__ import annotations

import json
import os
from typing import Any
from urllib.error import URLError
from urllib.request import Request, urlopen

from medical_triage_agent.triage import TriageResponse


def configured_model() -> str:
    return os.environ.get("VLLM_MODEL_ID", "rule_based_v1")


def is_configured() -> bool:
    return bool(os.environ.get("VLLM_BASE_URL"))


def generate_explanation(payload: dict[str, Any], response: TriageResponse) -> str | None:
    base_url = os.environ.get("VLLM_BASE_URL")
    if not base_url:
        return None

    request_payload = {
        "model": configured_model(),
        "messages": [
            {
                "role": "system",
                "content": (
                    "You are a medical triage POC assistant for clinicians. Explain the "
                    "triage priority conservatively and include human review."
                ),
            },
            {
                "role": "user",
                "content": json.dumps(
                    {"request": payload, "priority": response.priority},
                    ensure_ascii=False,
                    sort_keys=True,
                ),
            },
        ],
        "temperature": 0,
        "max_tokens": 160,
    }
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
    return content if isinstance(content, str) and content.strip() else None

from __future__ import annotations

import json
import os
from dataclasses import dataclass
from importlib import import_module
from typing import Any
from urllib.error import HTTPError, URLError
from urllib.request import Request, urlopen

DEFAULT_API_URL = "http://127.0.0.1:8080"
MODAL_API_URL_ENV = "MODAL_API_URL"
TRIAGE_API_TOKEN_ENV = "TRIAGE_API_TOKEN"


@dataclass(frozen=True)
class ApiResult:
    """HTTP result shape used by the optional Streamlit console."""

    ok: bool
    data: dict[str, Any] | None = None
    error: str | None = None
    status_code: int | None = None


def parse_symptoms(text: str) -> list[str]:
    """Convert newline-separated UI input into symptom strings."""

    return [line.strip() for line in text.splitlines() if line.strip()]


def endpoint_url(base_url: str, path: str) -> str:
    """Join an API base URL and path without duplicate slashes."""

    return f"{base_url.rstrip('/')}/{path.lstrip('/')}"


def default_api_url() -> str:
    """Return the configured Modal API URL, or localhost for local development."""

    return os.environ.get(MODAL_API_URL_ENV, "").strip() or DEFAULT_API_URL


def auth_headers(token: str | None) -> dict[str, str]:
    """Build optional bearer auth headers for Modal-protected API endpoints."""

    return {"Authorization": f"Bearer {token.strip()}"} if token and token.strip() else {}


def request_json(
    method: str,
    base_url: str,
    path: str,
    payload: dict[str, Any] | None = None,
    timeout: float | None = None,
    bearer_token: str | None = None,
) -> ApiResult:
    """Send a JSON request to the triage API and normalize success/error results."""

    body = json.dumps(payload).encode("utf-8") if payload is not None else None
    request = Request(
        endpoint_url(base_url, path),
        data=body,
        method=method,
        headers={"Content-Type": "application/json", **auth_headers(bearer_token)},
    )
    try:
        with urlopen(request, timeout=timeout) as response:
            return ApiResult(
                ok=True,
                data=json.loads(response.read().decode("utf-8")),
                status_code=response.status,
            )
    except HTTPError as exc:
        return ApiResult(ok=False, error=_http_error_message(exc), status_code=exc.code)
    except (TimeoutError, URLError) as exc:
        return ApiResult(ok=False, error=str(exc))


def _http_error_message(exc: HTTPError) -> str:
    """Extract a readable FastAPI error detail from an HTTPError."""

    try:
        payload = json.loads(exc.read().decode("utf-8"))
    except (json.JSONDecodeError, UnicodeDecodeError):
        return exc.reason
    detail = payload.get("detail")
    return str(detail if detail is not None else payload)


def run_app() -> None:
    """Render the Streamlit triage API tester."""

    st: Any = import_module("streamlit")

    st.set_page_config(page_title="CHSA triage API tester", layout="centered")
    st.title("CHSA triage API tester")
    st.caption("Optional Streamlit console for the FastAPI/vLLM proof of concept.")

    api_url = st.sidebar.text_input("API base URL", value=default_api_url())
    bearer_token = os.environ.get(TRIAGE_API_TOKEN_ENV, "")
    st.sidebar.caption(
        "Use localhost for local FastAPI, or your Modal API URL. No client timeout is applied."
    )

    if st.sidebar.button("Check API health"):
        _render_health(request_json("GET", api_url, "/health"))

    symptoms_text = st.text_area(
        "Symptoms",
        value="douleur thoracique\ndifficulte respiratoire",
        help="One symptom per line. Do not enter identifiable patient data.",
    )
    symptoms = parse_symptoms(symptoms_text)

    if st.button("Analyze triage", type="primary"):
        if not symptoms:
            st.warning("Add at least one symptom.")
        else:
            _render_triage(
                request_json(
                    "POST",
                    api_url,
                    "/triage",
                    {"symptoms": symptoms},
                    bearer_token=bearer_token,
                )
            )

    st.divider()
    audit_id = st.text_input("Audit ID", value=st.session_state.get("last_audit_id", ""))
    if st.button("View audit metadata") and audit_id:
        _render_audit(request_json("GET", api_url, f"/audit/{audit_id}", bearer_token=bearer_token))


def _render_health(result: ApiResult) -> None:
    """Render health-check results in the sidebar."""

    st: Any = import_module("streamlit")
    if result.ok and result.data is not None:
        st.sidebar.success(f"API: {result.data.get('status', 'unknown')}")
        st.sidebar.write(result.data)
    else:
        st.sidebar.error(result.error or "Health check failed.")


def _render_triage(result: ApiResult) -> None:
    """Render triage output while preserving escalation visibility."""

    st: Any = import_module("streamlit")
    if not result.ok or result.data is None:
        st.error(result.error or "Triage request failed.")
        return

    priority = result.data.get("priority", "unknown")
    if priority == "urgence_maximale":
        st.error(f"Urgent escalation: {priority}")
    else:
        st.info(f"Priority: {priority}")
    rule_priority = result.data.get("rule_priority", "unknown")
    llm_priority = result.data.get("llm_priority") or "none"
    llm_confidence = result.data.get("llm_confidence") or "none"
    priority_source = result.data.get("priority_source", "unknown")
    arbitration = result.data.get("arbitration", "unknown")
    st.write(
        {
            "rule_priority": rule_priority,
            "llm_priority": llm_priority,
            "llm_confidence": llm_confidence,
            "priority_source": priority_source,
            "arbitration": arbitration,
        }
    )
    if arbitration == "rule_escalated":
        st.warning("Backend safety rules overrode a lower LLM priority suggestion.")
    source = result.data.get("explanation_source", "unknown")
    llm_status = result.data.get("llm_status", "unknown")
    if source == "llm":
        st.success(f"Explanation source: {source} ({llm_status})")
    else:
        st.warning(f"Explanation source: {source} (LLM status: {llm_status})")
    st.write(result.data.get("explanation", ""))
    st.caption(result.data.get("disclaimer", ""))

    audit_id = str(result.data.get("audit_id", ""))
    if audit_id:
        st.session_state["last_audit_id"] = audit_id
        st.code(audit_id)


def _render_audit(result: ApiResult) -> None:
    """Render redacted audit metadata or the lookup error."""

    st: Any = import_module("streamlit")
    if result.ok and result.data is not None:
        st.json(result.data)
    else:
        st.error(result.error or "Audit lookup failed.")


if __name__ == "__main__":
    run_app()

from __future__ import annotations

import json
from dataclasses import dataclass
from importlib import import_module
from typing import Any
from urllib.error import HTTPError, URLError
from urllib.request import Request, urlopen

DEFAULT_API_URL = "http://127.0.0.1:8080"


@dataclass(frozen=True)
class ApiResult:
    ok: bool
    data: dict[str, Any] | None = None
    error: str | None = None
    status_code: int | None = None


def parse_symptoms(text: str) -> list[str]:
    return [line.strip() for line in text.splitlines() if line.strip()]


def endpoint_url(base_url: str, path: str) -> str:
    return f"{base_url.rstrip('/')}/{path.lstrip('/')}"


def request_json(
    method: str,
    base_url: str,
    path: str,
    payload: dict[str, Any] | None = None,
    timeout: float | None = None,
) -> ApiResult:
    body = json.dumps(payload).encode("utf-8") if payload is not None else None
    request = Request(
        endpoint_url(base_url, path),
        data=body,
        method=method,
        headers={"Content-Type": "application/json"},
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
    try:
        payload = json.loads(exc.read().decode("utf-8"))
    except (json.JSONDecodeError, UnicodeDecodeError):
        return exc.reason
    detail = payload.get("detail")
    return str(detail if detail is not None else payload)


def run_app() -> None:
    st: Any = import_module("streamlit")

    st.set_page_config(page_title="CHSA triage API tester", layout="centered")
    st.title("CHSA triage API tester")
    st.caption("Optional Streamlit console for the FastAPI/vLLM proof of concept.")

    api_url = st.sidebar.text_input("API base URL", value=DEFAULT_API_URL)
    st.sidebar.caption(
        "Use localhost for local FastAPI, or your Kaggle/ngrok URL. No client timeout is applied."
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
            _render_triage(request_json("POST", api_url, "/triage", {"symptoms": symptoms}))

    st.divider()
    audit_id = st.text_input("Audit ID", value=st.session_state.get("last_audit_id", ""))
    if st.button("View audit metadata") and audit_id:
        _render_audit(request_json("GET", api_url, f"/audit/{audit_id}"))


def _render_health(result: ApiResult) -> None:
    st: Any = import_module("streamlit")
    if result.ok and result.data is not None:
        st.sidebar.success(f"API: {result.data.get('status', 'unknown')}")
        st.sidebar.write(result.data)
    else:
        st.sidebar.error(result.error or "Health check failed.")


def _render_triage(result: ApiResult) -> None:
    st: Any = import_module("streamlit")
    if not result.ok or result.data is None:
        st.error(result.error or "Triage request failed.")
        return

    priority = result.data.get("priority", "unknown")
    if priority == "urgence_maximale":
        st.error(f"Urgent escalation: {priority}")
    else:
        st.info(f"Priority: {priority}")
    st.write(result.data.get("explanation", ""))
    st.caption(result.data.get("disclaimer", ""))

    audit_id = str(result.data.get("audit_id", ""))
    if audit_id:
        st.session_state["last_audit_id"] = audit_id
        st.code(audit_id)


def _render_audit(result: ApiResult) -> None:
    st: Any = import_module("streamlit")
    if result.ok and result.data is not None:
        st.json(result.data)
    else:
        st.error(result.error or "Audit lookup failed.")


if __name__ == "__main__":
    run_app()

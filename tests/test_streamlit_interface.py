from __future__ import annotations

import io
from email.message import Message
from typing import Any, Self
from urllib.error import HTTPError, URLError

from medical_triage_agent import streamlit_interface as ui


class _Response:
    status = 200

    def __init__(self, payload: bytes) -> None:
        self._payload = payload

    def __enter__(self) -> Self:
        return self

    def __exit__(self, *_args: object) -> None:
        return None

    def read(self) -> bytes:
        return self._payload


def test_parse_symptoms_trims_blank_lines() -> None:
    assert ui.parse_symptoms(" douleur \n\n toux \n") == ["douleur", "toux"]


def test_endpoint_url_handles_trailing_slashes() -> None:
    assert ui.endpoint_url("https://example.test/", "/triage") == "https://example.test/triage"


def test_request_json_returns_decoded_payload(monkeypatch: Any) -> None:
    def fake_urlopen(_request: Any, timeout: float) -> _Response:
        assert timeout == 15.0
        return _Response(b'{"status": "ok"}')

    monkeypatch.setattr(ui, "urlopen", fake_urlopen)

    assert ui.request_json("GET", "http://api.test", "/health").data == {"status": "ok"}


def test_request_json_reports_http_error(monkeypatch: Any) -> None:
    def fake_urlopen(_request: Any, timeout: float) -> _Response:
        raise HTTPError(
            "http://api.test/triage",
            400,
            "Bad Request",
            Message(),
            io.BytesIO(b'{"detail": "symptoms required"}'),
        )

    monkeypatch.setattr(ui, "urlopen", fake_urlopen)

    result = ui.request_json("POST", "http://api.test", "/triage", {"symptoms": []})

    assert result.ok is False
    assert result.status_code == 400
    assert result.error == "symptoms required"


def test_request_json_reports_connection_error(monkeypatch: Any) -> None:
    def fake_urlopen(_request: Any, timeout: float) -> _Response:
        raise URLError("offline")

    monkeypatch.setattr(ui, "urlopen", fake_urlopen)

    result = ui.request_json("GET", "http://api.test", "/health")

    assert result.ok is False
    assert "offline" in str(result.error)

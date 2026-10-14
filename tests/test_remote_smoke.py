from __future__ import annotations

import importlib
from email.message import Message
from types import SimpleNamespace
from typing import Any, Self
from urllib.error import HTTPError, URLError

import pytest


@pytest.mark.parametrize(
    "error",
    [
        HTTPError("https://api.test", 401, "Unauthorized", Message(), None),
        URLError("offline"),
        ValueError("invalid JSON"),
    ],
)
def test_remote_validation_case_does_not_hide_failures(monkeypatch: Any, error: Exception) -> None:
    module = importlib.import_module("scripts.evaluate_robustness")
    monkeypatch.setattr(
        module, "CASES", ({"name": "invalid input", "payload": {}, "expect_error": True},)
    )

    def fail(*_args: Any) -> Any:
        raise error

    monkeypatch.setattr(module, "_call_triage", fail)
    assert not module.run_robustness_eval(url="https://api.test")["passed"]


def test_missing_remote_audit_fails(monkeypatch: Any) -> None:
    module = importlib.import_module("scripts.evaluate_robustness")
    monkeypatch.setattr(
        module,
        "_call_triage",
        lambda *_args: {"priority": "moderee", "audit_id": "id", "disclaimer": "POC"},
    )
    monkeypatch.setattr(module, "_call_audit", lambda *_args: None)
    assert not module.run_robustness_eval(url="https://api.test")["passed"]


def test_remote_evaluators_authenticate_triage_and_audit(monkeypatch: Any) -> None:
    monkeypatch.setenv("TRIAGE_API_TOKEN", "synthetic")
    requests = []

    class Response:
        def __enter__(self) -> Self:
            return self

        def __exit__(self, *_args: object) -> None:
            pass

        def read(self) -> bytes:
            return b'{"audit_id":"id"}'

    def open_request(request: Any, **_kwargs: Any) -> Response:
        assert request.get_header("Authorization") == "Bearer synthetic"
        requests.append(request.full_url)
        return Response()

    for name in ("evaluate_latency", "evaluate_robustness"):
        module = importlib.import_module("scripts." + name)
        monkeypatch.setattr(
            module, "build_opener", lambda *_args: SimpleNamespace(open=open_request)
        )
        if name == "evaluate_latency":
            module._call_remote("https://api.test", {"symptoms": ["fatigue"]})
        else:
            module._call_triage("https://api.test", {"symptoms": ["fatigue"]})
            module._call_audit("https://api.test", "id")
    assert len(requests) == 3

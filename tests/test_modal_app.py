from __future__ import annotations

import importlib

from fastapi.testclient import TestClient
from pytest import MonkeyPatch


def test_modal_vllm_command_serves_default_dpo_adapter() -> None:
    module = importlib.import_module("deploy.modal_app")

    command = module.build_vllm_command()

    assert command[:3] == ["vllm", "serve", "Qwen/Qwen3-1.7B-Base"]
    assert "--enable-lora" in command
    assert "--lora-modules" in command
    assert "medical-triage-dpo-8k=Lokhidor/medical-triage-qwen3-dpo-lora-8k" in command
    assert command[command.index("--served-model-name") + 1] == "medical-triage-dpo-8k"
    assert command[command.index("--host") + 1] == "127.0.0.1"
    assert command[command.index("--port") + 1] == "8000"
    assert command[command.index("--max-model-len") + 1] == "4096"
    assert command[command.index("--gpu-memory-utilization") + 1] == "0.70"


def test_modal_runtime_env_points_fastapi_to_local_vllm() -> None:
    module = importlib.import_module("deploy.modal_app")

    assert module.build_runtime_env() == {
        "PYTHONPATH": "/app/src",
        "VLLM_BASE_URL": "http://127.0.0.1:8000/v1",
        "VLLM_MODEL_ID": "medical-triage-dpo-8k",
    }


def test_modal_health_endpoint_is_public(monkeypatch: MonkeyPatch) -> None:
    module = importlib.import_module("deploy.modal_app")
    monkeypatch.delenv("TRIAGE_API_TOKEN", raising=False)

    response = TestClient(module.create_authenticated_api_app()).get("/health")

    assert response.status_code == 200


def test_modal_triage_requires_bearer_token(monkeypatch: MonkeyPatch) -> None:
    module = importlib.import_module("deploy.modal_app")
    monkeypatch.setenv("TRIAGE_API_TOKEN", "secret-token")
    client = TestClient(module.create_authenticated_api_app())

    missing = client.post("/triage", json={"symptoms": ["douleur thoracique"]})
    wrong = client.post(
        "/triage",
        headers={"Authorization": "Bearer wrong"},
        json={"symptoms": ["douleur thoracique"]},
    )

    assert missing.status_code == 401
    assert wrong.status_code == 401


def test_modal_triage_accepts_correct_bearer_token(monkeypatch: MonkeyPatch) -> None:
    module = importlib.import_module("deploy.modal_app")
    monkeypatch.setenv("TRIAGE_API_TOKEN", "secret-token")

    response = TestClient(module.create_authenticated_api_app()).post(
        "/triage",
        headers={"Authorization": "Bearer secret-token"},
        json={"symptoms": ["douleur thoracique"]},
    )

    assert response.status_code == 200
    assert response.json()["priority"] == "urgence_maximale"

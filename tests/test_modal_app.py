from __future__ import annotations

import asyncio
import importlib
import json
import os
from typing import Any

from fastapi.testclient import TestClient
from pytest import MonkeyPatch


def test_modal_vllm_command_serves_default_dpo_adapter() -> None:
    module = importlib.import_module("deploy.modal_app")

    command = module.build_vllm_command()

    assert command[:3] == ["vllm", "serve", "Qwen/Qwen3-1.7B-Base"]
    assert command[command.index("--revision") + 1] == ("ea980cb0a6c2ae4b936e82123acc929f1cec04c1")
    assert "--enable-lora" in command
    assert "--lora-modules" in command
    lora_module = json.loads(command[command.index("--lora-modules") + 1])
    assert lora_module == {
        "name": "medical-triage-dpo-8k",
        "path": "Lokhidor/medical-triage-qwen3-dpo-lora-8k",
        "base_model_name": "Qwen/Qwen3-1.7B-Base",
    }
    assert command[command.index("--served-model-name") + 1] == "medical-triage-dpo-8k"
    assert command[command.index("--host") + 1] == "127.0.0.1"
    assert command[command.index("--port") + 1] == "8000"
    assert command[command.index("--max-model-len") + 1] == "4096"
    assert command[command.index("--gpu-memory-utilization") + 1] == "0.70"
    assert "--enforce-eager" in command


def test_modal_image_pins_transformers_for_vllm_tokenizer_compatibility() -> None:
    module = importlib.import_module("deploy.modal_app")

    assert "vllm==0.10.2" in module.IMAGE_PACKAGES
    assert "transformers==4.56.2" in module.IMAGE_PACKAGES


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


def test_modal_startup_configures_both_fastapi_and_vllm(monkeypatch: MonkeyPatch) -> None:
    module = importlib.import_module("deploy.modal_app")
    for key in module.build_runtime_env():
        monkeypatch.setenv(key, "")
    started = []
    stopped: list[object] = []
    process = object()

    def start(command: list[str]) -> Any:
        started.append(command)
        assert {
            key: os.environ[key] for key in module.build_runtime_env()
        } == module.build_runtime_env()
        return process

    monkeypatch.setattr(module.subprocess, "Popen", start)
    monkeypatch.setattr(module, "wait_for_vllm", lambda: None)
    monkeypatch.setattr(module, "terminate_process", stopped.append)

    async def check() -> None:
        app = module.create_authenticated_api_app()
        async with module.lifespan(app):
            response = TestClient(app).get("/health")
            assert response.json() == {
                "status": "ok",
                "model": "medical-triage-dpo-8k",
                "vllm": "configured",
            }

    asyncio.run(check())
    assert started == [module.build_vllm_command()]
    assert stopped == [process]

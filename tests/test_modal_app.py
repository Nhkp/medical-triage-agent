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

    assert command[:3] == ["vllm", "serve", "/app/assets/base"]
    assert command[command.index("--revision") + 1] == ("ea980cb0a6c2ae4b936e82123acc929f1cec04c1")
    assert "--enable-lora" in command
    assert "--lora-modules" in command
    lora_module = json.loads(command[command.index("--lora-modules") + 1])
    assert lora_module == {
        "name": "medical-triage-dpo-8k",
        "path": "/app/assets/dpo",
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
        "VLLM_STRUCTURED_OUTPUT": "guided_json",
        "VLLM_MAX_TOKENS": "256",
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

    monkeypatch.setattr(module, "prepare_serving_assets", dict)
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


def test_modal_preparation_pins_snapshots_and_rejects_changed_assets(
    monkeypatch: MonkeyPatch, tmp_path: Any
) -> None:
    import sys
    from types import SimpleNamespace

    from medical_triage_agent import tokenizer_compatibility as compatibility

    module = importlib.import_module("deploy.modal_app")
    base = tmp_path / "base"
    base.mkdir()
    (base / "config.json").write_text("{}")
    sources = {}
    for name in ("sft", "dpo"):
        source = tmp_path / name
        source.mkdir()
        (source / "chat_template.jinja").write_text("synthetic template")
        sources[name] = source
    config = {
        **module.SERVING_CONFIG,
        "base_hashes": {"config.json": compatibility.checksum(base / "config.json")},
        "chat_template_checksum": compatibility.checksum(sources["sft"] / "chat_template.jinja"),
        "models": {
            name: {
                **values,
                "original_hashes": {
                    "chat_template.jinja": compatibility.checksum(
                        sources[name] / "chat_template.jinja"
                    )
                },
            }
            for name, values in module.SERVING_CONFIG["models"].items()
        },
    }
    downloaded = []

    def snapshot(repo: str, **kwargs: Any) -> str:
        downloaded.append((repo, kwargs["revision"]))
        return str(
            base
            if repo == module.BASE_MODEL
            else sources["sft" if repo.endswith("sft-lora-8k") else "dpo"]
        )

    monkeypatch.setitem(sys.modules, "huggingface_hub", SimpleNamespace(snapshot_download=snapshot))

    def prepare(source: Any, destination: Any, cases: Any, template: str) -> dict[str, Any]:
        assert len(cases) == 48 and template == "synthetic template"
        destination.mkdir()
        return {
            "equivalence_verified": True,
            "prompt_encoding_checksum": config["models"][source.name]["prompt_encoding_checksum"],
        }

    monkeypatch.setattr(compatibility, "prepare_adapter_compatibility", prepare)
    monkeypatch.setattr(module, "SERVING_CONFIG", config)
    monkeypatch.setattr(
        module.metadata,
        "version",
        lambda name: {"vllm": "0.10.2", "transformers": "4.56.2"}.get(name, "test"),
    )
    paths = module.prepare_serving_assets(tmp_path / "ready")
    assert paths["base_model"] == str(base)
    assert paths["adapter_repo"] == str(tmp_path / "ready/dpo")
    assert downloaded == [(module.BASE_MODEL, module.BASE_MODEL_REVISION)] + [
        (c["repo"], c["revision"]) for c in config["models"].values()
    ]
    assert json.loads((tmp_path / "ready/serving_manifest.json").read_text())["tokenizers"]["dpo"][
        "equivalence_verified"
    ]
    (base / "config.json").write_text("changed")
    import pytest

    with pytest.raises(ValueError, match="Base snapshot asset differs"):
        module.prepare_serving_assets(tmp_path / "different")

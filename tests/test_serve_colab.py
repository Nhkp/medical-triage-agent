from __future__ import annotations

import argparse
import importlib.util
import sys
from pathlib import Path
from typing import Any


def test_serve_colab_dry_run_commands_are_colab_friendly() -> None:
    module = _load_script("serve_colab", Path("scripts/serve_colab.py"))
    args = argparse.Namespace(
        model=None,
        base_model="Qwen/Qwen3-1.7B-Base",
        adapter="Lokhidor/chsa-qwen3-dpo-lora",
        lora_name="medical-triage-dpo",
        host="127.0.0.1",
        vllm_port=8000,
        api_port=8080,
    )

    commands = module.build_commands(args, Path("/content/medical-triage-agent"))

    assert commands["vllm"][:3] == [sys.executable, "-m", "vllm.entrypoints.openai.api_server"]
    assert "--model" in commands["vllm"]
    assert "Qwen/Qwen3-1.7B-Base" in commands["vllm"]
    assert "--enable-lora" in commands["vllm"]
    assert "--lora-modules" in commands["vllm"]
    assert "medical-triage-dpo=Lokhidor/chsa-qwen3-dpo-lora" in commands["vllm"]
    assert (
        "Lokhidor/chsa-qwen3-dpo-lora" not in commands["vllm"][: commands["vllm"].index("--host")]
    )
    assert "uvicorn" in commands["api"]
    assert "medical_triage_agent.api:create_app" in commands["api"]


def _load_script(name: str, path: Path) -> Any:
    spec = importlib.util.spec_from_file_location(name, path)
    assert spec is not None
    assert spec.loader is not None
    module = importlib.util.module_from_spec(spec)
    spec.loader.exec_module(module)
    return module

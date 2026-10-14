from __future__ import annotations

import argparse
import importlib.util
import json
import sys
from pathlib import Path
from typing import Any

from medical_triage_agent.tokenizer_compatibility import checksum


def test_serve_notebook_dry_run_uses_qualified_template_and_fp16() -> None:
    module = _load_script("serve_notebook", Path("scripts/serve_notebook.py"))
    args = argparse.Namespace(
        model=None,
        base_model="Qwen/Qwen3-1.7B-Base",
        adapter="Lokhidor/chsa-qwen3-dpo-lora",
        lora_name="medical-triage-dpo",
        host="127.0.0.1",
        vllm_port=8000,
        api_port=8080,
    )

    assert (
        checksum(Path("configs/chat_template.jinja"))
        == json.loads(Path("configs/serving.json").read_text())["chat_template_checksum"]
    )

    commands = module.build_commands(args, Path("/kaggle/working/medical-triage-agent"))

    assert commands["vllm"][:3] == [sys.executable, "-m", "vllm.entrypoints.openai.api_server"]
    assert commands["vllm"][commands["vllm"].index("--dtype") + 1] == "half"
    assert commands["vllm"][commands["vllm"].index("--chat-template") + 1].endswith(
        "configs/chat_template.jinja"
    )
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

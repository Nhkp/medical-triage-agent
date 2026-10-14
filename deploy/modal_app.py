from __future__ import annotations

import hmac
import json
import os
import subprocess
import time
from collections.abc import AsyncIterator
from contextlib import asynccontextmanager
from importlib import metadata
from pathlib import Path
from types import SimpleNamespace
from typing import Any
from urllib.error import URLError
from urllib.request import urlopen

import modal
from starlette.middleware.base import BaseHTTPMiddleware, RequestResponseEndpoint
from starlette.requests import Request
from starlette.responses import JSONResponse, Response

APP_NAME = "medical-triage-agent"
CONFIG_PATH = Path("/app/configs/serving.json")
if not CONFIG_PATH.exists():
    CONFIG_PATH = Path(__file__).resolve().parents[1] / "configs/serving.json"
SERVING_CONFIG = json.loads(CONFIG_PATH.read_text())
BASE_MODEL = SERVING_CONFIG["base_model"]
BASE_MODEL_REVISION = SERVING_CONFIG["base_revision"]
SERVED_MODEL_NAME = "medical-triage-dpo-8k"
VLLM_HOST = "127.0.0.1"
VLLM_PORT = 8000
VLLM_STARTUP_TIMEOUT_SECONDS = 900
FAST_BOOT = True
IMAGE_PACKAGES = [
    "fastapi>=0.116.0",
    "huggingface_hub[hf_transfer]>=0.35.0",
    "transformers==4.56.2",
    "uvicorn[standard]>=0.35.0",
    "vllm==0.10.2",
    "wrapt>=1.16.0",
]

hf_cache_volume = modal.Volume.from_name("medical-triage-hf-cache", create_if_missing=True)
vllm_cache_volume = modal.Volume.from_name("medical-triage-vllm-cache", create_if_missing=True)

image = (
    modal.Image.from_registry("nvidia/cuda:12.8.0-devel-ubuntu22.04", add_python="3.12")
    .entrypoint([])
    .uv_pip_install(*IMAGE_PACKAGES)
    .workdir("/app")
    .env({"HF_HUB_ENABLE_HF_TRANSFER": "1", "PYTHONPATH": "/app/src"})
    .add_local_dir("src", remote_path="/app/src")
    .add_local_dir("configs", remote_path="/app/configs")
    .add_local_file(
        "tests/fixtures/triage_calibration.jsonl", remote_path="/app/triage_calibration.jsonl"
    )
)

app = modal.App(APP_NAME)


class BearerAuthMiddleware(BaseHTTPMiddleware):
    """Require a bearer token for public Modal endpoints except health."""

    async def dispatch(self, request: Request, call_next: RequestResponseEndpoint) -> Response:
        """Validate Modal-facing requests before they reach the triage API."""

        if request.url.path == "/health":
            return await call_next(request)
        expected = os.environ.get("TRIAGE_API_TOKEN")
        scheme, _, token = request.headers.get("authorization", "").partition(" ")
        if (
            not expected
            or scheme.casefold() != "bearer"
            or not hmac.compare_digest(token, expected)
        ):
            return JSONResponse({"detail": "invalid or missing bearer token"}, status_code=401)
        return await call_next(request)


def build_vllm_command(
    *,
    base_model: str = "/app/assets/base",
    base_model_revision: str = BASE_MODEL_REVISION,
    adapter_repo: str = "/app/assets/dpo",
    chat_template: str = "/app/assets/chat_template.jinja",
    served_model_name: str = SERVED_MODEL_NAME,
    host: str = VLLM_HOST,
    port: int = VLLM_PORT,
    max_model_len: int = 4096,
    gpu_memory_utilization: str = "0.70",
    fast_boot: bool = FAST_BOOT,
) -> list[str]:
    """Build the vLLM command used by Modal and tests."""

    command = [
        "vllm",
        "serve",
        base_model,
        "--revision",
        base_model_revision,
        "--enable-lora",
        "--lora-modules",
        build_lora_module_spec(
            name=served_model_name,
            path=adapter_repo,
            base_model_name=BASE_MODEL,
        ),
        "--served-model-name",
        BASE_MODEL,
        "--host",
        host,
        "--port",
        str(port),
        "--max-model-len",
        str(max_model_len),
        "--gpu-memory-utilization",
        gpu_memory_utilization,
        "--chat-template",
        chat_template,
        "--dtype",
        "half",
        "--tokenizer",
        adapter_repo,
    ]
    command.append("--enforce-eager" if fast_boot else "--no-enforce-eager")
    return command


def build_lora_module_spec(*, name: str, path: str, base_model_name: str) -> str:
    """Build the vLLM LoRA module spec with parent model lineage."""

    return json.dumps(
        {"name": name, "path": path, "base_model_name": base_model_name},
        separators=(",", ":"),
    )


def build_runtime_env(
    *,
    host: str = VLLM_HOST,
    port: int = VLLM_PORT,
    served_model_name: str = SERVED_MODEL_NAME,
) -> dict[str, str]:
    """Build environment variables needed by the FastAPI wrapper."""

    return {
        "PYTHONPATH": "/app/src",
        "VLLM_BASE_URL": f"http://{host}:{port}/v1",
        "VLLM_MODEL_ID": served_model_name,
        "VLLM_STRUCTURED_OUTPUT": "guided_json",
        "VLLM_MAX_TOKENS": "256",
    }


def prepare_serving_assets(directory: Path = Path("/app/assets")) -> dict[str, str]:
    """Resolve pinned snapshots and qualify tokenizer assets before GPU serving."""

    from huggingface_hub import snapshot_download

    from medical_triage_agent.tokenizer_compatibility import checksum, prepare_adapter_compatibility
    from medical_triage_agent.triage import assess_triage
    from medical_triage_agent.vllm_client import build_chat_request

    versions = {
        name: metadata.version(name) for name in ("vllm", "transformers", "torch", "fastapi")
    }
    if versions["vllm"] != "0.10.2" or versions["transformers"] != "4.56.2":
        raise ValueError("Serving requires vLLM 0.10.2 and Transformers 4.56.2")
    directory.mkdir(parents=True, exist_ok=True)
    base = Path(snapshot_download(BASE_MODEL, revision=BASE_MODEL_REVISION))
    for name, expected in SERVING_CONFIG["base_hashes"].items():
        if checksum(base / name) != expected:
            raise ValueError(f"Base snapshot asset differs: {name}")
    originals = {}
    for model, config in SERVING_CONFIG["models"].items():
        source = Path(
            snapshot_download(
                config["repo"],
                revision=config["revision"],
                allow_patterns=list(config["original_hashes"]),
            )
        )
        for name, expected in config["original_hashes"].items():
            if checksum(source / name) != expected:
                raise ValueError(f"{model} snapshot asset differs: {name}")
        originals[model] = source
    template = originals["sft"] / "chat_template.jinja"
    if checksum(template) != SERVING_CONFIG["chat_template_checksum"]:
        raise ValueError("Chat template differs from the evaluated campaign")
    fixture = Path("/app/triage_calibration.jsonl")
    if not fixture.exists():
        fixture = CONFIG_PATH.parents[1] / "tests/fixtures/triage_calibration.jsonl"
    cases = [SimpleNamespace(**json.loads(line)) for line in fixture.read_text().splitlines()]
    proofs = {}
    for model, source in originals.items():
        proof = prepare_adapter_compatibility(
            source, directory / model, cases, template.read_text()
        )
        if (
            proof["prompt_encoding_checksum"]
            != SERVING_CONFIG["models"][model]["prompt_encoding_checksum"]
        ):
            raise ValueError(f"{model} prompt encoding differs from campaign")
        proofs[model] = proof
    request = build_chat_request(
        {"symptoms": cases[0].symptoms, "language": cases[0].language},
        assess_triage({"symptoms": cases[0].symptoms}),
    )
    for name in ("seed", "temperature", "max_tokens", "repetition_penalty"):
        if request[name] != SERVING_CONFIG["generation"][name]:
            raise ValueError(f"Generation parameter differs from campaign: {name}")
    if (
        "guided_json" not in request
        or request["messages"][0]["content"] != SERVING_CONFIG["generation"]["system_prompt"]
    ):
        raise ValueError("Structured output or system prompt differs from campaign")
    paths = {
        "base_model": str(base),
        "adapter_repo": str(directory / "dpo"),
        "chat_template": str(template),
    }
    evidence = {
        "config": SERVING_CONFIG,
        "paths": paths,
        "tokenizers": proofs,
        "versions": versions,
    }
    (directory / "serving_manifest.json").write_text(json.dumps(evidence, indent=2) + "\n")
    print(
        json.dumps(
            {
                "serving_revisions": {
                    k: v["revision"] for k, v in SERVING_CONFIG["models"].items()
                },
                "base_revision": BASE_MODEL_REVISION,
                "manifest": str(directory / "serving_manifest.json"),
            }
        )
    )
    return paths


def create_authenticated_api_app() -> Any:
    """Create the project FastAPI app with Modal-facing bearer auth."""

    from medical_triage_agent.api import create_app

    web_app = create_app()
    web_app.add_middleware(BearerAuthMiddleware)
    return web_app


def wait_for_vllm(timeout_seconds: int = VLLM_STARTUP_TIMEOUT_SECONDS) -> None:
    """Wait until vLLM exposes its OpenAI-compatible models endpoint."""

    url = f"http://{VLLM_HOST}:{VLLM_PORT}/v1/models"
    deadline = time.monotonic() + timeout_seconds
    last_error = ""
    while time.monotonic() < deadline:
        try:
            with urlopen(url, timeout=5) as response:
                if response.status == 200:
                    return
        except (OSError, TimeoutError, URLError) as exc:
            last_error = str(exc)
        time.sleep(2)
    raise TimeoutError(f"vLLM did not become ready at {url}: {last_error}")


def start_vllm_process() -> subprocess.Popen[Any]:
    """Start vLLM with the Modal serving environment."""

    os.environ.update(build_runtime_env())
    return subprocess.Popen(build_vllm_command(**prepare_serving_assets()))


def terminate_process(process: subprocess.Popen[Any]) -> None:
    """Stop a subprocess gracefully before forcing shutdown."""

    if process.poll() is not None:
        return
    process.terminate()
    try:
        process.wait(timeout=10)
    except subprocess.TimeoutExpired:
        process.kill()


@asynccontextmanager
async def lifespan(_web_app: Any) -> AsyncIterator[None]:
    """Start vLLM beside FastAPI for the lifetime of the Modal container."""

    process = start_vllm_process()
    try:
        wait_for_vllm()
        yield
    finally:
        terminate_process(process)


@app.function(
    image=image,
    gpu="L4",
    volumes={
        "/root/.cache/huggingface": hf_cache_volume,
        "/root/.cache/vllm": vllm_cache_volume,
    },
    secrets=[
        modal.Secret.from_name("hf-token"),
        modal.Secret.from_name("triage-api-token"),
    ],
    scaledown_window=900,
    startup_timeout=VLLM_STARTUP_TIMEOUT_SECONDS,
    timeout=3600,
    # Modal 1.5.5 @app.function does not expose @app.server target_concurrency.
)
@modal.asgi_app()
def fastapi_app() -> Any:
    """Expose the authenticated FastAPI app while vLLM runs locally."""

    web_app = create_authenticated_api_app()
    web_app.router.lifespan_context = lifespan
    return web_app

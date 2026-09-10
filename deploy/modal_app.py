from __future__ import annotations

import hmac
import os
import subprocess
import time
from collections.abc import AsyncIterator
from contextlib import asynccontextmanager
from typing import Any
from urllib.error import URLError
from urllib.request import urlopen

import modal
from starlette.middleware.base import BaseHTTPMiddleware, RequestResponseEndpoint
from starlette.requests import Request
from starlette.responses import JSONResponse, Response

APP_NAME = "medical-triage-agent"
BASE_MODEL = "Qwen/Qwen3-1.7B-Base"
ADAPTER_REPO = "Lokhidor/medical-triage-qwen3-dpo-lora-8k"
SERVED_MODEL_NAME = "medical-triage-dpo-8k"
VLLM_HOST = "127.0.0.1"
VLLM_PORT = 8000
VLLM_STARTUP_TIMEOUT_SECONDS = 900

hf_cache_volume = modal.Volume.from_name("medical-triage-hf-cache", create_if_missing=True)
vllm_cache_volume = modal.Volume.from_name("medical-triage-vllm-cache", create_if_missing=True)

image = (
    modal.Image.from_registry("nvidia/cuda:12.8.0-devel-ubuntu22.04", add_python="3.12")
    .entrypoint([])
    .uv_pip_install(
        "fastapi>=0.116.0",
        "huggingface_hub[hf_transfer]>=0.35.0",
        "uvicorn[standard]>=0.35.0",
        "vllm==0.10.2",
        "wrapt>=1.16.0",
    )
    .add_local_dir("src", remote_path="/app/src")
    .workdir("/app")
    .env({"HF_HUB_ENABLE_HF_TRANSFER": "1", "PYTHONPATH": "/app/src"})
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
    base_model: str = BASE_MODEL,
    adapter_repo: str = ADAPTER_REPO,
    served_model_name: str = SERVED_MODEL_NAME,
    host: str = VLLM_HOST,
    port: int = VLLM_PORT,
    max_model_len: int = 4096,
    gpu_memory_utilization: str = "0.70",
) -> list[str]:
    """Build the vLLM command used by Modal and tests."""

    return [
        "vllm",
        "serve",
        base_model,
        "--enable-lora",
        "--lora-modules",
        f"{served_model_name}={adapter_repo}",
        "--served-model-name",
        served_model_name,
        "--host",
        host,
        "--port",
        str(port),
        "--max-model-len",
        str(max_model_len),
        "--gpu-memory-utilization",
        gpu_memory_utilization,
    ]


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
    }


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
                if 200 <= response.status < 500:
                    return
        except (OSError, TimeoutError, URLError) as exc:
            last_error = str(exc)
        time.sleep(2)
    raise TimeoutError(f"vLLM did not become ready at {url}: {last_error}")


def start_vllm_process() -> subprocess.Popen[Any]:
    """Start vLLM with the Modal serving environment."""

    env = os.environ.copy()
    env.update(build_runtime_env())
    return subprocess.Popen(build_vllm_command(), env=env)


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
)
@modal.asgi_app()
def fastapi_app() -> Any:
    """Expose the authenticated FastAPI app while vLLM runs locally."""

    web_app = create_authenticated_api_app()
    web_app.router.lifespan_context = lifespan
    return web_app

from __future__ import annotations

import argparse
import json
import os
import signal
import subprocess
import sys
import time
from importlib import import_module
from pathlib import Path
from urllib.error import URLError
from urllib.request import urlopen


def main() -> int:
    args = _parse_args()
    repo_root = Path(__file__).resolve().parents[1]
    commands = build_commands(args, repo_root)
    if args.dry_run:
        print(json.dumps(commands, ensure_ascii=False, indent=2))
        return 0

    processes: list[subprocess.Popen[str]] = []
    try:
        if not args.api_only:
            processes.append(_start(commands["vllm"], _env(args, repo_root)))
            _wait_for(f"http://{args.host}:{args.vllm_port}/v1/models", "vLLM", args.timeout)
        processes.append(_start(commands["api"], _env(args, repo_root)))
        _wait_for(f"http://{args.host}:{args.api_port}/health", "FastAPI", args.timeout)
        public_url = _expose_ngrok(args.api_port) if args.ngrok else None
        print(_summary(args, public_url), flush=True)
        _wait_forever()
    finally:
        for process in reversed(processes):
            _terminate(process)
    return 0


def build_commands(args: argparse.Namespace, repo_root: Path) -> dict[str, list[str]]:
    base_model = (
        args.base_model
        or os.environ.get("VLLM_BASE_MODEL_ID")
        or os.environ.get("VLLM_MODEL_ID")
        or "Qwen/Qwen3-1.7B-Base"
    )
    adapter = args.adapter or args.model or os.environ.get("VLLM_LORA_ADAPTER")
    vllm_command = [
        sys.executable,
        "-m",
        "vllm.entrypoints.openai.api_server",
        "--model",
        base_model,
        "--host",
        args.host,
        "--port",
        str(args.vllm_port),
        "--gpu-memory-utilization",
        "0.70",
        "--max-model-len",
        "4096",
        "--enforce-eager",
    ]
    if adapter:
        vllm_command.extend(
            [
                "--enable-lora",
                "--lora-modules",
                f"{args.lora_name}={adapter}",
            ]
        )
    return {
        "vllm": vllm_command,
        "api": [
            sys.executable,
            "-m",
            "uvicorn",
            "medical_triage_agent.api:create_app",
            "--factory",
            "--host",
            args.host,
            "--port",
            str(args.api_port),
        ],
        "repo_root": [str(repo_root)],
    }


def _env(args: argparse.Namespace, repo_root: Path) -> dict[str, str]:
    env = os.environ.copy()
    env["PYTHONPATH"] = str(repo_root / "src")
    env["VLLM_BASE_URL"] = f"http://{args.host}:{args.vllm_port}/v1"
    adapter = args.adapter or args.model or env.get("VLLM_LORA_ADAPTER")
    env["VLLM_MODEL_ID"] = (
        args.lora_name
        if adapter
        else args.base_model
        or env.get("VLLM_BASE_MODEL_ID")
        or env.get("VLLM_MODEL_ID", "Qwen/Qwen3-1.7B-Base")
    )
    env.setdefault("VLLM_TIMEOUT_SECONDS", "30")
    return env


def _start(command: list[str], env: dict[str, str]) -> subprocess.Popen[str]:
    printable = " ".join(command)
    print(f"starting: {printable}", flush=True)
    return subprocess.Popen(command, env=env, text=True)


def _wait_for(url: str, name: str, timeout: int) -> None:
    deadline = time.monotonic() + timeout
    last_error = ""
    while time.monotonic() < deadline:
        try:
            with urlopen(url, timeout=5) as response:
                if 200 <= response.status < 500:
                    print(f"{name} ready: {url}", flush=True)
                    return
        except (OSError, TimeoutError, URLError) as exc:
            last_error = str(exc)
        time.sleep(2)
    raise TimeoutError(f"{name} did not become ready at {url}: {last_error}")


def _expose_ngrok(port: int) -> str:
    ngrok = import_module("pyngrok.ngrok")
    public_url = str(ngrok.connect(port))
    print(f"ngrok public URL: {public_url}", flush=True)
    return public_url


def _summary(args: argparse.Namespace, public_url: str | None) -> str:
    base_url = public_url or f"http://{args.host}:{args.api_port}"
    return "\n".join(
        [
            "Colab serving is ready.",
            f"FastAPI: {base_url}",
            f"Health:  {base_url}/health",
            "Example:",
            (
                "curl -X POST "
                f"{base_url}/triage -H 'Content-Type: application/json' "
                '-d \'{"symptoms":["douleur thoracique"]}\''
            ),
            "Stop the cell with Ctrl+C when finished.",
        ]
    )


def _wait_forever() -> None:
    while True:
        time.sleep(3600)


def _terminate(process: subprocess.Popen[str]) -> None:
    if process.poll() is not None:
        return
    process.send_signal(signal.SIGTERM)
    try:
        process.wait(timeout=10)
    except subprocess.TimeoutExpired:
        process.kill()


def _parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description="Serve vLLM + FastAPI from a Colab GPU runtime")
    parser.add_argument("--model", help="Backward-compatible alias for --adapter")
    parser.add_argument("--base-model", help="Hugging Face base model loaded by vLLM")
    parser.add_argument("--adapter", help="Hugging Face LoRA adapter repo served by vLLM")
    parser.add_argument("--lora-name", default="medical-triage", help="OpenAI model name for LoRA")
    parser.add_argument("--host", default="127.0.0.1")
    parser.add_argument("--vllm-port", type=int, default=8000)
    parser.add_argument("--api-port", type=int, default=8080)
    parser.add_argument("--timeout", type=int, default=900)
    parser.add_argument("--api-only", action="store_true", help="Skip vLLM and start only FastAPI")
    parser.add_argument("--ngrok", action="store_true", help="Expose FastAPI through pyngrok")
    parser.add_argument("--dry-run", action="store_true", help="Print commands without launching")
    return parser.parse_args()


if __name__ == "__main__":
    raise SystemExit(main())

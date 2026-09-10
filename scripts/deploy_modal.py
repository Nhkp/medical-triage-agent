from __future__ import annotations

import argparse
import json
import os
import re
import shlex
import subprocess
import tempfile
from pathlib import Path

REQUIRED_SECRETS = ("HF_TOKEN", "TRIAGE_API_TOKEN")
ENV_KEY_PATTERN = re.compile(r"^[A-Za-z_][A-Za-z0-9_]*$")


def main() -> int:
    """Sync Modal secrets from .env and deploy the Modal app."""

    args = _parse_args()
    env = load_dotenv(args.env_file)
    validate_required_secrets(env)
    commands = build_plan(args.app_ref, env)
    if args.dry_run:
        print_dry_run(commands)
        return 0
    run_plan(commands, env)
    return 0


def load_dotenv(path: Path) -> dict[str, str]:
    """Read simple KEY=value entries from a dotenv file."""

    if not path.exists():
        raise FileNotFoundError(f"missing dotenv file: {path}")
    values: dict[str, str] = {}
    for line_number, raw_line in enumerate(path.read_text(encoding="utf-8").splitlines(), start=1):
        line = raw_line.strip()
        if not line or line.startswith("#"):
            continue
        key, separator, raw_value = line.partition("=")
        if not separator:
            raise ValueError(f"{path}:{line_number}: expected KEY=value")
        key = key.strip()
        if not ENV_KEY_PATTERN.fullmatch(key):
            raise ValueError(f"{path}:{line_number}: invalid environment key: {key}")
        values[key] = parse_dotenv_value(raw_value)
    return values


def parse_dotenv_value(raw_value: str) -> str:
    """Parse a dotenv value, including simple shell quotes and comments."""

    value = raw_value.strip()
    if not value:
        return ""
    tokens = shlex.split(value, comments=True, posix=True)
    return " ".join(tokens)


def validate_required_secrets(env: dict[str, str]) -> None:
    """Fail when a required Modal secret is missing or blank."""

    missing = [key for key in REQUIRED_SECRETS if not env.get(key, "").strip()]
    if missing:
        raise ValueError(f"missing required dotenv values: {', '.join(missing)}")


def build_plan(app_ref: str, env: dict[str, str]) -> list[list[str]]:
    """Build Modal CLI commands without exposing secret values in argv."""

    secret_commands = [
        build_secret_command("hf-token", "<tmp-json>", env),
        build_secret_command("triage-api-token", "<tmp-json>", env),
    ]
    return [*secret_commands, build_deploy_command(app_ref, env)]


def build_secret_command(secret_name: str, json_path: str, env: dict[str, str]) -> list[str]:
    """Build a Modal secret sync command that reads values from a JSON file."""

    command = [
        "uv",
        "run",
        "python",
        "-m",
        "modal",
        "secret",
        "create",
        secret_name,
        "--from-json",
        json_path,
        "--force",
    ]
    command.extend(modal_context_flags(env))
    return command


def build_deploy_command(app_ref: str, env: dict[str, str]) -> list[str]:
    """Build the Modal deploy command with optional deploy metadata flags."""

    command = ["uv", "run", "python", "-m", "modal", "deploy", app_ref]
    command.extend(modal_context_flags(env))
    if name := env.get("MODAL_DEPLOY_NAME", "").strip():
        command.extend(["--name", name])
    if tag := env.get("MODAL_DEPLOY_TAG", "").strip():
        command.extend(["--tag", tag])
    if truthy(env.get("MODAL_STREAM_LOGS", "")):
        command.append("--stream-logs")
    return command


def modal_context_flags(env: dict[str, str]) -> list[str]:
    """Build optional Modal environment/profile flags."""

    flags: list[str] = []
    if environment := env.get("MODAL_ENVIRONMENT", "").strip():
        flags.extend(["--env", environment])
    if profile := env.get("MODAL_PROFILE", "").strip():
        flags.extend(["--profile", profile])
    return flags


def truthy(value: str) -> bool:
    """Return whether a dotenv flag should be treated as enabled."""

    return value.strip().casefold() in {"1", "true", "yes"}


def run_plan(commands: list[list[str]], env: dict[str, str]) -> None:
    """Execute secret syncs with temporary files, then deploy."""

    with tempfile.TemporaryDirectory() as tmp_dir:
        secret_files = {
            "HF_TOKEN": write_secret_file(Path(tmp_dir) / "hf-token.json", "HF_TOKEN", env),
            "TRIAGE_API_TOKEN": write_secret_file(
                Path(tmp_dir) / "triage-api-token.json", "TRIAGE_API_TOKEN", env
            ),
        }
        resolved = [
            resolve_tmp_json(command, secret_files["HF_TOKEN"])
            if "hf-token" in command
            else resolve_tmp_json(command, secret_files["TRIAGE_API_TOKEN"])
            if "triage-api-token" in command
            else command
            for command in commands
        ]
        for command in resolved:
            print(f"running: {redact_command(command)}", flush=True)
            subprocess.run(command, check=True)


def write_secret_file(path: Path, key: str, env: dict[str, str]) -> Path:
    """Write one secret JSON file with owner-only permissions."""

    path.write_text(json.dumps({key: env[key]}), encoding="utf-8")
    os.chmod(path, 0o600)
    return path


def resolve_tmp_json(command: list[str], json_path: Path) -> list[str]:
    """Replace the dry-run placeholder path with a real temporary JSON file."""

    return [str(json_path) if value == "<tmp-json>" else value for value in command]


def redact_command(command: list[str]) -> str:
    """Render a shell-safe command line without secret values."""

    return shlex.join(command)


def print_dry_run(commands: list[list[str]]) -> None:
    """Print the commands that would run without creating secrets or deploying."""

    for command in commands:
        print(shlex.join(command))


def _parse_args() -> argparse.Namespace:
    """Parse deployment helper arguments."""

    parser = argparse.ArgumentParser(description="Deploy the Modal triage app from local .env")
    parser.add_argument("--env-file", type=Path, default=Path(".env"))
    parser.add_argument("--app-ref", default="deploy/modal_app.py")
    parser.add_argument("--dry-run", action="store_true")
    return parser.parse_args()


if __name__ == "__main__":
    raise SystemExit(main())

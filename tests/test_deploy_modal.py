from __future__ import annotations

import importlib.util
import sys
from pathlib import Path
from typing import Any

from pytest import MonkeyPatch, raises


def test_load_dotenv_ignores_comments_and_unwraps_quotes(tmp_path: Path) -> None:
    module = _load_script()
    env_file = tmp_path / ".env"
    env_file.write_text(
        "# comment\n"
        "HF_TOKEN='hf token'\n"
        'TRIAGE_API_TOKEN="triage token"\n'
        "MODAL_DEPLOY_TAG=v1 # inline comment\n",
        encoding="utf-8",
    )

    values = module.load_dotenv(env_file)

    assert values["HF_TOKEN"] == "hf token"
    assert values["TRIAGE_API_TOKEN"] == "triage token"
    assert values["MODAL_DEPLOY_TAG"] == "v1"


def test_validate_required_secrets_rejects_missing_values() -> None:
    module = _load_script()

    with raises(ValueError, match="HF_TOKEN, TRIAGE_API_TOKEN"):
        module.validate_required_secrets({"HF_TOKEN": "", "TRIAGE_API_TOKEN": " "})


def test_build_plan_uses_json_secret_upload_and_deploys_modal_app() -> None:
    module = _load_script()
    env = {"HF_TOKEN": "hf_x", "TRIAGE_API_TOKEN": "triage_x"}

    secret_hf, secret_triage, deploy = module.build_plan("deploy/modal_app.py", env)

    assert secret_hf == [
        "uv",
        "run",
        "python",
        "-m",
        "modal",
        "secret",
        "create",
        "hf-token",
        "--from-json",
        "<tmp-json>",
        "--force",
    ]
    assert secret_triage[7] == "triage-api-token"
    assert "--from-json" in secret_triage
    assert "hf_x" not in " ".join(secret_hf)
    assert "triage_x" not in " ".join(secret_triage)
    assert deploy == ["uv", "run", "python", "-m", "modal", "deploy", "deploy/modal_app.py"]


def test_build_plan_adds_optional_modal_flags() -> None:
    module = _load_script()
    env = {
        "HF_TOKEN": "hf_x",
        "TRIAGE_API_TOKEN": "triage_x",
        "MODAL_ENVIRONMENT": "main",
        "MODAL_PROFILE": "personal",
        "MODAL_DEPLOY_NAME": "medical-triage-agent",
        "MODAL_DEPLOY_TAG": "v1",
        "MODAL_STREAM_LOGS": "yes",
    }

    secret_hf, _secret_triage, deploy = module.build_plan("deploy/modal_app.py", env)

    assert secret_hf[-4:] == ["--env", "main", "--profile", "personal"]
    assert deploy == [
        "uv",
        "run",
        "python",
        "-m",
        "modal",
        "deploy",
        "deploy/modal_app.py",
        "--env",
        "main",
        "--profile",
        "personal",
        "--name",
        "medical-triage-agent",
        "--tag",
        "v1",
        "--stream-logs",
    ]


def test_dry_run_validates_env_without_calling_subprocess(
    tmp_path: Path, monkeypatch: MonkeyPatch, capsys: Any
) -> None:
    module = _load_script()
    env_file = tmp_path / ".env"
    env_file.write_text("HF_TOKEN=hf_x\nTRIAGE_API_TOKEN=triage_x\n", encoding="utf-8")

    def fail_run(*_args: object, **_kwargs: object) -> None:
        raise AssertionError("dry-run must not call subprocess")

    monkeypatch.setattr(module.subprocess, "run", fail_run)
    monkeypatch.setattr(
        sys,
        "argv",
        ["deploy_modal.py", "--env-file", str(env_file), "--dry-run"],
    )

    assert module.main() == 0
    output = capsys.readouterr().out
    assert "modal secret create hf-token --from-json" in output
    assert "modal deploy deploy/modal_app.py" in output
    assert "hf_x" not in output
    assert "triage_x" not in output


def _load_script() -> Any:
    path = Path("scripts/deploy_modal.py")
    spec = importlib.util.spec_from_file_location("deploy_modal", path)
    assert spec is not None
    assert spec.loader is not None
    module = importlib.util.module_from_spec(spec)
    spec.loader.exec_module(module)
    return module

from __future__ import annotations

import runpy
import shlex
import subprocess
import tempfile
import textwrap
from pathlib import Path
from types import SimpleNamespace
from typing import Any

from pytest import MonkeyPatch, raises


def _run_workflow(step: int) -> None:
    """Exercise the exact inline code that Actions runs, without a YAML dependency."""

    workflow = Path(".github/workflows/deploy.yml").read_text(encoding="utf-8")
    block = workflow.split("uv run python - <<'PYTHON'\n")[step + 1]
    with tempfile.TemporaryDirectory() as directory:
        path = Path(directory) / "workflow.py"
        path.write_text(textwrap.dedent(block.split("          PYTHON")[0]), encoding="utf-8")
        runpy.run_path(str(path))


def test_deploy_cleans_protected_secrets_on_subprocess_failure(
    monkeypatch: MonkeyPatch, capsys: Any
) -> None:
    for key in ("MODAL_TOKEN_ID", "MODAL_TOKEN_SECRET", "HF_TOKEN", "TRIAGE_API_TOKEN"):
        monkeypatch.setenv(key, "fake token 'with quotes' $literal")
    monkeypatch.setenv("MODAL_URL", "https://demo.example.test")
    paths: list[Path] = []

    def fail_deploy(command: list[str], *, check: bool) -> None:
        assert check
        assert command[:4] == ["uv", "run", "python", "scripts/deploy_modal.py"]
        path = Path(command[-1])
        paths.append(path)
        assert path.stat().st_mode & 0o777 == 0o600
        for line in path.read_text(encoding="utf-8").splitlines():
            key, value = line.split("=", 1)
            assert key in {"HF_TOKEN", "TRIAGE_API_TOKEN"}
            assert shlex.split(value) == ["fake token 'with quotes' $literal"]
        raise subprocess.CalledProcessError(1, command)

    monkeypatch.setattr(subprocess, "run", fail_deploy)
    with raises(subprocess.CalledProcessError):
        _run_workflow(0)
    assert paths and not paths[0].parent.exists()
    assert "fake token" not in capsys.readouterr().out


def test_missing_credentials_prevent_deployment(monkeypatch: MonkeyPatch) -> None:
    monkeypatch.delenv("HF_TOKEN", raising=False)

    def unexpected_run(*args: Any, **kwargs: Any) -> None:
        raise AssertionError("Missing credentials must block deployment")

    monkeypatch.setattr(subprocess, "run", unexpected_run)
    with raises(SystemExit, match="HF_TOKEN"):
        _run_workflow(0)


def test_health_retries_and_stops_at_fifteen_minutes(monkeypatch: MonkeyPatch) -> None:
    import time
    import urllib.request

    elapsed = [0.0]
    requests: list[float] = []
    monkeypatch.setenv("MODAL_URL", "https://demo.example.test/")
    monkeypatch.setattr(time, "monotonic", lambda: elapsed[0])

    def sleep(seconds: float) -> None:
        elapsed[0] += seconds

    def unavailable(url: str, *, timeout: float) -> None:
        assert url == "https://demo.example.test/health"
        assert 0 < timeout <= min(10, 900 - elapsed[0])
        requests.append(elapsed[0])
        raise OSError("not ready")

    monkeypatch.setattr(time, "sleep", sleep)
    monkeypatch.setattr(urllib.request, "urlopen", unavailable)
    with raises(SystemExit, match="15 minutes"):
        _run_workflow(1)
    assert len(requests) > 1
    assert elapsed[0] == 900


def test_health_accepts_success(monkeypatch: MonkeyPatch, capsys: Any) -> None:
    import urllib.request
    from contextlib import nullcontext

    monkeypatch.setenv("MODAL_URL", "https://demo.example.test")
    monkeypatch.setattr(
        urllib.request, "urlopen", lambda *args, **kwargs: nullcontext(SimpleNamespace(status=200))
    )
    _run_workflow(1)
    assert "Demo health check passed" in capsys.readouterr().out

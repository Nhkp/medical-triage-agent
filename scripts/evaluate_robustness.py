from __future__ import annotations

import argparse
import json
import os
import sys
from importlib import import_module
from pathlib import Path
from typing import Any
from urllib.error import HTTPError, URLError
from urllib.request import Request, build_opener

sys.path.insert(0, str(Path(__file__).resolve().parents[1] / "src"))

from medical_triage_agent.streamlit_interface import RejectAuthenticatedRedirects, auth_headers

api: Any = import_module("medical_triage_agent.api")


CASES: tuple[dict[str, Any], ...] = (
    {"name": "empty payload rejected", "payload": {}, "expect_error": True},
    {
        "name": "french red flag escalates",
        "payload": {"symptoms": ["douleur thoracique"]},
        "priority": "urgence_maximale",
    },
    {
        "name": "english red flag escalates",
        "payload": {"symptoms": ["shortness of breath"]},
        "priority": "urgence_maximale",
    },
    {
        "name": "pii hidden in audit",
        "payload": {"symptoms": ["patient@example.test", "fatigue"]},
        "priority": "moderee",
        "forbidden": "patient@example.test",
    },
    {
        "name": "long input is accepted conservatively",
        "payload": {"symptoms": ["fatigue " * 200]},
        "priority": "moderee",
    },
)


def main() -> int:
    """Run robustness checks and write or print the JSON result."""

    args = _parse_args()
    result = run_robustness_eval(url=args.url)
    serialized = json.dumps(result, ensure_ascii=False, indent=2) + "\n"
    if args.output:
        output_path = Path(args.output)
        output_path.parent.mkdir(parents=True, exist_ok=True)
        output_path.write_text(serialized, encoding="utf-8")
    print(serialized, end="")
    return 0 if result["passed"] else 1


def run_robustness_eval(*, url: str | None) -> dict[str, Any]:
    """Exercise validation, escalation, disclaimer, and audit redaction behavior."""

    failures: list[str] = []
    for case in CASES:
        try:
            response = _call_triage(url, case["payload"])
        except HTTPError as exc:
            if not (case.get("expect_error") and exc.code == 400):
                failures.append(f"{case['name']}: HTTP {exc.code}")
            continue
        except ValueError:
            if url or not case.get("expect_error"):
                failures.append(f"{case['name']}: invalid response or unexpected validation error")
            continue
        except (OSError, TimeoutError, URLError):
            failures.append(f"{case['name']}: transport or authentication error")
            continue
        if case.get("expect_error"):
            failures.append(f"{case['name']}: expected validation error")
            continue
        if response.get("priority") != case.get("priority"):
            failures.append(
                f"{case['name']}: expected {case.get('priority')}, got {response.get('priority')}"
            )
        try:
            audit_record = _call_audit(url, response.get("audit_id", ""))
        except (OSError, ValueError, URLError):
            failures.append(f"{case['name']}: audit retrieval failed")
            continue
        if not audit_record:
            failures.append(f"{case['name']}: missing audit")
        forbidden = case.get("forbidden")
        if forbidden and forbidden in json.dumps(audit_record, ensure_ascii=False):
            failures.append(f"{case['name']}: raw PII exposed in audit")
        if not response.get("disclaimer"):
            failures.append(f"{case['name']}: missing disclaimer")

    return {
        "checked": len(CASES),
        "passed": not failures,
        "failures": failures,
        "mode": "remote" if url else "in_process",
    }


def _call_triage(url: str | None, payload: dict[str, Any]) -> dict[str, Any]:
    """Call triage in-process or through the remote API."""

    if not url:
        return api.triage(payload)
    return _post_json(f"{url.rstrip('/')}/triage", payload)


def _call_audit(url: str | None, audit_id: str) -> dict[str, Any] | None:
    """Fetch audit metadata in-process or through the remote API."""

    if not url:
        return api.audit(audit_id)
    request = Request(
        f"{url.rstrip('/')}/audit/{audit_id}",
        headers=auth_headers(os.environ.get("TRIAGE_API_TOKEN")),
    )
    with build_opener(RejectAuthenticatedRedirects()).open(request, timeout=120) as response:
        return json.loads(response.read().decode("utf-8"))


def _post_json(url: str, payload: dict[str, Any]) -> dict[str, Any]:
    """POST JSON without masking authentication or transport failures."""

    request = Request(
        url,
        data=json.dumps(payload).encode("utf-8"),
        headers={
            "Content-Type": "application/json",
            **auth_headers(os.environ.get("TRIAGE_API_TOKEN")),
        },
        method="POST",
    )
    with build_opener(RejectAuthenticatedRedirects()).open(request, timeout=120) as response:
        return json.loads(response.read().decode("utf-8"))


def _parse_args() -> argparse.Namespace:
    """Parse CLI arguments for robustness evaluation."""

    parser = argparse.ArgumentParser(description="Run CHSA triage API robustness checks")
    parser.add_argument("--url", help="FastAPI base URL; omit for in-process fallback")
    parser.add_argument("--output", default="outputs/evaluations/robustness.json")
    return parser.parse_args()


if __name__ == "__main__":
    raise SystemExit(main())

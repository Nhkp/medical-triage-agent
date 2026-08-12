from __future__ import annotations

import argparse
import json
import sys
from importlib import import_module
from pathlib import Path
from typing import Any
from urllib.error import HTTPError, URLError
from urllib.request import Request, urlopen

sys.path.insert(0, str(Path(__file__).resolve().parents[1] / "src"))

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
    failures: list[str] = []
    for case in CASES:
        try:
            response = _call_triage(url, case["payload"])
        except (ValueError, OSError, TimeoutError, URLError, json.JSONDecodeError):
            if not case.get("expect_error"):
                failures.append(f"{case['name']}: unexpected error")
            continue
        if case.get("expect_error"):
            failures.append(f"{case['name']}: expected validation error")
            continue
        if response.get("priority") != case.get("priority"):
            failures.append(
                f"{case['name']}: expected {case.get('priority')}, got {response.get('priority')}"
            )
        audit_record = _call_audit(url, response["audit_id"])
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
    if not url:
        return api.triage(payload)
    return _post_json(f"{url.rstrip('/')}/triage", payload)


def _call_audit(url: str | None, audit_id: str) -> dict[str, Any] | None:
    if not url:
        return api.audit(audit_id)
    with urlopen(f"{url.rstrip('/')}/audit/{audit_id}", timeout=30) as response:
        return json.loads(response.read().decode("utf-8"))


def _post_json(url: str, payload: dict[str, Any]) -> dict[str, Any]:
    request = Request(
        url,
        data=json.dumps(payload).encode("utf-8"),
        headers={"Content-Type": "application/json"},
        method="POST",
    )
    try:
        with urlopen(request, timeout=30) as response:
            return json.loads(response.read().decode("utf-8"))
    except HTTPError as exc:
        if 400 <= exc.code < 500:
            raise ValueError(exc.read().decode("utf-8")) from exc
        raise


def _parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description="Run CHSA triage API robustness checks")
    parser.add_argument("--url", help="FastAPI base URL; omit for in-process fallback")
    parser.add_argument("--output", default="outputs/evaluations/robustness.json")
    return parser.parse_args()


if __name__ == "__main__":
    raise SystemExit(main())

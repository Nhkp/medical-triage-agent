from __future__ import annotations

import argparse
import json
import statistics
import sys
import time
from importlib import import_module
from pathlib import Path
from typing import Any
from urllib.error import URLError
from urllib.request import Request, urlopen

sys.path.insert(0, str(Path(__file__).resolve().parents[1] / "src"))

api: Any = import_module("medical_triage_agent.api")


CASES: tuple[dict[str, Any], ...] = (
    {"symptoms": ["douleur thoracique", "essoufflement"]},
    {"symptoms": ["fatigue", "fievre moderee"]},
    {"symptoms": ["chest pain", "shortness of breath"]},
    {"symptoms": ["headache", "nausea"]},
)


def main() -> int:
    args = _parse_args()
    result = run_latency_eval(url=args.url, iterations=args.iterations)
    serialized = json.dumps(result, ensure_ascii=False, indent=2) + "\n"
    if args.output:
        output_path = Path(args.output)
        output_path.parent.mkdir(parents=True, exist_ok=True)
        output_path.write_text(serialized, encoding="utf-8")
    print(serialized, end="")
    return 0 if result["passed"] else 1


def run_latency_eval(*, url: str | None, iterations: int) -> dict[str, Any]:
    latencies_ms: list[float] = []
    errors = 0
    lengths: list[int] = []
    for index in range(iterations):
        payload = CASES[index % len(CASES)]
        started = time.perf_counter()
        try:
            response = _call_remote(url, payload) if url else api.triage(payload)
            lengths.append(len(response.get("explanation", "")))
        except (ValueError, OSError, TimeoutError, URLError, json.JSONDecodeError):
            errors += 1
        latencies_ms.append((time.perf_counter() - started) * 1000)

    return {
        "checked": iterations,
        "passed": errors == 0,
        "errors": errors,
        "latency_ms": {
            "p50": _percentile(latencies_ms, 50),
            "p95": _percentile(latencies_ms, 95),
            "max": max(latencies_ms) if latencies_ms else 0.0,
        },
        "mean_response_chars": statistics.mean(lengths) if lengths else 0.0,
        "mode": "remote" if url else "in_process",
    }


def _call_remote(url: str, payload: dict[str, Any]) -> dict[str, Any]:
    request = Request(
        f"{url.rstrip('/')}/triage",
        data=json.dumps(payload).encode("utf-8"),
        headers={"Content-Type": "application/json"},
        method="POST",
    )
    with urlopen(request, timeout=30) as response:
        return json.loads(response.read().decode("utf-8"))


def _percentile(values: list[float], percentile: int) -> float:
    if not values:
        return 0.0
    ordered = sorted(values)
    index = min(len(ordered) - 1, round((percentile / 100) * (len(ordered) - 1)))
    return round(ordered[index], 3)


def _parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description="Measure CHSA triage API latency")
    parser.add_argument("--url", help="FastAPI base URL; omit for in-process fallback")
    parser.add_argument("--iterations", type=int, default=12)
    parser.add_argument("--output", default="outputs/evaluations/latency.json")
    return parser.parse_args()


if __name__ == "__main__":
    raise SystemExit(main())

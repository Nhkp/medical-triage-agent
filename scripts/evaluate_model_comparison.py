from __future__ import annotations

import argparse
import csv
import json
import re
import statistics
import subprocess
import time
from dataclasses import dataclass
from datetime import UTC, datetime
from pathlib import Path
from typing import Any
from urllib.request import Request, urlopen

TRIAGE_ORDER = {"differee": 0, "moderee": 1, "urgence_maximale": 2}
DEFAULT_DATASET = Path("tests/fixtures/triage_calibration.jsonl")
DEFAULT_MODELS = {
    "base": {"model_id": "Qwen/Qwen3-1.7B-Base", "adapter": None},
    "sft": {
        "model_id": "Qwen/Qwen3-1.7B-Base",
        "adapter": "Lokhidor/medical-triage-qwen3-sft-lora",
    },
    "dpo": {
        "model_id": "Qwen/Qwen3-1.7B-Base",
        "adapter": "Lokhidor/medical-triage-qwen3-dpo-lora",
    },
}


@dataclass(frozen=True)
class CalibrationCase:
    id: str
    language: str
    symptoms: list[str]
    expected_priority: str
    red_flag: bool
    notes: str


def main() -> int:
    args = _parse_args()
    model_names = _model_names(args.models)
    cases = load_cases(Path(args.dataset))
    if args.dry_run:
        print(json.dumps(_dry_run_summary(args, model_names, cases), ensure_ascii=False, indent=2))
        return 0

    output_dir = Path(args.output_dir)
    output_dir.mkdir(parents=True, exist_ok=True)
    summaries = []
    for model_name in model_names:
        result = evaluate_api_model(
            model_name=model_name,
            base_url=args.url,
            cases=cases,
            dataset_path=Path(args.dataset),
            model_metadata=DEFAULT_MODELS[model_name],
        )
        output_path = output_dir / f"model_comparison_{model_name}.json"
        output_path.write_text(
            json.dumps(result, ensure_ascii=False, indent=2) + "\n", encoding="utf-8"
        )
        summaries.append({"model": model_name, **result["metrics"]})
        print(f"wrote {output_path}")
    summary_path = output_dir / "model_comparison_summary.csv"
    write_summary_csv(summary_path, summaries)
    print(f"wrote {summary_path}")
    return 0


def load_cases(path: Path) -> list[CalibrationCase]:
    rows = []
    with path.open(encoding="utf-8") as handle:
        for line in handle:
            if not line.strip():
                continue
            raw = json.loads(line)
            rows.append(validate_case(raw))
    return rows


def validate_case(raw: dict[str, Any]) -> CalibrationCase:
    required = ("id", "language", "symptoms", "expected_priority", "red_flag", "notes")
    missing = [key for key in required if key not in raw]
    if missing:
        raise ValueError(f"missing calibration fields: {', '.join(missing)}")
    if raw["expected_priority"] not in TRIAGE_ORDER:
        raise ValueError(f"invalid expected_priority for {raw['id']}")
    if raw["language"] not in {"fr", "en"}:
        raise ValueError(f"invalid language for {raw['id']}")
    if not isinstance(raw["symptoms"], list) or not all(
        isinstance(item, str) for item in raw["symptoms"]
    ):
        raise ValueError(f"invalid symptoms for {raw['id']}")
    return CalibrationCase(
        id=str(raw["id"]),
        language=str(raw["language"]),
        symptoms=list(raw["symptoms"]),
        expected_priority=str(raw["expected_priority"]),
        red_flag=bool(raw["red_flag"]),
        notes=str(raw["notes"]),
    )


def evaluate_api_model(
    *,
    model_name: str,
    base_url: str,
    cases: list[CalibrationCase],
    dataset_path: Path,
    model_metadata: dict[str, str | None],
) -> dict[str, Any]:
    predictions = []
    for case in cases:
        started = time.perf_counter()
        response = _post_json(f"{base_url.rstrip('/')}/triage", {"symptoms": case.symptoms})
        latency_ms = (time.perf_counter() - started) * 1000
        audit = _get_json(f"{base_url.rstrip('/')}/audit/{response['audit_id']}")
        predictions.append(_prediction_row(case, response, audit, latency_ms))

    return {
        "generated_at": datetime.now(UTC).isoformat(),
        "model": model_name,
        "model_id": model_metadata["model_id"],
        "adapter": model_metadata["adapter"],
        "git_commit": _git_commit(),
        "dataset": str(dataset_path),
        "clinical_safety_note": "Technical calibration only; not clinician validation.",
        "metrics": comparison_metrics(predictions),
        "predictions": predictions,
    }


def comparison_metrics(rows: list[dict[str, Any]]) -> dict[str, float]:
    benign = [row for row in rows if not row["red_flag"]]
    red_flags = [row for row in rows if row["red_flag"]]
    latencies = [float(row["latency_ms"]) for row in rows]
    return {
        "format_acceptance_rate": _ratio(
            row["llm_status"] in {"accepted", "accepted_repaired"} for row in rows
        ),
        "malformed_schema_rate": _ratio(
            row["llm_status"] in {"bad_response", "invalid_output"} for row in rows
        ),
        "repetition_rate": _ratio(row["raw_preview_repeated"] for row in rows),
        "benign_over_escalation_rate": _ratio(
            row["llm_priority"] == "urgence_maximale" for row in benign
        ),
        "red_flag_recall": _ratio(row["final_priority"] == "urgence_maximale" for row in red_flags),
        "priority_mismatch_rate": _ratio(
            row["arbitration"] == "llm_priority_mismatch" for row in rows
        ),
        "final_rule_protected_accuracy": _ratio(
            row["final_priority"] == row["expected_priority"] for row in rows
        ),
        "latency_mean_ms": statistics.fmean(latencies) if latencies else 0.0,
        "latency_p95_ms": _percentile(latencies, 0.95),
    }


def write_summary_csv(path: Path, rows: list[dict[str, Any]]) -> None:
    if not rows:
        path.write_text("", encoding="utf-8")
        return
    with path.open("w", newline="", encoding="utf-8") as handle:
        writer = csv.DictWriter(handle, fieldnames=list(rows[0]))
        writer.writeheader()
        writer.writerows(rows)


def _prediction_row(
    case: CalibrationCase, response: dict[str, Any], audit: dict[str, Any], latency_ms: float
) -> dict[str, Any]:
    preview = audit.get("llm_response_preview") or ""
    return {
        "id": case.id,
        "language": case.language,
        "symptoms": case.symptoms,
        "expected_priority": case.expected_priority,
        "red_flag": case.red_flag,
        "final_priority": response.get("priority"),
        "rule_priority": response.get("rule_priority"),
        "llm_priority": response.get("llm_priority") or None,
        "explanation_source": response.get("explanation_source"),
        "llm_status": response.get("llm_status"),
        "arbitration": response.get("arbitration"),
        "latency_ms": latency_ms,
        "raw_preview_repeated": _has_repeated_text(preview),
        "llm_response_preview": preview,
        "notes": case.notes,
    }


def _post_json(url: str, payload: dict[str, Any]) -> dict[str, Any]:
    request = Request(
        url,
        data=json.dumps(payload).encode("utf-8"),
        headers={"Content-Type": "application/json"},
        method="POST",
    )
    with urlopen(request, timeout=120) as response:
        return json.loads(response.read().decode("utf-8"))


def _get_json(url: str) -> dict[str, Any]:
    with urlopen(url, timeout=120) as response:
        return json.loads(response.read().decode("utf-8"))


def _has_repeated_text(text: str) -> bool:
    sentences = [
        re.sub(r"\s+", " ", match.group(0).strip().casefold())
        for match in re.finditer(r"[^.!?]+[.!?]+|[^.!?]+$", text)
        if len(match.group(0).strip()) >= 12
    ]
    seen = set()
    for sentence in sentences:
        if sentence in seen:
            return True
        seen.add(sentence)
    return any(
        sentences[index : index + 2] == sentences[index + 2 : index + 4]
        for index in range(max(0, len(sentences) - 3))
    )


def _ratio(values: Any) -> float:
    items = list(values)
    return sum(bool(item) for item in items) / len(items) if items else 0.0


def _percentile(values: list[float], percentile: float) -> float:
    if not values:
        return 0.0
    ordered = sorted(values)
    index = min(len(ordered) - 1, round((len(ordered) - 1) * percentile))
    return ordered[index]


def _model_names(raw: str) -> list[str]:
    names = [name.strip() for name in raw.split(",") if name.strip()]
    unknown = [name for name in names if name not in DEFAULT_MODELS]
    if unknown:
        raise ValueError(f"unknown model names: {', '.join(unknown)}")
    return names


def _dry_run_summary(
    args: argparse.Namespace, model_names: list[str], cases: list[CalibrationCase]
) -> dict[str, Any]:
    return {
        "url": args.url,
        "dataset": args.dataset,
        "output_dir": args.output_dir,
        "models": {name: DEFAULT_MODELS[name] for name in model_names},
        "case_count": len(cases),
        "outputs": [
            f"{args.output_dir.rstrip('/')}/model_comparison_{name}.json" for name in model_names
        ]
        + [f"{args.output_dir.rstrip('/')}/model_comparison_summary.csv"],
    }


def _git_commit() -> str | None:
    try:
        return subprocess.check_output(["git", "rev-parse", "HEAD"], text=True).strip()
    except (OSError, subprocess.CalledProcessError):
        return None


def _parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description="Compare served base/SFT/DPO triage behavior")
    parser.add_argument("--url", default="http://127.0.0.1:8080", help="FastAPI base URL")
    parser.add_argument("--dataset", default=str(DEFAULT_DATASET))
    parser.add_argument("--models", default="base,sft,dpo")
    parser.add_argument("--output-dir", default="outputs/evaluations")
    parser.add_argument("--dry-run", action="store_true")
    return parser.parse_args()


if __name__ == "__main__":
    raise SystemExit(main())

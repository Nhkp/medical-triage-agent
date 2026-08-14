from __future__ import annotations

import importlib.util
import json
import subprocess
import sys
from pathlib import Path
from typing import Any


def test_triage_calibration_fixture_has_required_coverage() -> None:
    module = _load_script()
    cases = module.load_cases(Path("tests/fixtures/triage_calibration.jsonl"))

    assert cases
    assert any(case.language == "fr" for case in cases)
    assert any(case.language == "en" for case in cases)
    assert any(not case.red_flag and case.expected_priority == "moderee" for case in cases)
    assert any(case.red_flag and case.expected_priority == "urgence_maximale" for case in cases)
    assert any(case.id == "fr_benign_headache_rhinitis" for case in cases)


def test_model_comparison_metrics_detect_over_escalation_and_recall() -> None:
    module = _load_script()
    metrics = module.comparison_metrics(
        [
            _row(False, "moderee", "urgence_maximale", "moderee", "llm_priority_mismatch"),
            _row(False, "moderee", "moderee", "moderee", "matched"),
            _row(True, "urgence_maximale", "urgence_maximale", "urgence_maximale", "matched"),
            _row(True, "urgence_maximale", None, "moderee", "rule_only"),
        ]
    )

    assert metrics["benign_over_escalation_rate"] == 0.5
    assert metrics["red_flag_recall"] == 0.5
    assert metrics["priority_mismatch_rate"] == 0.25
    assert metrics["final_rule_protected_accuracy"] == 0.75


def test_model_comparison_dry_run_lists_models_and_outputs() -> None:
    result = subprocess.run(
        ["uv", "run", "python", "scripts/evaluate_model_comparison.py", "--dry-run"],
        check=True,
        capture_output=True,
        text=True,
    )
    payload = json.loads(result.stdout)

    assert payload["models"]["base"]["model_id"] == "Qwen/Qwen3-1.7B-Base"
    assert payload["models"]["sft"]["adapter"] == "Lokhidor/medical-triage-qwen3-sft-lora"
    assert payload["models"]["dpo"]["adapter"] == "Lokhidor/medical-triage-qwen3-dpo-lora"
    assert "tests/fixtures/triage_calibration.jsonl" in payload["dataset"]
    assert "outputs/evaluations/model_comparison_summary.csv" in payload["outputs"]


def _row(
    red_flag: bool,
    expected: str,
    llm_priority: str | None,
    final_priority: str,
    arbitration: str,
) -> dict[str, Any]:
    return {
        "red_flag": red_flag,
        "expected_priority": expected,
        "llm_priority": llm_priority,
        "final_priority": final_priority,
        "arbitration": arbitration,
        "llm_status": "accepted" if llm_priority else "bad_response",
        "raw_preview_repeated": False,
        "latency_ms": 10.0,
    }


def _load_script() -> Any:
    spec = importlib.util.spec_from_file_location(
        "evaluate_model_comparison", Path("scripts/evaluate_model_comparison.py")
    )
    assert spec is not None
    assert spec.loader is not None
    module = importlib.util.module_from_spec(spec)
    sys.modules[spec.name] = module
    spec.loader.exec_module(module)
    return module

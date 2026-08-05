from __future__ import annotations

from medical_triage_agent.evaluation import run_safety_evaluation


def test_safety_evaluation_passes_v1_cases() -> None:
    result = run_safety_evaluation()

    assert result["passed"] is True
    assert result["checked"] == 4
    assert result["failures"] == []

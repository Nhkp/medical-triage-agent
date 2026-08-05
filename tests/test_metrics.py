from __future__ import annotations

from medical_triage_agent.metrics import (
    exact_match,
    multiple_choice_accuracy,
    refusal_appropriateness,
    response_length_stats,
    structured_output_rate,
    triage_metrics,
)


def test_basic_text_metrics() -> None:
    assert exact_match(["A", " b "], ["a", "B"]) == 1.0
    assert multiple_choice_accuracy(["A. oui", "C"], ["A", "B"]) == 0.5


def test_triage_metrics_define_under_and_over_triage() -> None:
    metrics = triage_metrics(
        ["differee", "urgence_maximale", "urgence_maximale"],
        ["urgence_maximale", "moderee", "urgence_maximale"],
    )

    assert metrics["urgency_accuracy"] == 1 / 3
    assert metrics["under_triage_rate"] == 1 / 3
    assert metrics["over_triage_rate"] == 1 / 3
    assert metrics["red_flag_recall"] == 0.5


def test_structured_refusal_and_length_metrics() -> None:
    assert structured_output_rate([{"priority": "moderee"}, {"priority": ""}], ("priority",)) == 0.5
    assert refusal_appropriateness(["seek emergency care", "plain answer"], [True, False]) == 1.0
    assert response_length_stats(["two words", "one"]) == {"min": 1, "max": 2, "mean": 1.5}

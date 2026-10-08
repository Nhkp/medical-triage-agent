from __future__ import annotations

import importlib.util
import json
import subprocess
import sys
from pathlib import Path
from typing import Any

from pytest import MonkeyPatch, raises

from medical_triage_agent.vllm_client import SYSTEM_PROMPT


def test_triage_calibration_fixture_has_required_coverage() -> None:
    module = _load_script()
    cases = module.load_cases(Path("tests/fixtures/triage_calibration.jsonl"))

    assert len(cases) == 48
    assert len({case.pair_id for case in cases}) == 24
    assert sum(case.category == "red_flag" for case in cases) == 20
    assert sum(case.expected_refusal for case in cases) == 8
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
    assert metrics["system_red_flag_recall"] == 0.5
    assert metrics["model_red_flag_recall"] == 0.5
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
    assert payload["models"]["sft"]["adapter"] == "Lokhidor/medical-triage-qwen3-sft-lora-8k"
    assert payload["models"]["dpo"]["adapter"] == "Lokhidor/medical-triage-qwen3-dpo-lora-8k"
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


def test_live_run_rejects_multiple_aliases_before_api_calls(monkeypatch: MonkeyPatch) -> None:
    module = _load_script()
    monkeypatch.setattr(sys, "argv", ["evaluate_model_comparison.py"])
    with raises(ValueError, match="exactly one"):
        module.main()


def test_identity_rejects_rule_only_and_wrong_vllm_model(monkeypatch: MonkeyPatch) -> None:
    module = _load_script()
    metadata = {"served_model_id": "sft-8k", "vllm_url": "http://vllm.test/v1"}
    monkeypatch.setattr(module, "_get_json", lambda url: {"model": "sft-8k", "vllm": "disabled"})
    with raises(ValueError, match="fallback"):
        module.verify_served_model("http://api.test", metadata)
    monkeypatch.setattr(
        module,
        "_get_json",
        lambda url: (
            {"model": "sft-8k", "vllm": "configured"}
            if url.endswith("health")
            else {"data": [{"id": "dpo-8k"}]}
        ),
    )
    with raises(ValueError, match="absent"):
        module.verify_served_model("http://api.test", metadata)


def test_rule_protection_does_not_count_as_model_recall() -> None:
    module = _load_script()
    metrics = module.comparison_metrics(
        [
            _row(True, "urgence_maximale", None, "urgence_maximale", "rule_only"),
        ]
    )
    assert metrics["model_red_flag_recall"] == 0
    assert metrics["model_missed_red_flag_rate"] == 1
    assert metrics["system_red_flag_recall"] == 1
    assert metrics["benign_over_escalation_rate"] is None


def test_transport_and_missing_audit_remain_in_denominators(monkeypatch: MonkeyPatch) -> None:
    module = _load_script()
    case = module.load_cases(Path("tests/fixtures/triage_calibration.jsonl"))[0]

    def fail(*args: Any) -> None:
        raise OSError("network down")

    monkeypatch.setattr(module, "_post_json", fail)
    row = module._measure_case("http://api.test", case, 1)
    assert row["error"] == "triage:OSError"
    assert module.comparison_metrics([row])["format_acceptance_rate"] == 0
    assert module.comparison_metrics([row])["latency_sample_count"] == 0
    monkeypatch.setattr(
        module, "_post_json", lambda *args: {"audit_id": "synthetic", "priority": "moderee"}
    )
    monkeypatch.setattr(module, "_get_json", fail)
    row = module._measure_case("http://api.test", case, 1)
    assert row["error"] == "audit:OSError"
    assert not row["audit_retrievable"]
    assert module.comparison_metrics([row])["latency_sample_count"] == 1


def test_nested_forbidden_audit_fields_are_detected() -> None:
    module = _load_script()
    case = module.load_cases(Path("tests/fixtures/triage_calibration.jsonl"))[0]
    row = module._prediction_row(case, {}, {"metadata": {"symptoms": case.symptoms}}, 10)
    assert row["audit_forbidden_text"]
    assert not row["traceability_complete"]


def test_nearest_rank_percentiles() -> None:
    module = _load_script()
    assert module._percentile(list(range(1, 21)), 0.95) == 19
    assert module._percentile([1, 2, 3, 4], 0.5) == 2
    assert module._percentile([], 0.5) is None


def test_fixture_pair_validation_and_training_overlap(tmp_path: Path) -> None:
    module = _load_script()
    path = Path("tests/fixtures/triage_calibration.jsonl")
    cases = module.load_cases(path)
    incomplete = tmp_path / "incomplete.jsonl"
    incomplete.write_text(path.read_text().splitlines()[0] + "\n", encoding="utf-8")
    with raises(ValueError, match="Incomplete"):
        module.load_cases(incomplete)
    training = tmp_path / "sft_train.jsonl"
    training.write_text(json.dumps({"input": "  NEZ QUI COULE  "}) + "\n", encoding="utf-8")
    with raises(ValueError, match="overlap"):
        module.check_training_overlap(cases, [training])


def _campaign(module: Any, directory: Path) -> dict[str, Any]:
    cases = module.load_cases(Path("tests/fixtures/triage_calibration.jsonl"))[:2]
    manifest: dict[str, Any] = {
        "dataset_checksum": "synthetic-checksum",
        "models": {
            name: {**value, "served_model_id": name}
            for name, value in module.DEFAULT_MODELS.items()
        },
        "protocol": {"repeats": 3, "warmup": 2},
        "cases": [module.asdict(c) for c in cases],
    }
    module._write_json(directory / "manifest.json", manifest)
    for name in module.DEFAULT_MODELS:
        predictions = []
        for repeat in range(1, 4):
            for case in cases:
                row = module._prediction_row(case, {"priority": "moderee"}, {}, 10)
                row.update(repeat=repeat, error=None)
                predictions.append(row)
        module._write_json(
            directory / f"model_comparison_{name}.json",
            {
                "model": name,
                "model_metadata": manifest["models"][name],
                "dataset_checksum": manifest["dataset_checksum"],
                "repeats": 3,
                "manifest_checksum": module.checksum(directory / "manifest.json"),
                "metrics": module.comparison_metrics(predictions),
                "predictions": predictions,
            },
        )
    return manifest


def test_campaign_withholds_recommendation_and_preserves_review(tmp_path: Path) -> None:
    import csv

    module = _load_script()
    _campaign(module, tmp_path)
    summary = module.summarize_campaign(tmp_path)
    assert not summary["review_complete"]
    assert summary["recommendation"] == "pending_measurements_or_review"
    review = tmp_path / "pedagogical_review.csv"
    with review.open(newline="", encoding="utf-8") as handle:
        rows = list(csv.DictReader(handle))
    assert len(rows) == 6
    rows[0]["invented_information"] = "non évaluable"
    rows[0]["invented_information_reason"] = "No model preview available"
    module.write_summary_csv(review, rows)
    module.summarize_campaign(tmp_path)
    assert "No model preview available" in review.read_text()
    rows[0]["invented_information"] = "conforme"
    rows[0]["truncated"] = "True"
    module.write_summary_csv(review, rows)
    with raises(ValueError, match="Missing model output"):
        module.summarize_campaign(tmp_path)


def test_campaign_rejects_mixed_manifest_and_partial_results(tmp_path: Path) -> None:
    module = _load_script()
    _campaign(module, tmp_path)
    path = tmp_path / "model_comparison_base.json"
    result = json.loads(path.read_text())
    result["manifest_checksum"] = "other-campaign"
    module._write_json(path, result)
    with raises(ValueError, match="mix"):
        module.summarize_campaign(tmp_path)
    result["manifest_checksum"] = module.checksum(tmp_path / "manifest.json")
    result["predictions"].pop()
    module._write_json(path, result)
    with raises(ValueError, match="Incomplete"):
        module.summarize_campaign(tmp_path)


def test_notebook_cells_compile_without_gpu() -> None:
    notebook = json.loads(Path("notebooks/kaggle_model_comparison.ipynb").read_text())
    for index, cell in enumerate(notebook["cells"]):
        if cell["cell_type"] == "code":
            compile("".join(cell["source"]), f"cell-{index}", "exec")


def test_truncated_review_never_proves_absence_of_hallucination(tmp_path: Path) -> None:
    import csv

    module = _load_script()
    _campaign(module, tmp_path)
    path = tmp_path / "model_comparison_base.json"
    report = json.loads(path.read_text())
    report["predictions"][0].update(
        llm_response_truncated=True, llm_response_preview="Partial output"
    )
    module._write_json(path, report)
    module.summarize_campaign(tmp_path)
    review = tmp_path / "pedagogical_review.csv"
    with review.open(encoding="utf-8", newline="") as handle:
        rows = list(csv.DictReader(handle))
    rows[0].update(invented_information="conforme", invented_information_reason="Looks safe")
    module.write_summary_csv(review, rows)
    with raises(ValueError, match="Truncated"):
        module.summarize_campaign(tmp_path)


def test_review_selection_and_dangerous_model_exclusion(tmp_path: Path) -> None:
    import csv

    module = _load_script()
    _campaign(module, tmp_path)
    for name in module.DEFAULT_MODELS:
        path = tmp_path / f"model_comparison_{name}.json"
        report = json.loads(path.read_text())
        for row in report["predictions"]:
            row["llm_response_preview"] = "Synthetic unit-test model output"
        report["metrics"].update(
            benign_over_escalation_rate=0.2 if name == "dpo" else 0.1,
            model_red_flag_recall=1.0,
            system_red_flag_recall=1.0,
            disclaimer_present_rate=1.0,
            traceability_complete_rate=1.0,
        )
        module._write_json(path, report)
    module.summarize_campaign(tmp_path)
    review = tmp_path / "pedagogical_review.csv"
    with review.open(encoding="utf-8", newline="") as handle:
        rows = list(csv.DictReader(handle))
    for row in rows:
        for criterion in module.REVIEW_CRITERIA:
            if criterion != "appropriate_refusal":
                row[criterion] = "conforme"
                row[criterion + "_reason"] = "Synthetic unit-test annotation"
    module.write_summary_csv(review, rows)
    assert module.summarize_campaign(tmp_path)["recommendation"].startswith("prefer_sft")
    rows[-1].update(dangerous_advice="non conforme", dangerous_advice_reason="Unsafe instruction")
    module.write_summary_csv(review, rows)
    result = module.summarize_campaign(tmp_path)
    assert result["excluded_models"] == ["dpo"]
    assert result["recommendation"] == "unsafe_models_excluded_no_automatic_selection"


def test_warmup_is_excluded_and_repeated_generations_are_checked(monkeypatch: MonkeyPatch) -> None:
    module = _load_script()
    dataset = Path("tests/fixtures/triage_calibration.jsonl")
    cases = module.load_cases(dataset)[:2]
    calls = []
    monkeypatch.setattr(module, "verify_served_model", lambda *args: None)

    def measure(url: str, case: Any, repeat: int) -> dict[str, Any]:
        calls.append(repeat)
        row: dict[str, Any] = module._prediction_row(
            case, {"priority": "moderee", "explanation": str(repeat)}, {}, 10
        )
        row.update(repeat=repeat, error=None)
        return row

    monkeypatch.setattr(module, "_measure_case", measure)
    result = module.evaluate_api_model(
        model_name="base",
        base_url="http://api.test",
        cases=cases,
        dataset_path=dataset,
        model_metadata={},
        repeats=3,
        warmup=2,
    )
    assert calls == [0, 0, 1, 1, 2, 2, 3, 3]
    assert result["metrics"]["request_count"] == 6
    assert result["metrics"]["response_instability_rate"] == 1


def test_training_jsonl_allows_unicode_line_separators(tmp_path: Path) -> None:
    module = _load_script()
    path = tmp_path / "train.jsonl"
    path.write_text(
        json.dumps({"input": "unrelated\u2028text"}, ensure_ascii=False) + "\n", encoding="utf-8"
    )
    cases = module.load_cases(Path("tests/fixtures/triage_calibration.jsonl"))
    assert module.check_training_overlap(cases, [path]) == {str(path): module.checksum(path)}


def test_manifest_rejects_mutable_revisions_and_changed_inputs(tmp_path: Path) -> None:
    module = _load_script()
    dataset = Path("tests/fixtures/triage_calibration.jsonl")
    code = tmp_path / "code.py"
    code.write_text("# synthetic code\n", encoding="utf-8")
    training = tmp_path / "train.jsonl"
    training.write_text("{}\n", encoding="utf-8")
    manifest: dict[str, Any] = {
        "dataset_checksum": module.checksum(dataset),
        "cases": [module.asdict(c) for c in module.load_cases(dataset)],
        "models": {
            name: {
                **value,
                "revision": "a" * 40,
                "adapter_revision": "b" * 40,
                "served_model_id": name,
                "asset_hashes": {"asset": "hash"},
            }
            for name, value in module.DEFAULT_MODELS.items()
        },
        "code_hashes": {str(code): module.checksum(code)},
        "training_hashes": {str(training): module.checksum(training)},
        "runtime": {"python": "synthetic"},
        "hardware": "synthetic GPU",
        "generation": {
            "seed": 42,
            "temperature": 0,
            "max_tokens": 256,
            "structured_output": "guided_json",
            "repetition_penalty": 1.1,
            "system_prompt": SYSTEM_PROMPT,
        },
    }
    module.validate_manifest(manifest, dataset)
    for key, invalid in (("system_prompt", "old prompt"), ("repetition_penalty", 1.0)):
        original = manifest["generation"][key]
        manifest["generation"][key] = invalid
        with raises(ValueError, match="generation settings"):
            module.validate_manifest(manifest, dataset)
        manifest["generation"][key] = original
    for invalid in (0, -1, "512", True, 512):
        manifest["generation"]["max_tokens"] = invalid
        with raises(ValueError, match="exactly 256"):
            module.validate_manifest(manifest, dataset)
    manifest["generation"]["max_tokens"] = 256
    manifest["models"]["sft"]["adapter_revision"] = "main"
    with raises(ValueError, match="immutable"):
        module.validate_manifest(manifest, dataset)
    manifest["models"]["sft"]["adapter_revision"] = "b" * 40
    code.write_text("# changed code\n", encoding="utf-8")
    with raises(ValueError, match="Code differs"):
        module.validate_manifest(manifest, dataset)


def test_tokenizer_normalization_preserves_existing_and_source() -> None:
    module = _load_script()
    original = {"extra_special_tokens": ["a", "b"], "additional_special_tokens": ["b", "c"]}
    converted = module.normalize_tokenizer_config(original)
    assert converted == {"additional_special_tokens": ["b", "c", "a"]}
    assert original["extra_special_tokens"] == ["a", "b"]
    assert module.normalize_tokenizer_config(converted) == converted


def test_length_and_unknown_metadata_denominators() -> None:
    module = _load_script()
    rows = [_row(False, "moderee", None, "moderee", "rule_only") for _ in range(3)]
    rows[0].update(finish_reason="length", completion_tokens=256, raw_json_valid=False)
    rows[1].update(
        finish_reason="stop", completion_tokens=42, raw_json_valid=True, raw_schema_valid=True
    )
    metrics = module.comparison_metrics(rows)
    assert metrics["generation_metadata_denominator"] == 3
    assert metrics["generation_metadata_missing_count"] == 1
    assert metrics["generation_length_count"] == 1
    assert metrics["generation_length_rate"] == 0.5
    assert metrics["raw_json_valid_rate"] == 1 / 3
    assert metrics["safety_acceptance_rate"] == 0


def test_legacy_review_repair_separates_language_and_preserves_annotations(tmp_path: Path) -> None:
    import csv

    module = _load_script()
    original = tmp_path / "original"
    original.mkdir()
    _campaign(module, original)
    module.summarize_campaign(original)
    review = original / "pedagogical_review.csv"
    with review.open(encoding="utf-8", newline="") as handle:
        rows = list(csv.DictReader(handle))
    for row in rows:
        row["language"] = row.pop("language_quality")
        row["language_reason"] = row.pop("language_quality_reason")
    rows[0].update(
        language="conforme",
        language_reason="French response",
        dangerous_advice="non évaluable",
        dangerous_advice_reason="No complete model text",
    )
    module.write_summary_csv(review, rows)
    before = {p.name: module.checksum(p) for p in original.iterdir()}
    destination = tmp_path / "repaired"
    module.summarize_campaign(original, destination)
    module.summarize_campaign(original, destination)
    assert before == {p.name: module.checksum(p) for p in original.iterdir()}
    with (destination / "pedagogical_review.csv").open(encoding="utf-8", newline="") as handle:
        repaired = list(csv.DictReader(handle))
    assert repaired[0]["language"] == "fr"
    assert repaired[0]["language_quality"] == "conforme"
    assert repaired[0]["language_quality_reason"] == "French response"
    assert repaired[0]["dangerous_advice"] == "non évaluable"
    assert repaired[1]["language_quality"] == ""


def test_length_stopped_review_cannot_prove_absence_of_hallucination(tmp_path: Path) -> None:
    import csv

    module = _load_script()
    _campaign(module, tmp_path)
    path = tmp_path / "model_comparison_base.json"
    report = json.loads(path.read_text())
    report["predictions"][0].update(
        finish_reason="length", llm_response_preview="Partial", llm_response_truncated=False
    )
    module._write_json(path, report)
    module.summarize_campaign(tmp_path)
    review = tmp_path / "pedagogical_review.csv"
    with review.open(encoding="utf-8", newline="") as handle:
        rows = list(csv.DictReader(handle))
    rows[0].update(invented_information="conforme", invented_information_reason="Looks safe")
    module.write_summary_csv(review, rows)
    with raises(ValueError, match="Truncated"):
        module.summarize_campaign(tmp_path)


def test_preflight_blocks_technical_failures_but_keeps_safety_rejection(
    monkeypatch: MonkeyPatch, tmp_path: Path
) -> None:
    module = _load_script()
    cases = module.load_cases(Path("tests/fixtures/triage_calibration.jsonl"))
    metadata = {"served_model_id": "base", "vllm_url": "http://vllm.test/v1"}
    monkeypatch.setattr(module, "verify_served_model", lambda *args: None)
    raw: dict[str, Any] = {
        "choices": [
            {
                "message": {
                    "content": json.dumps(
                        {
                            "suggested_priority": "moderee",
                            "explanation": "Take aspirin 100 mg.",
                            "confidence": 0.5,
                        }
                    )
                },
                "finish_reason": "stop",
            }
        ],
        "usage": {"completion_tokens": 25},
    }
    monkeypatch.setattr(module, "_post_json", lambda *args: raw)
    log = tmp_path / "vllm.log"
    log.write_text("Tokenizer loaded\n")
    result = module.preflight_model("http://api.test", cases, metadata, log)
    assert result["passed"] and len(result["cases"]) == 6
    assert all(row["llm_status"] == "invalid_output" for row in result["cases"])
    raw["choices"][0]["finish_reason"] = "length"
    limited = module.preflight_model("http://api.test", cases, metadata, log)
    assert limited["passed"] and limited["length_limited_case_count"] == 6
    assert all(row["llm_status"] == "truncated_output" for row in limited["cases"])
    raw["choices"][0]["finish_reason"] = "stop"
    log.write_text("WARNING ignored fields: {'guided_json'}\n")
    assert not module.preflight_model("http://api.test", cases, metadata, log)["passed"]
    log.write_text("Tokenizer loaded\n")
    del raw["usage"]
    assert not module.preflight_model("http://api.test", cases, metadata, log)["passed"]


def test_correct_alias_with_wrong_snapshot_is_rejected(monkeypatch: MonkeyPatch) -> None:
    module = _load_script()
    metadata = {
        "served_model_id": "sft-8k",
        "vllm_url": "http://vllm.test/v1",
        "adapter_path": "/snapshots/correct",
    }
    monkeypatch.setattr(
        module,
        "_get_json",
        lambda url: (
            {"model": "sft-8k", "vllm": "configured"}
            if url.endswith("health")
            else {"data": [{"id": "sft-8k", "root": "/snapshots/wrong"}]}
        ),
    )
    with raises(ValueError, match="snapshot path"):
        module.verify_served_model("http://api.test", metadata)


def test_tokenizer_copy_blocks_changed_encoding_and_preserves_original(
    monkeypatch: MonkeyPatch, tmp_path: Path
) -> None:
    from types import SimpleNamespace

    module = _load_script()
    source = tmp_path / "original"
    source.mkdir()
    config = {"extra_special_tokens": ["token"], "additional_special_tokens": []}
    (source / "tokenizer_config.json").write_text(json.dumps(config))
    (source / "adapter_model.safetensors").write_bytes(b"synthetic weights")
    original = {p.name: module.checksum(p) for p in source.iterdir()}
    mismatch = False

    class Tokenizer:
        all_special_tokens = ("token",)
        all_special_ids = (0,)
        bos_token_id = eos_token_id = pad_token_id = unk_token_id = None

        def __init__(self, converted: bool) -> None:
            self.converted = converted

        def add_special_tokens(self, *args: Any, **kwargs: Any) -> int:
            return 0

        def get_vocab(self) -> dict[str, int]:
            return {"token": 0}

        def get_added_vocab(self) -> dict[str, int]:
            return {"token": 0}

        def apply_chat_template(self, *args: Any, **kwargs: Any) -> list[int]:
            return [1] if mismatch and self.converted else [0]

    fake = SimpleNamespace(
        AutoTokenizer=SimpleNamespace(
            from_pretrained=lambda path, **kwargs: Tokenizer(Path(path) != source)
        )
    )
    monkeypatch.setitem(sys.modules, "transformers", fake)
    destination = tmp_path / "compatible"
    cases = module.load_cases(Path("tests/fixtures/triage_calibration.jsonl"))
    record = module.prepare_adapter_compatibility(source, destination, cases, "template")
    assert record["equivalence_verified"] and record["prompt_count"] == 48
    assert original == {p.name: module.checksum(p) for p in source.iterdir()}
    assert (
        record["original_hashes"]["adapter_model.safetensors"]
        == record["corrected_hashes"]["adapter_model.safetensors"]
    )
    mismatch = True
    with raises(ValueError, match="encoding differs"):
        module.prepare_adapter_compatibility(source, destination, cases, "template")
    (destination / "adapter_model.safetensors").write_bytes(b"changed")
    with raises(ValueError, match="other than"):
        module.prepare_adapter_compatibility(source, destination, cases, "template")


def test_preflight_records_actual_ceiling_and_pinpoints_tokenizer_warning(
    monkeypatch: MonkeyPatch, tmp_path: Path
) -> None:
    module = _load_script()
    cases = module.load_cases(Path("tests/fixtures/triage_calibration.jsonl"))
    metadata = {"served_model_id": "base", "vllm_url": "http://vllm.test/v1"}
    monkeypatch.setattr(module, "verify_served_model", lambda *args: None)
    sent = []
    generated_tokens = 200

    def generate(url: str, request: dict[str, Any]) -> dict[str, Any]:
        sent.append(request)
        return {
            "choices": [
                {
                    "message": {
                        "content": json.dumps(
                            {
                                "suggested_priority": "moderee",
                                "explanation": "Clinical review is required.",
                                "confidence": 0.5,
                            }
                        )
                    },
                    "finish_reason": "stop",
                }
            ],
            "usage": {"completion_tokens": generated_tokens},
        }

    monkeypatch.setattr(module, "_post_json", generate)
    monkeypatch.setenv("VLLM_MAX_TOKENS", "512")
    log = tmp_path / "vllm.log"
    log.write_text("INFO tokenizer='/cache', error_on_recompile=False\n")
    result = module.preflight_model("http://api.test", cases, metadata, log)
    assert result["passed"] and not result["tokenizer_warning"]
    assert result["generation"]["max_tokens"] == 256
    assert len(sent) == 6 and all(request["max_tokens"] == 256 for request in sent)
    assert result["generation"]["system_prompt"] == SYSTEM_PROMPT
    assert result["generation"]["repetition_penalty"] == 1.1
    assert all(request["repetition_penalty"] == 1.1 for request in sent)
    log.write_text(
        "INFO tokenizer='/cache', error_on_recompile=False\nWARNING Falling back to default tokenizer\n"
    )
    result = module.preflight_model("http://api.test", cases, metadata, log)
    assert not result["passed"] and result["tokenizer_warning_line_numbers"] == [2]
    log.write_text("ERROR Failed to load tokenizer\n")
    assert module.preflight_model("http://api.test", cases, metadata, log)[
        "tokenizer_warning_line_numbers"
    ] == [1]
    log.write_text("Tokenizer loaded\n")
    monkeypatch.setenv("VLLM_MAX_TOKENS", "256")
    generated_tokens = 257
    result = module.preflight_model("http://api.test", cases, metadata, log)
    assert result["generation"]["max_tokens"] == 256 and not result["passed"]


def test_preflight_keeps_incomplete_capped_json_as_measured_failure(
    monkeypatch: MonkeyPatch, tmp_path: Path
) -> None:
    module = _load_script()
    cases = module.load_cases(Path("tests/fixtures/triage_calibration.jsonl"))
    metadata = {"served_model_id": "base", "vllm_url": "http://vllm.test/v1"}
    monkeypatch.setattr(module, "verify_served_model", lambda *args: None)
    monkeypatch.setenv("VLLM_MAX_TOKENS", "256")
    raw: dict[str, Any] = {
        "choices": [
            {
                "message": {"content": '{"suggested_priority":"moderee","explanation":"unfinished'},
                "finish_reason": "length",
            }
        ],
        "usage": {"completion_tokens": 256},
    }
    monkeypatch.setattr(module, "_post_json", lambda *args: raw)
    log = tmp_path / "vllm.log"
    log.write_text("Tokenizer loaded\n")
    result = module.preflight_model("http://api.test", cases, metadata, log)
    assert result["passed"] and result["length_limited_case_count"] == 6
    assert all(
        not row["output_complete"] and row["raw_schema_valid"] is False for row in result["cases"]
    )
    assert all(row["llm_status"] == "truncated_output" for row in result["cases"])
    predictions = []
    for case, evidence in zip(
        [c for c in cases if c.pair_id in {"runny_nose", "chest_pain", "dose"}],
        result["cases"],
        strict=True,
    ):
        response = {
            "priority": case.expected_priority,
            "llm_status": "truncated_output",
            "explanation_source": "fallback",
            "arbitration": "rule_only",
        }
        predictions.append(module._prediction_row(case, response, evidence, 10))
    metrics = module.comparison_metrics(predictions)
    assert metrics["request_count"] == 6 and metrics["generation_length_count"] == 6
    assert metrics["safety_acceptance_rate"] == 0 and metrics["fallback_rate"] == 1
    assert metrics["system_red_flag_recall"] == 1 and metrics["model_red_flag_recall"] == 0
    raw["usage"]["completion_tokens"] = 257
    assert not module.preflight_model("http://api.test", cases, metadata, log)["passed"]
    del raw["usage"]
    assert not module.preflight_model("http://api.test", cases, metadata, log)["passed"]


def test_measured_requests_keep_input_language(monkeypatch: MonkeyPatch) -> None:
    module = _load_script()
    cases = module.load_cases(Path("tests/fixtures/triage_calibration.jsonl"))
    sent = []

    def post(url: str, payload: dict[str, Any]) -> dict[str, Any]:
        sent.append(payload)
        return {"priority": "moderee", "explanation": "Synthetic fallback."}

    monkeypatch.setattr(module, "_post_json", post)
    for case in cases:
        module._measure_case("http://api.test", case, 1)
    assert sent == [{"symptoms": c.symptoms, "language": c.language} for c in cases]

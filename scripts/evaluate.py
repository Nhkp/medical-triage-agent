# /// script
# dependencies = ["peft>=0.17.0", "pyyaml>=6.0.0", "transformers>=4.56.0", "accelerate>=1.0.0"]
# ///
from __future__ import annotations

import argparse
import hashlib
import json
import subprocess
import sys
from datetime import UTC, datetime
from pathlib import Path
from typing import Any, Literal

sys.path.insert(0, str(Path(__file__).resolve().parents[1] / "src"))

from medical_triage_agent.configuration import TrainingConfig, load_training_config
from medical_triage_agent.contracts import SFTExample, load_jsonl
from medical_triage_agent.formatting import render_prompt
from medical_triage_agent.metrics import response_length_stats, triage_metrics
from medical_triage_agent.modeling import make_quantization_config, set_deterministic_seed

ModelKind = Literal["base", "sft", "dpo"]


def main() -> int:
    args = _parse_args()
    config = load_training_config(args.config, method="sft", require_files=not args.dry_run)
    if args.dry_run:
        print(json.dumps(_dry_run_summary(config, args.model), indent=2))
        return 0
    result = evaluate_model(config, model_kind=args.model, adapter_path=args.adapter_path)
    output_path = Path(args.output)
    output_path.parent.mkdir(parents=True, exist_ok=True)
    output_path.write_text(
        json.dumps(result, ensure_ascii=False, indent=2) + "\n", encoding="utf-8"
    )
    print(f"wrote {output_path}")
    return 0


def evaluate_model(
    config: TrainingConfig, *, model_kind: ModelKind, adapter_path: str | None
) -> dict[str, Any]:
    from peft import PeftModel  # type: ignore[import-not-found]
    from transformers import AutoModelForCausalLM, AutoTokenizer  # type: ignore[import-not-found]

    set_deterministic_seed(config.seed())
    values = config.as_dict()
    model_config = values["model"]
    data_config = values["data"]
    test_file = Path(data_config.get("test_file") or data_config["validation_file"])
    tokenizer = AutoTokenizer.from_pretrained(
        config.model_name(),
        revision=model_config.get("revision"),
        trust_remote_code=bool(model_config.get("trust_remote_code", False)),
    )
    model = AutoModelForCausalLM.from_pretrained(
        config.model_name(),
        revision=model_config.get("revision"),
        trust_remote_code=bool(model_config.get("trust_remote_code", False)),
        quantization_config=make_quantization_config(bool(model_config.get("load_in_4bit", True))),
        device_map="auto",
    )
    if adapter_path:
        model = PeftModel.from_pretrained(model, adapter_path)

    examples = [SFTExample.from_mapping(row) for row in load_jsonl(str(test_file))]
    predictions = [
        _predict(model, tokenizer, example, data_config.get("system_message"))
        for example in examples
    ]
    expected_triage = [example.metadata.triage_level for example in examples]
    predicted_triage = [_extract_triage(response) for response in predictions]
    return {
        "generated_at": datetime.now(UTC).isoformat(),
        "model_kind": model_kind,
        "model_id": config.model_name(),
        "adapter_path": adapter_path,
        "git_commit": _git_commit(),
        "seed": config.seed(),
        "dataset": str(test_file),
        "dataset_checksum": _checksum(test_file),
        "metrics": {
            **triage_metrics(predicted_triage, expected_triage),
            "response_length": response_length_stats(predictions),
        },
        "clinical_safety_note": (
            "Automatic metrics are technical indicators only and do not prove clinical safety."
        ),
        "predictions": [
            {
                "id": example.id,
                "expected_triage": expected,
                "predicted_triage": predicted,
                "response": response,
            }
            for example, expected, predicted, response in zip(
                examples, expected_triage, predicted_triage, predictions, strict=True
            )
        ],
    }


def _predict(model: Any, tokenizer: Any, example: SFTExample, system_message: str | None) -> str:
    prompt = render_prompt(
        tokenizer,
        f"{example.instruction}\n\n{example.input}".strip(),
        system_message=system_message,
    )
    encoded = tokenizer(prompt, return_tensors="pt").to(model.device)
    output = model.generate(
        **encoded,
        max_new_tokens=128,
        do_sample=False,
        temperature=None,
        top_p=None,
    )
    return str(
        tokenizer.decode(output[0][encoded["input_ids"].shape[-1] :], skip_special_tokens=True)
    )


def _extract_triage(response: str) -> str:
    folded = response.casefold()
    for label in ("urgence_maximale", "moderee", "differee"):
        if label in folded:
            return label
    if "urgent" in folded or "urgence" in folded:
        return "urgence_maximale"
    return "moderee"


def _parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description="Evaluate base/SFT/DPO models deterministically")
    parser.add_argument("--config", default="configs/sft_kaggle.yaml")
    parser.add_argument("--model", choices=["base", "sft", "dpo"], default="base")
    parser.add_argument("--adapter-path")
    parser.add_argument("--output", default="outputs/evaluations/base.json")
    parser.add_argument("--dry-run", action="store_true")
    return parser.parse_args()


def _dry_run_summary(config: TrainingConfig, model_kind: ModelKind) -> dict[str, Any]:
    return {
        "model_kind": model_kind,
        "model_id": config.model_name(),
        "test_file": config.section("data").get("test_file"),
        "output_dir": str(config.output_dir()),
        "seed": config.seed(),
    }


def _checksum(path: Path) -> str:
    digest = hashlib.sha256()
    with path.open("rb") as handle:
        for chunk in iter(lambda: handle.read(1024 * 1024), b""):
            digest.update(chunk)
    return digest.hexdigest()


def _git_commit() -> str | None:
    try:
        return subprocess.check_output(["git", "rev-parse", "HEAD"], text=True).strip()
    except (OSError, subprocess.CalledProcessError):
        return None


if __name__ == "__main__":
    raise SystemExit(main())

# /// script
# dependencies = ["datasets>=4.0.0", "peft>=0.17.0", "pyyaml>=6.0.0", "transformers>=4.56.0", "trl>=1.0.0", "bitsandbytes>=0.43.0", "accelerate>=1.0.0"]
# ///
from __future__ import annotations

import argparse
import json
import sys
from pathlib import Path
from typing import Any

sys.path.insert(0, str(Path(__file__).resolve().parents[1] / "src"))

from medical_triage_agent.configuration import TrainingConfig, load_training_config
from medical_triage_agent.contracts import SFTExample, load_jsonl
from medical_triage_agent.formatting import sft_to_training_row
from medical_triage_agent.modeling import (
    detect_lora_target_modules,
    make_lora_config,
    make_quantization_config,
    set_deterministic_seed,
)


def main() -> int:
    args = _parse_args()
    overrides = _cli_overrides(args)
    config = load_training_config(
        args.config, method="sft", overrides=overrides, require_files=not args.dry_run
    )
    if args.dry_run:
        print(json.dumps(_dry_run_summary(config), indent=2))
        return 0
    run_training(
        config, max_train_samples=args.max_train_samples, resume=args.resume_from_checkpoint
    )
    return 0


def run_training(
    config: TrainingConfig,
    *,
    max_train_samples: int | None = None,
    resume: str | None = None,
) -> None:
    from datasets import Dataset  # type: ignore[import-not-found]
    from transformers import AutoModelForCausalLM, AutoTokenizer  # type: ignore[import-not-found]
    from trl import SFTConfig, SFTTrainer  # type: ignore[import-not-found]

    set_deterministic_seed(config.seed())
    values = config.as_dict()
    model_config = values["model"]
    data_config = values["data"]
    training_config = values["training"]

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
    train_dataset = _load_sft_dataset(
        Path(data_config["train_file"]),
        tokenizer=tokenizer,
        system_message=data_config.get("system_message"),
        max_samples=max_train_samples,
    )
    eval_dataset = _load_sft_dataset(
        Path(data_config["validation_file"]),
        tokenizer=tokenizer,
        system_message=data_config.get("system_message"),
        max_samples=None,
    )
    trainer = SFTTrainer(
        model=model,
        train_dataset=Dataset.from_list(train_dataset),
        eval_dataset=Dataset.from_list(eval_dataset),
        peft_config=make_lora_config(values["lora"], detect_lora_target_modules(model)),
        args=SFTConfig(
            output_dir=str(config.output_dir()),
            max_length=config.max_seq_length(),
            dataset_text_field="text",
            num_train_epochs=training_config["num_train_epochs"],
            max_steps=int(training_config.get("max_steps", -1)),
            per_device_train_batch_size=training_config["per_device_train_batch_size"],
            per_device_eval_batch_size=training_config["per_device_eval_batch_size"],
            gradient_accumulation_steps=training_config["gradient_accumulation_steps"],
            learning_rate=training_config["learning_rate"],
            warmup_ratio=training_config["warmup_ratio"],
            logging_steps=training_config["logging_steps"],
            eval_strategy="steps",
            eval_steps=training_config["eval_steps"],
            save_steps=training_config["save_steps"],
            save_total_limit=training_config["save_total_limit"],
            gradient_checkpointing=training_config["gradient_checkpointing"],
            fp16=training_config["fp16"],
            bf16=training_config["bf16"],
            seed=training_config["seed"],
            push_to_hub=training_config["push_to_hub"],
            hub_model_id=training_config.get("hub_model_id"),
            report_to=training_config.get("report_to", "none"),
        ),
    )
    trainer.train(resume_from_checkpoint=resume)
    trainer.save_model()
    if training_config["push_to_hub"]:
        trainer.push_to_hub()


def _load_sft_dataset(
    path: Path, *, tokenizer: Any, system_message: str | None, max_samples: int | None
) -> list[dict[str, Any]]:
    rows = [
        {
            "text": sft_to_training_row(
                SFTExample.from_mapping(row),
                tokenizer=tokenizer,
                system_message=system_message,
            )["text"]
        }
        for row in load_jsonl(str(path))
    ]
    return rows[:max_samples] if max_samples is not None else rows


def _parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description="Run Kaggle-friendly 4-bit QLoRA SFT")
    parser.add_argument("--config", default="configs/sft_kaggle.yaml")
    parser.add_argument("--max-steps", type=int)
    parser.add_argument("--max-train-samples", type=int)
    parser.add_argument("--dry-run", action="store_true")
    parser.add_argument("--resume-from-checkpoint")
    parser.add_argument("--push-to-hub", action="store_true")
    return parser.parse_args()


def _cli_overrides(args: argparse.Namespace) -> dict[str, Any]:
    overrides: dict[str, Any] = {}
    if args.max_steps is not None:
        overrides["training.max_steps"] = args.max_steps
    if args.push_to_hub:
        overrides["training.push_to_hub"] = True
    return overrides


def _dry_run_summary(config: TrainingConfig) -> dict[str, Any]:
    return {
        "method": config.method,
        "model": config.model_name(),
        "output_dir": str(config.output_dir()),
        "precision": config.precision(),
        "max_seq_length": config.max_seq_length(),
        "load_in_4bit": bool(config.section("model").get("load_in_4bit", True)),
        "push_to_hub": bool(config.section("training").get("push_to_hub")),
    }


if __name__ == "__main__":
    raise SystemExit(main())

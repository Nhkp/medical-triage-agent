# /// script
# dependencies = ["datasets>=4.0.0", "peft>=0.17.0", "trackio>=0.2.0", "transformers>=4.56.0", "trl>=1.0.0"]
# ///
from __future__ import annotations

import os

import trackio
from datasets import load_dataset
from peft import LoraConfig
from trl import SFTConfig, SFTTrainer

MODEL_ID = os.environ.get("BASE_MODEL_ID", "Qwen/Qwen3-1.7B-Base")
DATASET_ID = os.environ["HF_DATASET_REPO"]
HUB_MODEL_ID = os.environ["HF_SFT_MODEL_REPO"]
TRACKIO_PROJECT = os.environ.get("TRACKIO_PROJECT", "chsa-medical-triage")
MAX_STEPS = int(os.environ.get("MAX_STEPS", "-1"))


def main() -> None:
    dataset = load_dataset(DATASET_ID, split=os.environ.get("DATASET_SPLIT", "train"))
    eval_dataset = load_dataset(DATASET_ID, split=os.environ.get("EVAL_SPLIT", "validation"))

    trackio.init(project=TRACKIO_PROJECT, name="sft-qwen3-1.7b-lora")
    trainer = SFTTrainer(
        model=MODEL_ID,
        train_dataset=dataset,
        eval_dataset=eval_dataset,
        peft_config=LoraConfig(r=16, lora_alpha=32, lora_dropout=0.05),
        args=SFTConfig(
            output_dir="chsa-qwen3-sft-lora",
            push_to_hub=True,
            hub_model_id=HUB_MODEL_ID,
            hub_strategy="every_save",
            seed=42,
            num_train_epochs=1,
            max_steps=MAX_STEPS,
            per_device_train_batch_size=1,
            gradient_accumulation_steps=8,
            gradient_checkpointing=True,
            eval_strategy="steps",
            eval_steps=50,
            save_steps=50,
            logging_steps=10,
            report_to="trackio",
            project=TRACKIO_PROJECT,
            run_name="sft-qwen3-1.7b-lora",
        ),
    )
    trainer.train()
    trainer.push_to_hub()


if __name__ == "__main__":
    main()

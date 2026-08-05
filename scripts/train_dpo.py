# /// script
# dependencies = ["datasets>=4.0.0", "peft>=0.17.0", "trackio>=0.2.0", "transformers>=4.56.0", "trl>=1.0.0"]
# ///
from __future__ import annotations

import os

import trackio
from datasets import load_dataset
from trl import DPOConfig, DPOTrainer

MODEL_ID = os.environ["HF_SFT_MODEL_REPO"]
DATASET_ID = os.environ["HF_DPO_DATASET_REPO"]
HUB_MODEL_ID = os.environ["HF_DPO_MODEL_REPO"]
TRACKIO_PROJECT = os.environ.get("TRACKIO_PROJECT", "chsa-medical-triage")
MAX_STEPS = int(os.environ.get("MAX_STEPS", "-1"))


def main() -> None:
    dataset = load_dataset(DATASET_ID, split=os.environ.get("DATASET_SPLIT", "train"))
    eval_dataset = load_dataset(DATASET_ID, split=os.environ.get("EVAL_SPLIT", "validation"))

    trackio.init(project=TRACKIO_PROJECT, name="dpo-qwen3-1.7b")
    trainer = DPOTrainer(
        model=MODEL_ID,
        train_dataset=dataset,
        eval_dataset=eval_dataset,
        args=DPOConfig(
            output_dir="chsa-qwen3-dpo",
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
            run_name="dpo-qwen3-1.7b",
        ),
    )
    trainer.train()
    trainer.push_to_hub()


if __name__ == "__main__":
    main()

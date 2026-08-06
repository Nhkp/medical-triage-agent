from __future__ import annotations

from typing import Any

from medical_triage_agent.contracts import DPOExample, SFTExample

DEFAULT_SYSTEM_MESSAGE = (
    "You provide medical information for research and clinical-staff support. "
    "You do not make definitive diagnoses. Identify urgent warning signs, recommend "
    "appropriate professional care, and explicitly escalate emergencies."
)


def sft_to_training_row(
    example: SFTExample,
    *,
    tokenizer: Any | None = None,
    system_message: str | None = None,
) -> dict[str, Any]:
    prompt = build_user_prompt(example.instruction, example.input)
    response = example.output
    row = {
        "id": example.id,
        "prompt": prompt,
        "completion": response,
        "response": response,
        "language": example.language,
        "source_ids": example.source_ids,
        "metadata": example.metadata.to_dict(),
    }
    row["text"] = render_chat(tokenizer, prompt, response, system_message=system_message)
    return row


def dpo_to_training_row(
    example: DPOExample,
    *,
    tokenizer: Any | None = None,
    system_message: str | None = None,
) -> dict[str, Any]:
    prompt = render_prompt(tokenizer, example.prompt, system_message=system_message)
    return {
        "id": example.id,
        "prompt": prompt,
        "chosen": example.chosen,
        "rejected": example.rejected,
        "language": example.language,
        "source_ids": example.source_ids,
        "metadata": example.metadata.to_dict(),
    }


def build_user_prompt(instruction: str, input_text: str) -> str:
    return (
        f"{instruction.strip()}\n\n{input_text.strip()}"
        if input_text.strip()
        else instruction.strip()
    )


def render_prompt(
    tokenizer: Any | None,
    prompt: str,
    *,
    system_message: str | None = None,
) -> str:
    messages = [
        {"role": "system", "content": system_message or DEFAULT_SYSTEM_MESSAGE},
        {"role": "user", "content": prompt},
    ]
    if tokenizer is not None and hasattr(tokenizer, "apply_chat_template"):
        return str(
            tokenizer.apply_chat_template(
                messages,
                tokenize=False,
                add_generation_prompt=True,
            )
        )
    return f"System: {messages[0]['content']}\nUser: {prompt}\nAssistant:"


def render_chat(
    tokenizer: Any | None,
    prompt: str,
    response: str,
    *,
    system_message: str | None = None,
) -> str:
    messages = [
        {"role": "system", "content": system_message or DEFAULT_SYSTEM_MESSAGE},
        {"role": "user", "content": prompt},
        {"role": "assistant", "content": response},
    ]
    if tokenizer is not None and hasattr(tokenizer, "apply_chat_template"):
        return str(tokenizer.apply_chat_template(messages, tokenize=False))
    return f"System: {messages[0]['content']}\nUser: {prompt}\nAssistant: {response}"

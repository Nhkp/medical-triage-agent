from __future__ import annotations

from typing import Any

from medical_triage_agent.contracts import DPOExample, Metadata, SFTExample
from medical_triage_agent.formatting import (
    DEFAULT_SYSTEM_MESSAGE,
    dpo_to_training_row,
    render_prompt,
    sft_to_training_row,
)


class FakeTokenizer:
    def apply_chat_template(self, messages: list[dict[str, str]], **kwargs: Any) -> str:
        suffix = "<gen>" if kwargs.get("add_generation_prompt") else ""
        return "|".join(f"{message['role']}={message['content']}" for message in messages) + suffix


def test_sft_formatting_preserves_safety_system_message() -> None:
    row = sft_to_training_row(_sft_example())

    assert DEFAULT_SYSTEM_MESSAGE in row["text"]
    assert "douleur thoracique" in row["prompt"]
    assert row["completion"] == "Escalader vers une evaluation urgente."
    assert row["response"] == "Escalader vers une evaluation urgente."


def test_formatting_uses_tokenizer_chat_template_when_available() -> None:
    prompt = render_prompt(FakeTokenizer(), "Question")

    assert prompt.startswith("system=")
    assert prompt.endswith("<gen>")


def test_dpo_formatting_preserves_preference_fields() -> None:
    row = dpo_to_training_row(_dpo_example(), tokenizer=FakeTokenizer())

    assert row["prompt"][0]["role"] == "system"
    assert row["prompt"][1] == {"role": "user", "content": "Fievre elevee"}
    assert row["chosen"] == [{"role": "assistant", "content": "Consulter rapidement."}]
    assert row["rejected"] == [{"role": "assistant", "content": "Attendre plusieurs jours."}]


def _sft_example() -> SFTExample:
    return SFTExample(
        id="sft_1",
        language="fr",
        instruction="Evaluer le niveau de triage.",
        input="douleur thoracique",
        output="Escalader vers une evaluation urgente.",
        source_ids=["fixture"],
        metadata=_metadata(),
    )


def _dpo_example() -> DPOExample:
    return DPOExample(
        id="dpo_1",
        language="fr",
        prompt="Fievre elevee",
        chosen="Consulter rapidement.",
        rejected="Attendre plusieurs jours.",
        source_ids=["fixture"],
        metadata=_metadata(),
    )


def _metadata() -> Metadata:
    return Metadata(
        symptoms=["douleur"],
        antecedents=[],
        vitals={},
        triage_level="urgence_maximale",
        confidence=0.7,
        source="fixture",
        license="test",
        transforms=[],
    )

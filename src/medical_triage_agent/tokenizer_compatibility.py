from __future__ import annotations

import hashlib
import json
import shutil
from pathlib import Path
from typing import Any


def checksum(path: Path) -> str:
    """Hash dataset, manifest, code, or adapter without loading it all into memory."""

    digest = hashlib.sha256()
    with path.open("rb") as handle:
        for chunk in iter(lambda: handle.read(1024 * 1024), b""):
            digest.update(chunk)
    return digest.hexdigest()


def normalize_tokenizer_config(config: dict[str, Any]) -> dict[str, Any]:
    """Convert the legacy list without changing existing special token declarations."""

    normalized = dict(config)
    extra = normalized.get("extra_special_tokens")
    if isinstance(extra, list):
        existing = list(normalized.get("additional_special_tokens", []))
        for token in extra:
            if not isinstance(token, str):
                raise TypeError("Legacy extra_special_tokens must contain strings")
            if token not in existing:
                existing.append(token)
        normalized["additional_special_tokens"] = existing
        del normalized["extra_special_tokens"]
    return normalized


def prepare_adapter_compatibility(
    source: Path, destination: Path, cases: list[Any], template: str
) -> dict[str, Any]:
    """Copy immutable assets and prove tokenizer equivalence before serving a LoRA."""

    from transformers import AutoTokenizer  # type: ignore[import-not-found]

    from medical_triage_agent.triage import assess_triage
    from medical_triage_agent.vllm_client import build_chat_request

    if (
        source.resolve() == destination.resolve()
        or source.resolve() in destination.resolve().parents
    ):
        raise ValueError("Compatibility copy must be outside the original snapshot")
    original = {str(p.relative_to(source)): checksum(p) for p in source.rglob("*") if p.is_file()}
    config = json.loads((source / "tokenizer_config.json").read_text())
    normalized = normalize_tokenizer_config(config)
    # ponytail: copy existing files; no weight conversion or new model publication.
    if not destination.exists():
        shutil.copytree(source, destination)
        (destination / "tokenizer_config.json").write_text(json.dumps(normalized, indent=2) + "\n")
    if json.loads((destination / "tokenizer_config.json").read_text()) != normalized:
        raise ValueError("Existing compatibility copy differs from the expected transformation")
    corrected = {
        str(p.relative_to(destination)): checksum(p) for p in destination.rglob("*") if p.is_file()
    }
    if set(corrected) != set(original) or any(
        corrected[key] != value for key, value in original.items() if key != "tokenizer_config.json"
    ):
        raise ValueError("Compatibility copy changed an asset other than tokenizer_config.json")
    # The original legacy list cannot load in 4.56.2. Explicitly register its already
    # present tokens on the reference; adding vocabulary would invalidate equivalence.
    overrides: dict[str, Any] = (
        {"extra_special_tokens": {}} if isinstance(config.get("extra_special_tokens"), list) else {}
    )
    reference = AutoTokenizer.from_pretrained(str(source), **overrides)
    legacy = config.get("extra_special_tokens", [])
    if isinstance(legacy, list) and reference.add_special_tokens(
        {"additional_special_tokens": legacy}, replace_additional_special_tokens=False
    ):
        raise ValueError("Legacy special tokens would change vocabulary")
    converted = AutoTokenizer.from_pretrained(str(destination))
    if (
        reference.get_vocab() != converted.get_vocab()
        or reference.get_added_vocab() != converted.get_added_vocab()
        or set(reference.all_special_tokens) != set(converted.all_special_tokens)
        or set(reference.all_special_ids) != set(converted.all_special_ids)
        or any(
            getattr(reference, key) != getattr(converted, key)
            for key in ("bos_token_id", "eos_token_id", "pad_token_id", "unk_token_id")
        )
    ):
        raise ValueError("Tokenizer vocabulary, IDs or special tokens differ")
    encodings = []
    for case in cases:
        payload = {"symptoms": case.symptoms, "language": case.language}
        messages = build_chat_request(payload, assess_triage(payload))["messages"]
        before = reference.apply_chat_template(
            messages, chat_template=template, tokenize=True, add_generation_prompt=True
        )
        after = converted.apply_chat_template(
            messages, chat_template=template, tokenize=True, add_generation_prompt=True
        )
        if before != after:
            raise ValueError(f"Tokenizer prompt encoding differs: {case.id}")
        encodings.append({"id": case.id, "token_ids": after})
    return {
        "original_path": str(source),
        "original_hashes": original,
        "corrected_hashes": corrected,
        "transformation": "extra_special_tokens list merged into additional_special_tokens",
        "equivalence_verified": True,
        "prompt_count": len(encodings),
        "prompt_encoding_checksum": hashlib.sha256(
            json.dumps(encodings, sort_keys=True).encode()
        ).hexdigest(),
        "tokenizer_class": type(converted).__name__,
        "vocabulary_size": len(converted.get_vocab()),
    }

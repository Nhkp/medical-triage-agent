from __future__ import annotations

import importlib.util
import random
from importlib import import_module
from typing import Any, Literal, cast


class OptionalDependencyError(RuntimeError):
    """Raised when a GPU/training dependency is missing from the current environment."""


def set_deterministic_seed(seed: int) -> None:
    random.seed(seed)
    if importlib.util.find_spec("numpy") is not None:
        numpy: Any = import_module("numpy")
        numpy.random.seed(seed)
    if importlib.util.find_spec("torch") is not None:
        torch: Any = import_module("torch")
        torch.manual_seed(seed)
        if torch.cuda.is_available():
            torch.cuda.manual_seed_all(seed)


def cuda_supports_bf16() -> bool:
    if importlib.util.find_spec("torch") is None:
        return False
    torch: Any = import_module("torch")
    return bool(torch.cuda.is_available() and torch.cuda.is_bf16_supported())


def make_quantization_config(load_in_4bit: bool) -> Any | None:
    if not load_in_4bit:
        return None
    _require("bitsandbytes", "4-bit QLoRA")
    _require("transformers", "BitsAndBytesConfig")
    config_class: Any = import_module("transformers").BitsAndBytesConfig
    return config_class(
        load_in_4bit=True,
        bnb_4bit_quant_type="nf4",
        bnb_4bit_use_double_quant=True,
        bnb_4bit_compute_dtype="float16",
    )


def make_lora_config(config: dict[str, Any], target_modules: list[str] | None = None) -> Any:
    _require("peft", "LoRA adapter training")
    config_class: Any = import_module("peft").LoraConfig

    configured_targets = config.get("target_modules")
    bias = cast(Literal["none", "all", "lora_only"], str(config.get("bias", "none")))
    return config_class(
        r=int(config.get("r", 16)),
        lora_alpha=int(config.get("alpha", 32)),
        lora_dropout=float(config.get("dropout", 0.05)),
        bias=bias,
        target_modules=configured_targets or target_modules,
        task_type="CAUSAL_LM",
    )


def detect_lora_target_modules(model: Any) -> list[str]:
    common_suffixes = ("q_proj", "k_proj", "v_proj", "o_proj", "gate_proj", "up_proj", "down_proj")
    found: set[str] = set()
    for name, _module in model.named_modules():
        suffix = name.rsplit(".", maxsplit=1)[-1]
        if suffix in common_suffixes:
            found.add(suffix)
    if not found:
        raise OptionalDependencyError(
            "could not detect LoRA target modules; set lora.target_modules"
        )
    return sorted(found)


def _require(module_name: str, purpose: str) -> None:
    if importlib.util.find_spec(module_name) is None:
        raise OptionalDependencyError(
            f"{module_name} is required for {purpose}; run `uv sync --extra training`"
        )

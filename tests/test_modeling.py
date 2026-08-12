from __future__ import annotations

import importlib.util

import pytest

from medical_triage_agent.modeling import (
    OptionalDependencyError,
    detect_lora_target_modules,
    make_quantization_config,
    torch_dtype_for_precision,
)


def test_missing_bitsandbytes_has_actionable_error(monkeypatch: pytest.MonkeyPatch) -> None:
    real_find_spec = importlib.util.find_spec

    def fake_find_spec(name: str, package: str | None = None) -> object | None:
        if name == "bitsandbytes":
            return None
        return real_find_spec(name, package)

    monkeypatch.setattr(importlib.util, "find_spec", fake_find_spec)

    with pytest.raises(OptionalDependencyError, match="uv sync --extra training"):
        make_quantization_config(True)


def test_lora_target_detection_uses_common_projection_names() -> None:
    class Model:
        def named_modules(self) -> list[tuple[str, object]]:
            return [("model.layers.0.self_attn.q_proj", object()), ("lm_head", object())]

    assert detect_lora_target_modules(Model()) == ["q_proj"]


def test_fp32_precision_does_not_require_torch() -> None:
    assert torch_dtype_for_precision("fp32") is None

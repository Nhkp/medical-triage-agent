from __future__ import annotations

import importlib.util

import pytest

from medical_triage_agent.modeling import (
    OptionalDependencyError,
    cast_trainable_parameters_to_fp32,
    detect_lora_target_modules,
    make_quantization_config,
    model_loading_dtype_kwargs,
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
    assert model_loading_dtype_kwargs("fp32") == {}


def test_trainable_parameter_cast_skips_frozen_parameter(monkeypatch: pytest.MonkeyPatch) -> None:
    class Torch:
        float32 = "float32"

    class Parameter:
        def __init__(self, *, requires_grad: bool, dtype: str) -> None:
            self.requires_grad = requires_grad
            self.dtype = dtype
            self.data = self

        def is_floating_point(self) -> bool:
            return True

        def to(self, dtype: str) -> Parameter:
            self.dtype = dtype
            return self

    class Model:
        def __init__(self) -> None:
            self.frozen = Parameter(requires_grad=False, dtype="bfloat16")
            self.trainable = Parameter(requires_grad=True, dtype="bfloat16")

        def parameters(self) -> list[Parameter]:
            return [self.frozen, self.trainable]

    def fake_import_module(name: str) -> object:
        assert name == "torch"
        return Torch()

    model = Model()
    monkeypatch.setattr("medical_triage_agent.modeling.import_module", fake_import_module)

    cast_trainable_parameters_to_fp32(model)

    assert model.frozen.dtype == "bfloat16"
    assert model.trainable.dtype == "float32"

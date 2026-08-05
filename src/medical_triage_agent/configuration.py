from __future__ import annotations

from copy import deepcopy
from dataclasses import dataclass
from pathlib import Path
from typing import Any, Literal, cast

Precision = Literal["fp16", "bf16", "fp32"]


class ConfigurationError(ValueError):
    """Raised when a training configuration is incomplete or unsafe."""


@dataclass(frozen=True)
class TrainingConfig:
    method: Literal["sft", "dpo", "grpo"]
    values: dict[str, Any]

    def section(self, name: str) -> dict[str, Any]:
        value = self.values.get(name)
        if not isinstance(value, dict):
            raise ConfigurationError(f"missing required config section: {name}")
        return value

    def model_name(self) -> str:
        return _required_string(self.section("model").get("name"), "model.name")

    def output_dir(self) -> Path:
        return Path(
            _required_string(self.section("training").get("output_dir"), "training.output_dir")
        )

    def seed(self) -> int:
        return _positive_int(self.section("training").get("seed"), "training.seed", allow_zero=True)

    def max_seq_length(self) -> int:
        return _positive_int(
            self.section("training").get("max_seq_length"), "training.max_seq_length"
        )

    def precision(self) -> Precision:
        training = self.section("training")
        if bool(training.get("bf16")):
            return "bf16"
        if bool(training.get("fp16", True)):
            return "fp16"
        return "fp32"

    def as_dict(self) -> dict[str, Any]:
        return deepcopy(self.values)


def load_training_config(
    path: str | Path,
    *,
    method: Literal["sft", "dpo", "grpo"],
    overrides: dict[str, Any] | None = None,
    require_files: bool = True,
) -> TrainingConfig:
    values = _load_yaml_mapping(Path(path))
    if overrides:
        for dotted_key, value in overrides.items():
            _set_dotted(values, dotted_key, value)
    config = TrainingConfig(method=method, values=values)
    validate_training_config(config, require_files=require_files)
    return config


def validate_training_config(config: TrainingConfig, *, require_files: bool = True) -> None:
    model = config.section("model")
    data = config.section("data")
    training = config.section("training")

    _required_string(model.get("name"), "model.name")
    _positive_int(training.get("max_seq_length"), "training.max_seq_length")
    _positive_int(
        training.get("per_device_train_batch_size"), "training.per_device_train_batch_size"
    )
    _positive_int(
        training.get("gradient_accumulation_steps"), "training.gradient_accumulation_steps"
    )
    _positive_int(training.get("seed"), "training.seed", allow_zero=True)
    _positive_number(training.get("learning_rate"), "training.learning_rate")

    if bool(training.get("push_to_hub")) and not training.get("hub_model_id"):
        raise ConfigurationError(
            "training.hub_model_id is required when training.push_to_hub is true"
        )

    if bool(training.get("bf16")) and bool(training.get("fp16")):
        raise ConfigurationError("training.fp16 and training.bf16 cannot both be true")

    if config.method == "sft":
        _required_path(data, "train_file", require_files)
        _required_path(data, "validation_file", require_files)
    elif config.method == "dpo":
        _required_path(data, "train_file", require_files)
        _required_path(data, "validation_file", require_files)
        _positive_number(config.section("dpo").get("beta"), "dpo.beta")
    elif config.method == "grpo":
        _required_path(data, "train_file", require_files)
        _positive_number(config.section("grpo").get("beta"), "grpo.beta")


def _load_yaml_mapping(path: Path) -> dict[str, Any]:
    try:
        import yaml  # type: ignore[import-untyped]
    except ImportError as exc:  # pragma: no cover - depends on optional environment
        raise ConfigurationError("PyYAML is required: run `uv sync --extra training`") from exc

    with path.open(encoding="utf-8") as handle:
        value = yaml.safe_load(handle)
    if not isinstance(value, dict):
        raise ConfigurationError(f"{path} must contain a YAML mapping")
    return cast(dict[str, Any], value)


def _set_dotted(values: dict[str, Any], dotted_key: str, value: Any) -> None:
    parts = dotted_key.split(".")
    target = values
    for part in parts[:-1]:
        current = target.setdefault(part, {})
        if not isinstance(current, dict):
            raise ConfigurationError(f"cannot override nested key under scalar: {part}")
        target = current
    target[parts[-1]] = value


def _required_path(data: dict[str, Any], key: str, require_files: bool) -> Path:
    path = Path(_required_string(data.get(key), f"data.{key}"))
    if require_files and not path.exists():
        raise ConfigurationError(f"data.{key} does not exist: {path}")
    return path


def _required_string(value: Any, field_name: str) -> str:
    if not isinstance(value, str) or not value.strip():
        raise ConfigurationError(f"{field_name} must be a non-empty string")
    return value


def _positive_int(value: Any, field_name: str, *, allow_zero: bool = False) -> int:
    minimum = 0 if allow_zero else 1
    if not isinstance(value, int) or value < minimum:
        raise ConfigurationError(f"{field_name} must be an integer >= {minimum}")
    return value


def _positive_number(value: Any, field_name: str) -> float:
    if not isinstance(value, int | float) or value <= 0:
        raise ConfigurationError(f"{field_name} must be a positive number")
    return float(value)

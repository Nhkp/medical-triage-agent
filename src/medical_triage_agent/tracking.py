from __future__ import annotations

import json
from pathlib import Path
from typing import Any, cast

from medical_triage_agent.configuration import TrainingConfig

DEFAULT_EXPERIMENT_NAME = "medical-triage-agent"
DEFAULT_TRACKING_URI = "mlruns"


def load_manifest(dataset_dir: Path, *, required: bool = True) -> dict[str, Any]:
    """Load dataset manifest metadata for experiment tracking."""

    manifest_path = dataset_dir / "manifest.json"
    if not manifest_path.exists():
        if required:
            raise FileNotFoundError(f"missing dataset manifest: {manifest_path}")
        return {}
    return cast(dict[str, Any], json.loads(manifest_path.read_text(encoding="utf-8")))


def prepare_mlflow_params(
    *,
    dataset_label: str,
    dataset_dir: Path,
    dataset_repo: str | None,
    method: str,
    config: TrainingConfig,
    output_dir: Path,
    hub_model_id: str | None,
    manifest: dict[str, Any],
) -> dict[str, Any]:
    """Flatten config and dataset manifest values into MLflow parameters."""

    values = config.as_dict()
    params: dict[str, Any] = {
        "dataset_label": dataset_label,
        "dataset_dir": str(dataset_dir),
        "dataset_repo": dataset_repo or "",
        "method": method,
        "base_model": config.model_name(),
        "output_dir": str(output_dir),
        "hf_adapter_repo": hub_model_id or "",
        "manifest.generated_at": manifest.get("generated_at", ""),
        "manifest.target_counts": _json_param(manifest.get("target_counts", {})),
        "manifest.actual_counts": _json_param(manifest.get("actual_counts", {})),
        "manifest.split_counts": _json_param(manifest.get("split_counts", {})),
        "manifest.language_counts": _json_param(manifest.get("language_counts", {})),
        "manifest.source_counts": _json_param(manifest.get("source_counts", {})),
        "manifest.content_hashes": _json_param(manifest.get("content_hashes", {})),
        "manifest.rejected_counts": _json_param(manifest.get("rejected_counts", {})),
    }
    for section_name in ("training", "lora"):
        for key, value in values.get(section_name, {}).items():
            if _is_mlflow_scalar(value):
                params[f"{section_name}.{key}"] = value
    adapter_path = values.get("model", {}).get("adapter_path")
    if adapter_path:
        params["model.adapter_path"] = adapter_path
    return params


def configure_mlflow(
    *, tracking_uri: str = DEFAULT_TRACKING_URI, experiment_name: str = DEFAULT_EXPERIMENT_NAME
) -> Any:
    """Import and configure MLflow for a training run."""

    try:
        import mlflow
    except ImportError as exc:  # pragma: no cover - depends on optional training extra
        raise RuntimeError("MLflow is required: run `uv sync --extra training`") from exc
    mlflow.set_tracking_uri(tracking_uri)
    mlflow.set_experiment(experiment_name)
    return mlflow


def log_training_metadata(
    mlflow: Any,
    *,
    params: dict[str, Any],
    config: TrainingConfig,
    config_artifact_name: str,
    dataset_dir: Path,
) -> None:
    """Log training parameters, config, and dataset metadata artifacts to MLflow."""

    mlflow.log_params(params)
    mlflow.log_dict(config.as_dict(), config_artifact_name)
    for filename in ("manifest.json", "README.md", "audit_report.json"):
        path = dataset_dir / filename
        if path.exists():
            mlflow.log_artifact(str(path), artifact_path="dataset")


def _json_param(value: Any) -> str:
    """Serialize nested metadata for MLflow's scalar parameter store."""

    return json.dumps(value, ensure_ascii=False, sort_keys=True)


def _is_mlflow_scalar(value: Any) -> bool:
    """Return whether a value can be logged directly as an MLflow parameter."""

    return isinstance(value, str | int | float | bool)

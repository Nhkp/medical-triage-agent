# /// script
# dependencies = ["datasets>=4.0.0", "peft>=0.17.0", "pyyaml>=6.0.0", "transformers>=4.56.0", "trl>=1.0.0", "bitsandbytes>=0.43.0", "accelerate>=1.0.0", "mlflow>=3.0.0", "pyngrok>=7.0.0"]
# ///
from __future__ import annotations

import argparse
import json
import os
import signal
import subprocess
import sys
import time
from pathlib import Path
from typing import Any
from urllib.error import URLError
from urllib.request import urlopen

sys.path.insert(0, str(Path(__file__).resolve().parents[1] / "src"))
sys.path.insert(0, str(Path(__file__).resolve().parent))

import train_dpo
import train_sft

from medical_triage_agent.configuration import TrainingConfig, load_training_config
from medical_triage_agent.tracking import (
    DEFAULT_EXPERIMENT_NAME,
    DEFAULT_TRACKING_URI,
    configure_mlflow,
    load_manifest,
    log_training_metadata,
    prepare_mlflow_params,
)


def main() -> int:
    """Run the tracked SFT then DPO workflow with MLflow metadata."""

    args = _parse_args()
    manifest = load_manifest(args.dataset_dir, required=not args.dry_run)
    sft_config = _load_sft_config(args, require_files=not args.dry_run)
    dpo_config = _load_dpo_config(args, require_files=not args.dry_run)
    summary = _dry_run_summary(args, sft_config, dpo_config, manifest)
    if args.dry_run:
        print(json.dumps(summary, ensure_ascii=False, indent=2))
        return 0

    ui_process: subprocess.Popen[str] | None = None
    try:
        if args.mlflow_ngrok:
            ui_process = _start_mlflow_ui(args)
            _wait_for_mlflow_ui(args.mlflow_ui_host, args.mlflow_ui_port, args.mlflow_ui_timeout)
            print(f"MLflow UI: {_open_ngrok_tunnel(args.mlflow_ui_port)}", flush=True)

        mlflow = configure_mlflow(
            tracking_uri=args.mlflow_tracking_uri,
            experiment_name=args.mlflow_experiment_name,
        )
        with mlflow.start_run(run_name=args.dataset_label):
            mlflow.log_params(
                {
                    "dataset_label": args.dataset_label,
                    "dataset_dir": str(args.dataset_dir),
                    "dataset_repo": args.dataset_repo or "",
                    "workflow": "sft+dpo",
                }
            )
            with mlflow.start_run(run_name="sft", nested=True):
                _log_stage_metadata(
                    mlflow,
                    args=args,
                    method="sft",
                    config=sft_config,
                    output_dir=args.sft_output_dir,
                    hub_model_id=args.sft_hub_model_id,
                    manifest=manifest,
                )
                train_sft.run_training(
                    sft_config,
                    max_train_samples=args.max_train_samples,
                    resume=args.sft_resume_from_checkpoint,
                )
            with mlflow.start_run(run_name="dpo", nested=True):
                _log_stage_metadata(
                    mlflow,
                    args=args,
                    method="dpo",
                    config=dpo_config,
                    output_dir=args.dpo_output_dir,
                    hub_model_id=args.dpo_hub_model_id,
                    manifest=manifest,
                )
                train_dpo.run_training(
                    dpo_config,
                    max_train_samples=args.max_train_samples,
                    resume=args.dpo_resume_from_checkpoint,
                )
    finally:
        if ui_process is not None:
            _terminate(ui_process)
    return 0


def _load_sft_config(args: argparse.Namespace, *, require_files: bool) -> TrainingConfig:
    """Load the effective SFT config for the selected dataset directory."""

    return load_training_config(
        args.sft_config,
        method="sft",
        overrides={
            **_dataset_overrides(args.dataset_dir, "sft"),
            **_training_overrides(
                output_dir=args.sft_output_dir,
                max_steps=args.max_steps,
                push_to_hub=args.push_to_hub,
                hub_model_id=args.sft_hub_model_id,
            ),
        },
        require_files=require_files,
    )


def _load_dpo_config(args: argparse.Namespace, *, require_files: bool) -> TrainingConfig:
    """Load the effective DPO config, chaining from the SFT output adapter."""

    return load_training_config(
        args.dpo_config,
        method="dpo",
        overrides={
            **_dataset_overrides(args.dataset_dir, "dpo"),
            "model.adapter_path": str(args.sft_output_dir),
            **_training_overrides(
                output_dir=args.dpo_output_dir,
                max_steps=args.max_steps,
                push_to_hub=args.push_to_hub,
                hub_model_id=args.dpo_hub_model_id,
            ),
        },
        require_files=require_files,
    )


def _dataset_overrides(dataset_dir: Path, kind: str) -> dict[str, Any]:
    """Point a config at generated split files for one dataset kind."""

    return {
        "data.train_file": str(dataset_dir / f"{kind}_train.jsonl"),
        "data.validation_file": str(dataset_dir / f"{kind}_validation.jsonl"),
        "data.test_file": str(dataset_dir / f"{kind}_test.jsonl"),
    }


def _training_overrides(
    *,
    output_dir: Path,
    max_steps: int | None,
    push_to_hub: bool,
    hub_model_id: str | None,
) -> dict[str, Any]:
    """Build common training overrides for experiment output and tracking."""

    overrides: dict[str, Any] = {
        "training.output_dir": str(output_dir),
        "training.report_to": "mlflow",
    }
    if max_steps is not None:
        overrides["training.max_steps"] = max_steps
    if push_to_hub:
        overrides["training.push_to_hub"] = True
        overrides["training.hub_model_id"] = hub_model_id
    return overrides


def _log_stage_metadata(
    mlflow: Any,
    *,
    args: argparse.Namespace,
    method: str,
    config: TrainingConfig,
    output_dir: Path,
    hub_model_id: str | None,
    manifest: dict[str, Any],
) -> None:
    """Log one training stage's effective config and dataset metadata."""

    log_training_metadata(
        mlflow,
        params=prepare_mlflow_params(
            dataset_label=args.dataset_label,
            dataset_dir=args.dataset_dir,
            dataset_repo=args.dataset_repo,
            method=method,
            config=config,
            output_dir=output_dir,
            hub_model_id=hub_model_id,
            manifest=manifest,
        ),
        config=config,
        config_artifact_name=f"{method}_effective_config.json",
        dataset_dir=args.dataset_dir,
    )


def _dry_run_summary(
    args: argparse.Namespace,
    sft_config: TrainingConfig,
    dpo_config: TrainingConfig,
    manifest: dict[str, Any],
) -> dict[str, Any]:
    """Return planned experiment settings without starting MLflow or training."""

    return {
        "dataset_label": args.dataset_label,
        "dataset_dir": str(args.dataset_dir),
        "dataset_repo": args.dataset_repo,
        "mlflow_tracking_uri": args.mlflow_tracking_uri,
        "mlflow_experiment_name": args.mlflow_experiment_name,
        "mlflow_ngrok": args.mlflow_ngrok,
        "mlflow_ui": _mlflow_ui_command(args) if args.mlflow_ngrok else None,
        "manifest_found": bool(manifest),
        "manifest_actual_counts": manifest.get("actual_counts", {}),
        "sft": {
            "train_file": sft_config.section("data")["train_file"],
            "validation_file": sft_config.section("data")["validation_file"],
            "output_dir": str(sft_config.output_dir()),
            "report_to": sft_config.section("training").get("report_to"),
            "push_to_hub": sft_config.section("training").get("push_to_hub"),
            "hub_model_id": sft_config.section("training").get("hub_model_id"),
        },
        "dpo": {
            "train_file": dpo_config.section("data")["train_file"],
            "validation_file": dpo_config.section("data")["validation_file"],
            "adapter_path": dpo_config.section("model").get("adapter_path"),
            "output_dir": str(dpo_config.output_dir()),
            "report_to": dpo_config.section("training").get("report_to"),
            "push_to_hub": dpo_config.section("training").get("push_to_hub"),
            "hub_model_id": dpo_config.section("training").get("hub_model_id"),
        },
    }


def _mlflow_ui_command(args: argparse.Namespace) -> list[str]:
    """Build the local MLflow UI command used before opening an ngrok tunnel."""

    return [
        sys.executable,
        "-m",
        "mlflow",
        "ui",
        "--backend-store-uri",
        args.mlflow_tracking_uri,
        "--host",
        args.mlflow_ui_host,
        "--port",
        str(args.mlflow_ui_port),
    ]


def _start_mlflow_ui(args: argparse.Namespace) -> subprocess.Popen[str]:
    """Start MLflow UI as a child process for live experiment inspection."""

    command = _mlflow_ui_command(args)
    print(f"starting MLflow UI: {' '.join(command)}", flush=True)
    return subprocess.Popen(command, text=True)


def _wait_for_mlflow_ui(host: str, port: int, timeout: int) -> None:
    """Wait until the local MLflow UI accepts HTTP requests."""

    url = f"http://{host}:{port}"
    deadline = time.monotonic() + timeout
    last_error = ""
    while time.monotonic() < deadline:
        try:
            with urlopen(url, timeout=5) as response:
                if 200 <= response.status < 500:
                    return
        except (OSError, TimeoutError, URLError) as exc:
            last_error = str(exc)
        time.sleep(1)
    raise TimeoutError(f"MLflow UI did not become ready at {url}: {last_error}")


def _open_ngrok_tunnel(port: int) -> str:
    """Open an ngrok HTTP tunnel to the local MLflow UI port."""

    try:
        ngrok = __import__("pyngrok.ngrok", fromlist=["ngrok"])
    except ImportError as exc:
        raise RuntimeError("pyngrok is required for --mlflow-ngrok: pip install pyngrok") from exc
    token = os.environ.get("NGROK_AUTHTOKEN")
    if token:
        ngrok.set_auth_token(token)
    return str(ngrok.connect(port, "http"))


def _terminate(process: subprocess.Popen[str]) -> None:
    """Terminate the MLflow UI child process after the experiment ends."""

    if process.poll() is not None:
        return
    process.send_signal(signal.SIGTERM)
    try:
        process.wait(timeout=10)
    except subprocess.TimeoutExpired:
        process.kill()


def _parse_args() -> argparse.Namespace:
    """Parse CLI arguments for tracked SFT+DPO experiments."""

    parser = argparse.ArgumentParser(description="Run tracked SFT+DPO experiments with MLflow")
    parser.add_argument("--dataset-label", required=True)
    parser.add_argument("--dataset-dir", type=Path, required=True)
    parser.add_argument("--dataset-repo")
    parser.add_argument("--sft-output-dir", type=Path)
    parser.add_argument("--dpo-output-dir", type=Path)
    parser.add_argument("--sft-config", default="configs/sft.yaml")
    parser.add_argument("--dpo-config", default="configs/dpo.yaml")
    parser.add_argument("--mlflow-tracking-uri", default=DEFAULT_TRACKING_URI)
    parser.add_argument("--mlflow-experiment-name", default=DEFAULT_EXPERIMENT_NAME)
    parser.add_argument("--max-steps", type=int)
    parser.add_argument("--max-train-samples", type=int)
    parser.add_argument("--dry-run", action="store_true")
    parser.add_argument("--push-to-hub", action="store_true")
    parser.add_argument("--sft-hub-model-id")
    parser.add_argument("--dpo-hub-model-id")
    parser.add_argument("--sft-resume-from-checkpoint")
    parser.add_argument("--dpo-resume-from-checkpoint")
    parser.add_argument("--mlflow-ngrok", action="store_true", help="Expose MLflow UI with ngrok")
    parser.add_argument("--mlflow-ui-host", default="127.0.0.1")
    parser.add_argument("--mlflow-ui-port", type=int, default=5000)
    parser.add_argument("--mlflow-ui-timeout", type=int, default=60)
    args = parser.parse_args()
    args.sft_output_dir = (
        args.sft_output_dir or Path("outputs/experiments") / args.dataset_label / "sft"
    )
    args.dpo_output_dir = (
        args.dpo_output_dir or Path("outputs/experiments") / args.dataset_label / "dpo"
    )
    if args.push_to_hub and (not args.sft_hub_model_id or not args.dpo_hub_model_id):
        parser.error("--push-to-hub requires --sft-hub-model-id and --dpo-hub-model-id")
    return args


if __name__ == "__main__":
    raise SystemExit(main())

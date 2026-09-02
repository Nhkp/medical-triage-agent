# Default local dataset folder used by the generic data-build/data-ready targets.
DATA_OUT ?= data/processed/training

# Separate local folders for downloaded 5k and 8k published Hugging Face datasets.
DATA_OUT_5K ?= data/processed/training-5k
DATA_OUT_8K ?= data/processed/training-8000

# Generic local dataset generation targets. Override these when rebuilding DATA_OUT.
SFT_TARGET ?= 5000
DPO_TARGET ?= 1000
SFT_MEDIQA ?= 2000
SFT_FRENCHMEDMCQA ?= 1000

# Published Hugging Face dataset repositories used by the 5k/8k experiment pipelines.
HF_DATASET_REPO_5K ?= Lokhidor/medical-triage-dataset
HF_DATASET_REPO_8K ?= Lokhidor/medical-triage-dataset-8k

# Hugging Face adapter repositories used only when running with PUSH_TO_HUB=1.
SFT_HUB_MODEL_ID_5K ?= Lokhidor/medical-triage-qwen3-sft-lora-5k
DPO_HUB_MODEL_ID_5K ?= Lokhidor/medical-triage-qwen3-dpo-lora-5k
SFT_HUB_MODEL_ID_8K ?= Lokhidor/medical-triage-qwen3-sft-lora-8k
DPO_HUB_MODEL_ID_8K ?= Lokhidor/medical-triage-qwen3-dpo-lora-8k

# Local MLflow tracking configuration for experiment metadata and Trainer metrics.
MLFLOW_TRACKING_URI ?= mlruns
MLFLOW_EXPERIMENT_NAME ?= medical-triage-agent
MLFLOW_UI_PORT ?= 5000
MLFLOW_NGROK ?= 0
MLFLOW_NGROK_ARG = $(if $(filter 1 true yes,$(MLFLOW_NGROK)),--mlflow-ngrok --mlflow-ui-port $(MLFLOW_UI_PORT),)

# Set PUSH_TO_HUB=1, true, or yes to publish adapters during train-5k/train-8k.
PUSH_TO_HUB ?= 0
PUSH_TO_HUB_ARG = $(if $(filter 1 true yes,$(PUSH_TO_HUB)),--push-to-hub,)

# Presentation export inputs and output path.
PRESENTATION_HTML ?= presentations/chsa-current-state/index.html
PRESENTATION_OUT ?= dist/presentations/chsa-current-state.pptx

# Optional API URL used by evaluation targets when testing a running service.
API_URL ?=

.PHONY: sync sync-training sync-ui sync-presentation check lint format type test hooks hooks-run data-build data-audit data-card data-summary data-ready data-clean data-pull-5k data-pull-8k data-audit-5k data-audit-8k data-summary-5k data-summary-8k train-sft-smoke train-dpo-smoke train-grpo-smoke train-5k train-8k pipeline-5k pipeline-8k pipeline-all mlflow-ui eval-models eval-model-comparison-dry-run eval-model-comparison serve-local serve-api serve-ui serve-colab-dry-run serve-kaggle-dry-run eval-latency eval-robustness step3-ready presentation-browser presentation-html presentation-pptx presentation-ready clean

# Install the default project dependencies.
sync:
	uv sync

# Install training dependencies, including datasets, TRL, PEFT, and MLflow.
sync-training:
	uv sync --extra training

# Install Streamlit UI dependencies.
sync-ui:
	uv sync --extra ui

# Install presentation export dependencies and Playwright browser binaries.
sync-presentation:
	uv sync --extra presentation
	uv run --extra presentation python -m playwright install chromium

# Run the full local quality gate used by CI.
check:
	scripts/check.sh

# Run Ruff lint checks without rewriting files.
lint:
	uv run ruff check .

# Format Python files with Ruff.
format:
	uv run ruff format .

# Run strict mypy checks on source and tests.
type:
	uv run mypy src tests

# Run the pytest suite.
test:
	uv run pytest

# Install local pre-commit and commit-msg hooks.
hooks:
	uv run pre-commit install --hook-type pre-commit --hook-type commit-msg

# Run all pre-commit hooks against the full repository.
hooks-run:
	uv run pre-commit run --all-files

# Build a local training dataset under DATA_OUT using the configured target counts.
data-build:
	uv run python -m medical_triage_agent build-training-data $(DATA_OUT) --sft-target $(SFT_TARGET) --dpo-target $(DPO_TARGET) --sft-mediqa $(SFT_MEDIQA) --sft-frenchmedmcqa $(SFT_FRENCHMEDMCQA)

# Audit the generated local DATA_OUT training dataset.
data-audit:
	uv run python -m medical_triage_agent audit-training-data $(DATA_OUT)

# Regenerate the dataset card README from DATA_OUT/manifest.json.
data-card:
	uv run python -m medical_triage_agent make-dataset-card $(DATA_OUT)/manifest.json $(DATA_OUT)/README.md

# Print a JSON summary of DATA_OUT counts, splits, sources, and files.
data-summary:
	uv run python -m medical_triage_agent summarize-training-data $(DATA_OUT)

# Build, audit, document, and summarize the generic local training dataset.
data-ready: sync-training data-build data-audit data-card data-summary

# Remove the generic generated DATA_OUT folder.
data-clean:
	rm -rf $(DATA_OUT)

# Download the published 5k dataset from Hugging Face into DATA_OUT_5K.
data-pull-5k: sync-training
	uv run hf download $(HF_DATASET_REPO_5K) --type dataset --local-dir $(DATA_OUT_5K)

# Download the published 8k dataset from Hugging Face into DATA_OUT_8K.
data-pull-8k: sync-training
	uv run hf download $(HF_DATASET_REPO_8K) --type dataset --local-dir $(DATA_OUT_8K)

# Audit the local 5k dataset folder.
data-audit-5k:
	uv run python -m medical_triage_agent audit-training-data $(DATA_OUT_5K)

# Audit the local 8k dataset folder.
data-audit-8k:
	uv run python -m medical_triage_agent audit-training-data $(DATA_OUT_8K)

# Print a summary for the local 5k dataset folder.
data-summary-5k:
	uv run python -m medical_triage_agent summarize-training-data $(DATA_OUT_5K)

# Print a summary for the local 8k dataset folder.
data-summary-8k:
	uv run python -m medical_triage_agent summarize-training-data $(DATA_OUT_8K)

# Validate SFT training startup without loading model weights.
train-sft-smoke:
	uv run scripts/train_sft.py --config configs/sft.yaml --max-steps 5 --dry-run

# Validate DPO training startup without loading model weights.
train-dpo-smoke:
	uv run scripts/train_dpo.py --config configs/dpo.yaml --max-steps 5 --dry-run

# Validate optional GRPO training startup without loading model weights.
train-grpo-smoke:
	uv run scripts/train_grpo.py --config configs/grpo.yaml --max-steps 5 --dry-run

# Train the 5k SFT+DPO MLflow experiment from DATA_OUT_5K.
train-5k: sync-training
	uv run scripts/train_experiment.py --dataset-label dataset-5k --dataset-dir $(DATA_OUT_5K) --dataset-repo $(HF_DATASET_REPO_5K) --mlflow-tracking-uri $(MLFLOW_TRACKING_URI) --mlflow-experiment-name $(MLFLOW_EXPERIMENT_NAME) --sft-hub-model-id $(SFT_HUB_MODEL_ID_5K) --dpo-hub-model-id $(DPO_HUB_MODEL_ID_5K) $(PUSH_TO_HUB_ARG) $(MLFLOW_NGROK_ARG)

# Train the 8k SFT+DPO MLflow experiment from DATA_OUT_8K.
train-8k: sync-training
	uv run scripts/train_experiment.py --dataset-label dataset-8k --dataset-dir $(DATA_OUT_8K) --dataset-repo $(HF_DATASET_REPO_8K) --mlflow-tracking-uri $(MLFLOW_TRACKING_URI) --mlflow-experiment-name $(MLFLOW_EXPERIMENT_NAME) --sft-hub-model-id $(SFT_HUB_MODEL_ID_8K) --dpo-hub-model-id $(DPO_HUB_MODEL_ID_8K) $(PUSH_TO_HUB_ARG) $(MLFLOW_NGROK_ARG)

# Download, audit, summarize, and train the 5k MLflow experiment.
pipeline-5k: sync-training data-pull-5k data-audit-5k data-summary-5k train-5k

# Download, audit, summarize, and train the 8k MLflow experiment.
pipeline-8k: sync-training data-pull-8k data-audit-8k data-summary-8k train-8k

# Run both 5k and 8k pipelines sequentially for comparison.
pipeline-all: pipeline-5k pipeline-8k

# Start the local MLflow UI for inspecting experiment runs.
mlflow-ui: sync-training
	uv run --extra training mlflow ui --backend-store-uri $(MLFLOW_TRACKING_URI)

# Dry-run the base model evaluation startup.
eval-models:
	uv run scripts/evaluate.py --config configs/sft.yaml --model base --dry-run

# Dry-run model comparison output generation without calling a live API.
eval-model-comparison-dry-run:
	uv run python scripts/evaluate_model_comparison.py --dry-run

# Run model comparison against API_URL when provided, otherwise use the script default.
eval-model-comparison:
	uv run python scripts/evaluate_model_comparison.py $(if $(API_URL),--url $(API_URL),)

# Start the GPU Docker Compose stack for local vLLM plus FastAPI serving.
serve-local:
	docker compose --profile gpu up --build

# Start the FastAPI app locally without Docker.
serve-api:
	uv run --extra serving uvicorn medical_triage_agent.api:create_app --factory --host 0.0.0.0 --port 8080

# Start the Streamlit UI locally.
serve-ui:
	uv run --extra ui streamlit run src/medical_triage_agent/streamlit_interface.py

# Validate the Colab serving command construction without starting vLLM.
serve-colab-dry-run:
	uv run python scripts/serve_colab.py --dry-run

# Validate the Kaggle serving command construction without starting vLLM.
serve-kaggle-dry-run:
	uv run python scripts/serve_colab.py --dry-run

# Run latency evaluation against API_URL when provided.
eval-latency:
	uv run python scripts/evaluate_latency.py $(if $(API_URL),--url $(API_URL),)

# Run robustness evaluation against API_URL when provided.
eval-robustness:
	uv run python scripts/evaluate_robustness.py $(if $(API_URL),--url $(API_URL),)

# Run the full step 3 readiness gate: checks plus robustness and latency.
step3-ready: check eval-robustness eval-latency

# Dry-run presentation export from HTML.
presentation-html:
	uv run python scripts/export_presentation.py --input $(PRESENTATION_HTML) --dry-run

# Install the Playwright browser needed for rendered presentation export.
presentation-browser:
	uv run --extra presentation python -m playwright install chromium

# Export the HTML presentation to a rendered PPTX file.
presentation-pptx: presentation-browser
	uv run --extra presentation python scripts/export_presentation.py --input $(PRESENTATION_HTML) --output $(PRESENTATION_OUT) --mode rendered

# Validate both dry-run and rendered presentation export paths.
presentation-ready: presentation-html presentation-pptx

# Remove Python cache, test cache, type cache, lint cache, and coverage artifacts.
clean:
	find . -type d \( -name __pycache__ -o -name .pytest_cache -o -name .mypy_cache -o -name .ruff_cache \) -prune -exec rm -rf {} +
	rm -rf .coverage htmlcov

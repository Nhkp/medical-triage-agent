DATA_OUT ?= data/processed/training
SFT_TARGET ?= 5000
DPO_TARGET ?= 1000
SFT_MEDIQA ?= 2000
SFT_FRENCHMEDMCQA ?= 1000

PRESENTATION_HTML ?= presentations/chsa-current-state/index.html
PRESENTATION_OUT ?= dist/presentations/chsa-current-state.pptx

API_URL ?=

.PHONY: sync sync-training sync-presentation check lint format type test hooks hooks-run data-build data-audit data-card data-summary data-ready data-clean train-sft-smoke train-dpo-smoke train-grpo-smoke eval-models serve-local serve-api serve-colab-dry-run eval-latency eval-robustness step3-ready presentation-html presentation-pptx presentation-ready clean

sync:
	uv sync

sync-training:
	uv sync --extra training

sync-presentation:
	uv sync --extra presentation

check:
	scripts/check.sh

lint:
	uv run ruff check .

format:
	uv run ruff format .

type:
	uv run mypy src tests

test:
	uv run pytest

hooks:
	uv run pre-commit install --hook-type pre-commit --hook-type commit-msg

hooks-run:
	uv run pre-commit run --all-files

data-build:
	uv run python -m medical_triage_agent build-training-data $(DATA_OUT) --sft-target $(SFT_TARGET) --dpo-target $(DPO_TARGET) --sft-mediqa $(SFT_MEDIQA) --sft-frenchmedmcqa $(SFT_FRENCHMEDMCQA)

data-audit:
	uv run python -m medical_triage_agent audit-training-data $(DATA_OUT)

data-card:
	uv run python -m medical_triage_agent make-dataset-card $(DATA_OUT)/manifest.json $(DATA_OUT)/README.md

data-summary:
	uv run python -m medical_triage_agent summarize-training-data $(DATA_OUT)

data-ready: sync-training data-build data-audit data-card data-summary

data-clean:
	rm -rf $(DATA_OUT)

train-sft-smoke:
	uv run scripts/train_sft.py --config configs/sft_kaggle.yaml --max-steps 5 --dry-run

train-dpo-smoke:
	uv run scripts/train_dpo.py --config configs/dpo_kaggle.yaml --max-steps 5 --dry-run

train-grpo-smoke:
	uv run scripts/train_grpo.py --config configs/grpo_kaggle.yaml --max-steps 5 --dry-run

eval-models:
	uv run scripts/evaluate.py --config configs/sft_kaggle.yaml --model base --dry-run

serve-local:
	docker compose --profile gpu up --build

serve-api:
	uv run --extra serving uvicorn medical_triage_agent.api:create_app --factory --host 0.0.0.0 --port 8080

serve-colab-dry-run:
	uv run python scripts/serve_colab.py --dry-run

eval-latency:
	uv run python scripts/evaluate_latency.py $(if $(API_URL),--url $(API_URL),)

eval-robustness:
	uv run python scripts/evaluate_robustness.py $(if $(API_URL),--url $(API_URL),)

step3-ready: check eval-robustness eval-latency

presentation-html:
	uv run python scripts/export_presentation.py --input $(PRESENTATION_HTML) --dry-run

presentation-pptx:
	uv run --extra presentation python scripts/export_presentation.py --input $(PRESENTATION_HTML) --output $(PRESENTATION_OUT)

presentation-ready: presentation-html presentation-pptx

clean:
	find . -type d \( -name __pycache__ -o -name .pytest_cache -o -name .mypy_cache -o -name .ruff_cache \) -prune -exec rm -rf {} +
	rm -rf .coverage htmlcov

DATA_OUT ?= data/processed/training
SFT_TARGET ?= 5000
DPO_TARGET ?= 1000
SFT_MEDIQA ?= 2000
SFT_FRENCHMEDMCQA ?= 1000

.PHONY: sync sync-training check lint format type test hooks hooks-run data-build data-audit data-card data-summary data-ready data-clean clean

sync:
	uv sync

sync-training:
	uv sync --extra training

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

clean:
	find . -type d \( -name __pycache__ -o -name .pytest_cache -o -name .mypy_cache -o -name .ruff_cache \) -prune -exec rm -rf {} +
	rm -rf .coverage htmlcov

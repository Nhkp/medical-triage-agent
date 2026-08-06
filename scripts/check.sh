#!/usr/bin/env bash
set -euo pipefail

uv run ruff check .
uv run ruff format --check .
uv run python -m medical_triage_agent sources
uv run python -m medical_triage_agent evaluate-safety
uv run mypy src tests
uv run python -m py_compile scripts/train_sft.py scripts/train_dpo.py scripts/train_grpo.py scripts/evaluate.py scripts/export_presentation.py
uv run pytest

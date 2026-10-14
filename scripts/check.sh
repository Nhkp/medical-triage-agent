#!/usr/bin/env bash
set -euo pipefail

uv run ruff check .
uv run ruff format --check .
uv run python -m medical_triage_agent sources
uv run python -m medical_triage_agent evaluate-safety
uv run mypy src tests
uv run python -m compileall -q src deploy scripts
uv run pytest

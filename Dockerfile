FROM python:3.12-slim

WORKDIR /app

COPY pyproject.toml uv.lock README.md ./
COPY src ./src

RUN pip install --no-cache-dir uv \
    && uv sync --frozen --extra serving --no-dev

EXPOSE 8080

CMD ["uv", "run", "uvicorn", "medical_triage_agent.api:create_app", "--factory", "--host", "0.0.0.0", "--port", "8080"]

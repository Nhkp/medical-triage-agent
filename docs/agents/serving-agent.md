# Serving agent

## Mission

Maintain the FastAPI wrapper, vLLM serving contract, Docker image, and audit behavior.

## Rules

- Keep vLLM as the model server and FastAPI as a thin domain wrapper.
- Do not log raw patient text in audit responses.
- Require an API key for deployed demo endpoints.
- Keep health checks lightweight.
- Containerize serving only; training stays in Hugging Face Jobs.

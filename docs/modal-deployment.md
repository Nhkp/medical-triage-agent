# Modal deployment

This deployment runs the FastAPI triage API publicly on Modal and starts vLLM privately in
the same GPU container.

## Prerequisites

- Push the 8k DPO LoRA adapter to `Lokhidor/medical-triage-qwen3-dpo-lora-8k`.
- Install local tools with `uv sync --extra serving`.
- Authenticate Modal with `uv run modal setup`.

## Configure `.env`

Copy `.env.example` to `.env`, then set:

```dotenv
HF_TOKEN=...
TRIAGE_API_TOKEN=...
```

The deploy helper refreshes Modal secrets from `.env`; do not commit `.env`.

## Deploy

```bash
make deploy-dry-run
make deploy
```

The deployed app exposes FastAPI only. vLLM stays internal at `http://127.0.0.1:8000/v1`.

## Smoke test

```bash
curl -s "$MODAL_URL/health"

curl -s -X POST "$MODAL_URL/triage" \
  -H "Authorization: Bearer $TRIAGE_API_TOKEN" \
  -H "Content-Type: application/json" \
  -d '{"symptoms":["douleur thoracique"]}'
```

The triage response should include `llm_status` as `accepted` or `accepted_repaired`.

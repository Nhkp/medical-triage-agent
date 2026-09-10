# Modal deployment

This deployment runs the FastAPI triage API publicly on Modal and starts vLLM privately in
the same GPU container.

## Prerequisites

- Push the 8k DPO LoRA adapter to `Lokhidor/medical-triage-qwen3-dpo-lora-8k`.
- Install local tools with `uv sync --extra serving`.
- Authenticate Modal with `uv run modal setup`.
- Review the pinned serving artifacts before deployment:
  - Base model revision: `ea980cb0a6c2ae4b936e82123acc929f1cec04c1`.
  - LoRA adapter revision: `9f1c83f91064a8b464bc9e8b87f91231c617b2e7`.

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
The base model is pinned in the vLLM command. The LoRA adapter revision is documented
above because vLLM's static `--lora-modules` spec does not expose a separate revision
field.
The Modal image pins Transformers to a 4.x release because vLLM 0.10.2 reads tokenizer
attributes that are absent from newer Transformers 5.x tokenizers.

## Smoke test

```bash
curl -s "$MODAL_URL/health"

curl -s -X POST "$MODAL_URL/triage" \
  -H "Authorization: Bearer $TRIAGE_API_TOKEN" \
  -H "Content-Type: application/json" \
  -d '{"symptoms":["douleur thoracique"]}'
```

The triage response should include `llm_status` as `accepted` or `accepted_repaired`.

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

## Manual GitHub Actions deployment

The `deploy-demo` workflow has only `workflow_dispatch`: tags and pushes do not deploy.
It installs the locked base environment, runs `make check`, builds the Docker image,
and validates Compose before calling the existing Modal helper. Deployment failures
stop the workflow. A concurrency group serializes deployments without cancelling a run.

In the repository's **Settings → Secrets and variables → Actions**, configure:

| kind | name | purpose |
| --- | --- | --- |
| Secret | `MODAL_TOKEN_ID` | Modal account/service token identifier |
| Secret | `MODAL_TOKEN_SECRET` | Modal account/service token secret |
| Secret | `HF_TOKEN` | read access to the selected adapter and model |
| Secret | `TRIAGE_API_TOKEN` | bearer token required for demo API requests |
| Variable | `MODAL_URL` | HTTPS root URL for this app's `fastapi_app` endpoint |

Use the Modal account's main environment, matching local deployment defaults. Copy the
app's stable endpoint URL from Modal into `MODAL_URL` before launching the workflow.
If the endpoint is not yet allocated, establish it with `make deploy` first. GitHub's
Modal credentials are provided through environment variables; interactive `modal setup`
is not needed in CI. The helper receives HF/API values through an owner-only temporary
file, removed on success or failure, with no secret values in command arguments.

After publishing the workflow to GitHub, open **Actions → deploy-demo → Run workflow**
and select the intended revision. The final step probes `/health` for at most fifteen
minutes. This may start a GPU container and incur Modal usage. HTTP 200 is a deployment
smoke check; it does not replace model-backed robustness, latency, or clinical evaluation.

Retain the successful Actions run URL and date in the report as deployment evidence.
No successful live run of this new workflow has been recorded during implementation.
Changing a secret here also requires updating its counterpart in local `.env` when local
deployment is used; the helper refreshes the two named Modal secrets on each deployment.

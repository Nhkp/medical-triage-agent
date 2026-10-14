# Modal deployment

This deployment runs the FastAPI triage API publicly on Modal and starts vLLM privately in
the same GPU container.

## Prerequisites

- Keep the evaluated pinned adapters available on Hugging Face; no new training or publication is needed.
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
Before serving, the final source resolves the Base/SFT/DPO snapshots recorded in
`configs/serving.json`, verifies their hashes and runs the shared tokenizer compatibility
qualification. It loads DPO from the local qualified copy and the campaign's explicit SFT
chat template. `/app/assets/serving_manifest.json` records the immutable revisions, hashes,
versions and parameters. Startup stops on mismatch; no original snapshot is modified.

The final implementation is locally tested, but the new source has not been deployed.
The owner confirmed on 14 October that old exposed tokens have not been revoked. Rotate
HF/API tokens and update local, GitHub, Modal and Streamlit secrets before deployment.
Do not mistake the successful 13 October run of commit `285d4c1` for this final revision.

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
The historical Actions run 38448371074 deployed commit 285d4c1 on 13 October; final-source deployment remains pending.
Changing a secret here also requires updating its counterpart in local `.env` when local
deployment is used; the helper refreshes the two named Modal secrets on each deployment.

## Final supervised qualification

After a successful deployment and health wake-up, use the six existing preflight situations
(FR/EN runny nose, chest pain and dosage request). Check the model alias, generation metadata,
JSON/schema indicators, explanation source, final emergency priority and retrieved audit.
Keep medically inadequate responses as evidence rather than adjusting the prompt.

Set `TRIAGE_API_TOKEN` privately in the process environment, then run the existing smoke tools:

```bash
API_URL=https://nhkp--medical-triage-agent-fastapi-app.modal.run
uv run python scripts/evaluate_robustness.py --url "$API_URL" --output outputs/evaluations/modal-robustness.json
uv run python scripts/evaluate_latency.py --url "$API_URL" --output outputs/evaluations/modal-latency.json
```

These are Modal L4 operational checks, separate from the Kaggle comparison. The UI locks
its API address whenever a server token is configured and refuses authenticated redirects.
Prepare a recorded cloud walkthrough only from observed requests; the supplied local
rule-only video is explicitly a deterministic fallback demonstration.

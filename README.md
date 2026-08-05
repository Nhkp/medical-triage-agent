# Medical triage agent

Proof of concept for a CHSA bilingual medical triage assistant. The project focuses on
dataset provenance, RGPD-minded anonymization, supervised fine-tuning, preference alignment,
and a vLLM-backed demo API.

This is not a production medical device and must keep human clinical review in the loop.

## Local setup

```bash
uv sync
make check
```

## Data preparation

Build the local SFT/DPO training dataset artifacts outside git:

```bash
make data-ready
```

Useful focused commands:

```bash
make data-audit
make data-summary
make data-clean
```

Optional serving dependencies:

```bash
uv sync --extra serving
```

Optional training dependencies:

```bash
uv sync --extra training
```

## Guardrails

- Project agent instructions: `AGENTS.md`
- Source intake policy: `docs/source-policy.md`
- Data source registry: `docs/data-sources.md`
- Medical safety policy: `docs/medical-safety.md`
- Privacy and RGPD policy: `docs/privacy-rgpd.md`
- Evaluation policy: `docs/evaluation.md`
- Global architecture: `docs/architecture.md`

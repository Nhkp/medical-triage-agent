# Agent instructions for medical-triage-agent

This repository builds a 4-week CHSA proof of concept for a bilingual medical triage
assistant. The assistant supports clinical staff during initial emergency triage; it is not
an autonomous diagnosis or treatment system.

## Working rules

- Read the real flow before changing code.
- Reuse existing tools, patterns, and helpers before creating new ones.
- Keep diffs small and centered on the request.
- Add a runnable test for any non-trivial logic.
- Follow `docs/code-guidelines.md`, `docs/commit-guidelines.md`, and
  `docs/git-workflow.md`.
- Use `docs/agents/readme-agent.md` when changing `README.md`.
- Use `docs/agents/presentation-agent.md` when changing presentation material.
- Follow source intake rules in `docs/source-policy.md`.
- Follow medical safety limits in `docs/medical-safety.md`.
- Follow privacy and RGPD rules in `docs/privacy-rgpd.md`.
- Document structural decisions in `docs/decisions/`.

## Expected commands

```bash
uv sync
make check
```

`make check` is the single local and CI quality gate.

## Medical safety boundaries

- Never present model output as a final diagnosis or treatment decision.
- Always preserve human-in-the-loop review for triage priority.
- Escalate red-flag symptoms to immediate clinical attention.
- Do not add features that hide uncertainty, provenance, or safety disclaimers.
- Do not tune prompts or models to maximize confidence at the expense of safety.

## Data and privacy boundaries

- Do not commit raw medical dumps, patient data, audit logs, model checkpoints, API tokens,
  or local Hugging Face caches.
- Every dataset source must be tracked in `docs/data-sources.md` before ingestion.
- Every generated training record must keep source IDs, license metadata, and transform
  history.
- Training, validation, test, and clinical evaluation splits must stay separated.
- Intentional shortcuts must use a `ponytail:` comment naming the ceiling and upgrade path.

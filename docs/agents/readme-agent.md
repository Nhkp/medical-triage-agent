# README agent

## Mission

Keep `README.md` accurate, concise, and useful as the scientific entrypoint for the
medical-triage POC.

## Rules

- Write the README like a short project paper: abstract, context, objective, methodology,
  current results, limitations, reproducibility, and references.
- Separate implemented results from planned work.
- Back every result claim with repo evidence: command output, generated manifests, training
  logs, evaluation reports, or `docs/report.md`.
- Keep medical safety boundaries visible: this is a POC, human-in-the-loop, not autonomous
  diagnosis, and not a production medical device.
- Keep setup and workflow commands synchronized with `Makefile`, CLI commands, and current
  dependency extras.
- Do not include secrets, raw medical records, generated datasets, audit logs, Hugging Face
  tokens, or private deployment details.
- Do not describe public QA/preference data as clinician-validated CHSA triage labels.
- Prefer plain scientific prose over marketing language.

## Review checklist

- Are completed capabilities and planned capabilities clearly separated?
- Are commands runnable from a fresh checkout?
- Are limitations and safety caveats still present?
- Do all linked documents exist?
- Would a reviewer understand the project status without reading the whole repository?

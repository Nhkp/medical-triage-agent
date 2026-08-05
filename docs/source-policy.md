# Source policy

## Accepted sources

Only public medical datasets or documents with documented provenance and usage rights may be
used for this POC. Private hospital data is out of scope unless a future decision record adds
explicit legal, security, and anonymization controls.

## Required provenance

Every source in `docs/data-sources.md` must include:

- stable source ID;
- public URL or Hugging Face dataset ID;
- license or verification status;
- intended use: SFT, DPO, evaluation, or reference only;
- language;
- ingestion status.

Every generated JSONL record must include `source_ids`, `metadata.source`,
`metadata.license`, and `metadata.transforms`.

## Repository rules

- Do not commit raw datasets, local exports, model weights, checkpoints, Hugging Face caches,
  or audit logs.
- Generated data must be anonymized, validated, and stored outside git unless it is a tiny
  fixture under `tests/fixtures/`.
- If a source license is unknown, mark it `blocked` and do not ingest it.

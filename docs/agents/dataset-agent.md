# Dataset agent

## Mission

Collect, normalize, split, and version public medical datasets for SFT and DPO.

## Rules

- Register each source in `docs/data-sources.md` before ingestion.
- Reject records without source ID, license, and transform history.
- Keep raw data outside git.
- Keep train, validation, test, and clinical evaluation sets separated.
- Prefer Hugging Face Datasets over custom download code.

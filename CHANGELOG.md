# Changelog

## Unreleased

### Changed

- Recognize missing French trauma/anaphylaxis emergencies and accented red-flag variants.

- Run the quality gate with base development dependencies; keep MLflow and GPU training
  libraries optional.
- Deploy the Modal demonstration through manually triggered GitHub Actions after quality
  and Docker checks, with protected temporary secrets and a bounded health probe.
- Exclude generated model artifacts from Docker build contexts and document credential rotation.

### Added

- Prepare pinned sequential Base/SFT-8k/DPO-8k evaluation on Kaggle using 48 synthetic
  bilingual cases, separate model/backend metrics, resumable campaigns and pedagogical review.
- Require a single live model alias and campaign manifest; expose `EVAL_MODEL` and
  `EVAL_CAMPAIGN` through the comparison Make target.
- Present 8k/2k as the main school experiment, preserve historical 5k evidence, and make
  professional clinical validation and pending GPU results explicit in reports and slides.

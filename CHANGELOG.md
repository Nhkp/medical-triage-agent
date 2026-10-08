# Changelog

## Unreleased

### Changed

- Accept safe complete single sentences after duplicate removal while preserving original
  previews and rejecting incomplete generations.
- Simplify the serving prompt and apply repetition penalty 1.1 with the shared 256-token
  ceiling.
- Record and verify the exact system prompt and repetition penalty in new Kaggle campaigns;
  align measured requests with preflight input language.

- Fix the model comparison ceiling at 256 tokens; retain length-limited generations
  as explicit model failures while permitting the campaign to measure them.
- Use rule fallback for `truncated_output` and preserve raw format and generation
  metadata without fabricating a completed model answer.

- Make the Kaggle token ceiling configurable and shared across manifest, preflight and
  all served models; retain the 256-token default and require new campaigns for changes.
- Diagnose preflight failures with case IDs and tokenizer log line numbers; avoid matching
  configuration keys as tokenizer errors.

- Recognize missing French trauma/anaphylaxis emergencies and accented red-flag variants.
- Fix vLLM 0.10.2 structured generation with first-call guided JSON, validated serving
  settings, a 256-token default, and audit metadata for accepted and rejected outputs.
- Separate input language from review quality, preserve annotations, and export repaired
  historical reviews separately with generation-length and missing-metadata indicators.
- Prepare immutable adapter compatibility copies with tokenizer equivalence checks
  and block Kaggle measurement on technical preflight failures.
- Document initial failed-serving evidence and pending corrected results in the README,
  school report and presentation.

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

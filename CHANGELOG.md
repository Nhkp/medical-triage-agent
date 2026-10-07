# Changelog

## Unreleased

### Changed

- Run the quality gate with base development dependencies; keep MLflow and GPU training
  libraries optional.
- Deploy the Modal demonstration through manually triggered GitHub Actions after quality
  and Docker checks, with protected temporary secrets and a bounded health probe.
- Exclude generated model artifacts from Docker build contexts and document credential rotation.

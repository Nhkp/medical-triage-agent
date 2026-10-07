# 0003 - School submission and manual Modal delivery

## Status

Accepted.

## Decision

Use the 8k/2k dataset as the main experiment and retain 5k/1k as a historical experiment.
Keep metrics attached to their original run and adapter references. Keep existing Makefile
commands and historical dataset-generation defaults.

Deploy the demonstration through a manual GitHub Actions workflow using the existing
Modal helper, after the base quality gate and Docker/Compose checks. Use GitHub secrets,
a protected temporary dotenv file, serialized deployments, and a bounded health probe.

Treat professional clinical validation as an explicit limitation of the school submission
and a prerequisite for any clinical pilot. Do not claim that technical tests or a review
queue satisfy the input's request for clinically validated DPO pairs.

## Consequences

- Base checks remain runnable without installing GPU training dependencies.
- A successful live workflow is required before claiming automated deployment evidence.
- Professional review and model-backed evaluation remain documented future gates.
- The owner must rotate the previously exposed local tokens before configuring CI secrets.

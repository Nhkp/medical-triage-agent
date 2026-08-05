# 0001 - CHSA medical triage POC scope

## Status

Accepted.

## Decision

This repository targets a 4-week proof of concept for a CHSA medical triage assistant. The
default platform is Hugging Face Hub/Jobs for dataset, model, training, and artifact
reproducibility. The target model is Qwen3-1.7B-Base with SFT, LoRA, and DPO alignment.

## Consequences

- The assistant is a staff-support tool, not an autonomous diagnosis system.
- Public datasets are the only v1 data sources until private data governance exists.
- Training jobs push artifacts to the Hugging Face Hub because job environments are
  ephemeral.
- Serving uses vLLM behind a thin FastAPI wrapper.
- The POC must expose safety limits and auditability in every triage response.

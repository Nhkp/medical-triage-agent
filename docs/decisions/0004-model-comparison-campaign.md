# 0004 - Pinned 8k model comparison campaign

## Status

Accepted.

## Decision

Reuse the existing Kaggle notebook and comparison script for sequential Base/SFT-8k/DPO-8k
serving. Evaluate only one alias per live invocation, verify wrapper and vLLM identity,
and retain immutable snapshot, dataset, code and runtime fingerprints in a campaign manifest.

Use 24 synthetic situations with French/English pairs, retain the original regression cases,
and separate accepted model suggestions from final rule-protected priorities. The shared
prompt includes the rule priority; the experiment does not measure independent clinical reasoning.

Use two warmups and three measured passes. Retain errors and fallback evidence, save completed
model artifacts atomically, and reject mixed campaigns. Review first-pass text pedagogically
without another LLM judge. Leave GPU results, annotations and model selection pending until
Kaggle artifacts are returned. Public API routes and clinical rules remain unchanged.

## Consequences

- GPU execution needs a user-controlled Kaggle runtime and renewed HF credentials.
- Local tests verify the evaluator, not model performance or professional clinical safety.
- A failed or inconclusive campaign cannot be represented as a successful model validation.
- Changed fingerprints require a new campaign; interrupted models restart from their first case.

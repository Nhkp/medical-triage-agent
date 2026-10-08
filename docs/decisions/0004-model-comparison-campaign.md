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
Kaggle artifacts are returned. Public API routes remain unchanged. The 2026-10-08 correction adds missing French red-flag
phrasing without changing fixture expectations.

## Consequences

- GPU execution needs a user-controlled Kaggle runtime and renewed HF credentials.
- Local tests verify the evaluator, not model performance or professional clinical safety.
- A failed or inconclusive campaign cannot be represented as a successful model validation.
- Changed fingerprints require a new campaign; interrupted models restart from their first case.

## Serving remediation — 2026-10-08

Preserve the initial `20261008T061839Z-8k` campaign. Keep vLLM 0.10.2, Transformers 4.56.2
and pinned model revisions; use first-call guided JSON and a shared 256-token ceiling.
Record finish metadata independently of preview storage truncation. Transform only local
adapter tokenizer configurations and require token/prompt equivalence before serving.
Technical preflight failures block measurements; medical inadequacy stays in the evidence.
Separate input `language` from review `language_quality` and repair historical reviews in a
new output directory. Corrected performance and conclusions depend on a returned Kaggle archive.

## Revised truncation policy — user instruction

Fix the comparison ceiling at 256 generated tokens for every model. Treat
`finish_reason=length` as an observed model failure rather than a preflight blocker.
Keep the incomplete output and generation metadata in evaluation denominators, use
the existing rule fallback, and do not fabricate a completed JSON answer. Technical
configuration and transport failures still block qualification. Preserve earlier
campaigns and record the changed policy in a new manifest.

# CHSA medical triage POC presentation plan

## 1. Medical Triage Agent

- Purpose: open with the project identity and maturity.
- Key message: bilingual, auditable, safety-oriented POC; not autonomous diagnosis.
- Evidence: `README.md`, `docs/medical-safety.md`.
- Visual: premium dark command-center hero with three proof points.
- Narrative: frame the work as clinical support infrastructure, not a medical device claim.

## 2. The CHSA Pressure Point

- Purpose: establish the emergency triage context from the mission brief.
- Key message: triage overload creates a need for faster symptom structuring and escalation cues.
- Evidence: `input/context.md`.
- Visual: patient/context/decision-pressure flow into clinician review.
- Narrative: the opportunity is synthesis and traceability under pressure.

## 3. Assist, Never Replace

- Purpose: make the safety boundary unmissable.
- Key message: the assistant supports clinicians; it must not diagnose, prescribe, or discharge.
- Evidence: `docs/medical-safety.md`, `AGENTS.md`, `README.md`.
- Visual: split contrast between supported behavior and forbidden behavior.
- Narrative: safety rules come before model optimization.

## 4. System in One Picture

- Purpose: show the full repository architecture.
- Key message: each stage is separated for auditability.
- Evidence: `docs/architecture.md`, `Makefile`, `docker-compose.yml`.
- Visual: professional multi-stage pipeline from sources to vLLM/FastAPI.
- Narrative: no magic box; data, training, evaluation, and serving are inspectable.

## 5. Step 0: Method Before Model

- Purpose: connect SFT/DPO concepts with the project safety design.
- Key message: the repo turns Step 0 into guardrails, contracts, and human-in-the-loop rules.
- Evidence: `input/step_0.md`, `docs/source-policy.md`, `docs/privacy-rgpd.md`.
- Visual: four foundations: scope, privacy, provenance, evaluation.
- Narrative: the model is only one component in a governed workflow.

## 6. Step 1: Dataset Built and Audited

- Purpose: present completed data-preparation evidence.
- Key message: the POC generated a bilingual SFT/DPO dataset with audit artifacts.
- Evidence: `docs/report.md`, `docs/data-sources.md`.
- Visual: metric wall with SFT, DPO, FR, EN counts.
- Narrative: Step 1 is technically complete, with generated data kept out of git.

## 7. Data Governance

- Purpose: explain source licensing and provenance.
- Key message: every source is tracked by ID, license, language, intended use, and status.
- Evidence: `docs/data-sources.md`, `docs/source-policy.md`.
- Visual: compact registry table.
- Narrative: verified public datasets enable a technical POC, not CHSA clinical labels.

## 8. Privacy and Audit Evidence

- Purpose: show RGPD-minded controls and automated findings.
- Key message: local audit passed with 0 PII findings, 0 duplicates, 0 missing provenance.
- Evidence: `docs/report.md`, `docs/privacy-rgpd.md`, `src/medical_triage_agent/privacy.py`.
- Visual: three green audit indicators plus rejection counts.
- Narrative: automated privacy checks are necessary evidence, not clinical approval.

## 9. Clinical Validation Debt

- Purpose: prevent overclaiming.
- Key message: 5,000 public QA records remain queued for clinician review.
- Evidence: `docs/report.md`, `docs/evaluation.md`.
- Visual: risk ledger with what is proven vs what is pending.
- Narrative: clinical sign-off is a go/no-go requirement.

## 10. Step 2: Low-VRAM Training Path

- Purpose: explain SFT/DPO execution strategy.
- Key message: Qwen3-1.7B-Base, 4-bit QLoRA SFT, then DPO, with conservative Colab/Kaggle defaults.
- Evidence: `configs/sft_kaggle.yaml`, `configs/dpo_kaggle.yaml`, `scripts/train_sft.py`, `scripts/train_dpo.py`.
- Visual: base model -> SFT adapter -> DPO adapter.
- Narrative: start small, smoke-test, push adapters to Hugging Face when stable.

## 11. Evaluation: Technical Indicators Only

- Purpose: show how progress is measured.
- Key message: model metrics, safety checks, latency, and traceability are tracked, but do not prove clinical safety.
- Evidence: `docs/evaluation.md`, `scripts/evaluate.py`, `scripts/evaluate_latency.py`, `scripts/evaluate_robustness.py`.
- Visual: evaluation matrix.
- Narrative: the project separates technical readiness from clinical validation.

## 12. Step 3: Serving and Demo API

- Purpose: show deployability.
- Key message: FastAPI exposes stable endpoints and vLLM can serve the selected HF model/adapter on GPU.
- Evidence: `src/medical_triage_agent/api.py`, `src/medical_triage_agent/vllm_client.py`, `Dockerfile`, `docker-compose.yml`.
- Visual: client -> FastAPI -> vLLM -> model with audit side channel.
- Narrative: fallback rules allow API validation without GPU; vLLM enables model-backed explanations.

## 13. Reproducibility and CI/CD

- Purpose: make the engineering quality visible.
- Key message: `make check`, data commands, serving commands, GitHub Actions, and notebooks make the work repeatable.
- Evidence: `Makefile`, `.github/workflows/check.yml`, `.github/workflows/deploy.yml`, `docs/colab-workflow.md`.
- Visual: command tiles and CI pipeline.
- Narrative: the project is designed to be rerun and inspected, not manually reconstructed.

## 14. Roadmap and Decision Gate

- Purpose: close with concrete next decisions.
- Key message: finish model-backed metrics, clinician review, and pilot hosting before any production claim.
- Evidence: `docs/report.md`, `input/step_3.md`, `docs/medical-safety.md`.
- Visual: go/no-go checklist.
- Narrative: the next milestone is not “more AI”; it is evidence, safety, and deployment discipline.

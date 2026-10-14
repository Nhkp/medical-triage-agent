# CHSA school defence: 14 slides / approximately 15 minutes

Canonical source: `presentations/chsa-current-state/index.html`. Both report and slides are in English. Technical campaign results are measured; AI-assisted pedagogical preannotations were reviewed and validated by the user on 14 October 2026. No unrestricted clinical demonstration is recommended.

## 1. Mission and POC Outcome

45 s. State the mission and the result early. This is a school POC for clinical staff, not autonomous diagnosis. Main data is 8,000 SFT and 2,000 DPO pairs; the final serving campaign has 144 measurements per model. User validation of all 144 AI-assisted preannotations is complete: 140 accepted and 4 corrected on 14 October 2026.

## 2. CHSA Need and Delivered Scope

45 s. Reference input/context.md. Do not describe a text-area API tester as an adaptive questionnaire. There are no private hospital data or deployed hospital integration. The clinically validated DPO-pair requirement remains an acknowledged gap.

## 3. Architecture and Backend Authority

60 s. Streamlit is a separate thin client. On Modal, FastAPI shares the container with vLLM. The backend keeps rule priority even when the model disagrees. It can still expose an unsafe accepted explanation when the priority matches. Audit state is in memory, not durable hospital infrastructure.

## 4. Main Bilingual Dataset

45 s. Main manifest timestamp 20 August 2026 10:48:52 UTC. Counting a DPO pair once gives 10,000 total records. The 8k SFT set extends the approximately 5k request. Do not merge historical 5k training metrics into 8k comparison results.

## 5. Provenance Privacy and Splits

60 s. Sources: MediQAl 3,951; FrenchMedMCQA 594; MedQuad 3,455; UltraMedical 2,000. Licence metadata is recorded; FrenchMedMCQA and MedQuad checks were owner-verifications. Zero PII findings only establishes what the automated scanner detected. Exact overlap checks are not semantic contamination guarantees.

## 6. SFT and DPO Training

60 s. Repository configuration: length 512, batch 1, accumulation 8, FP16, seed 42; SFT learning rate 2e-4 and DPO 5e-6, beta 0.1. These are configuration values, not a claim that every evaluated snapshot was reconstructed from local weights. Evaluated Hub snapshots are pinned separately. GRPO is not part of the comparison.

## 7. Reproducible Comparison Protocol

70 s. Code d9244c4, clean checkout. vLLM 0.10.2, Transformers 4.56.2, torch 2.8.0, T4 listed, FP16 eager. Two listed GPUs do not prove parallel use. Three repeated passes are not independent clinical samples. The case expectations follow POC rules. Measurements include explicit input language; protocol supplies rule priority.

## 8. Format and Repetition Results

70 s. Historical initial serving ignored JSON constraints; intermediate serving truncated all SFT/DPO outputs. Last prompt, penalty and language changes were simultaneous, so no isolated penalty effect is claimed. Stored previews are bounded, and exact repetition is not semantic redundancy. All generation metadata was present.

## 9. Latency and Model Differences

60 s. Startup separately: Base 76.10 s, SFT 64.06 s, DPO 60.06 s. Model output length contributes to latency. The earlier equal-length comparison suggested additional adapter-path overhead, but no profile proves a cause. Kaggle T4 timings cannot be substituted for Modal L4 timings.

## 10. Relevance and Pedagogical Review

70 s. User-validated nonconformity counts for invented/unsupported information: Base 21/48, SFT 21/46 with 2 unknown, DPO 23/48. User-validated dangerous advice: 3/48, 8/47 with 1 unknown, 8/48. These AI-assisted assessments were reviewed by the user: human_validated=true, 140 accepted and 4 corrected. DPO uncertainty nonconformity is now 42/48. A validated annotation can describe an unsafe response; clinical rates are not established.

## 11. Observed Failures and Protection Limits

90 s. Paraphrases only, not medical instructions. Also SFT/DPO French anaphylaxis contains prohibited treatment procedures. Final emergency rule priority is correct for 60/60 each, while accepted emergency suggestions are 35/60, 39/60 and 48/60. Raw priorities are 60/60, 55/60 and 60/60. Distinguish missing/rejected suggestions from final under-triage. User review confirms exclusion from unrestricted explanation use.

## 12. Modal Deployment and Demo Walkthrough

60 s. State the FastAPI parent-environment correction. User screenshot before the fix was rule-only; it must not be used as model success proof. Modal does not yet enforce the same static adapter SHA/compatibility copies as Kaggle. UI is separate, observed on Render and also used on Community Cloud. Use synthetic FR/EN input, emergency and dangerous request, show audit/status. Keep recorded backup, not fabricated output.

## 13. Evidence Limits and Demo Decision

55 s. Separate school completion from clinical readiness. No-go for clinical pilot. A deterministic explanation mode is recommended for future demo safety but has not been implemented here. Existing demonstration should be a supervised engineering case study. Completed human educational validation does not replace clinician sign-off.

## 14. Roadmap and Clinical Pilot Gates

50 s. Total planned narrative approximately 900 seconds, allowing transitions. Report stays under 20 pages and historical evidence is separate. Before pilot: qualified review, safe refusals, uncertainty, versioned deployment, monitoring, privacy and operational governance. End on the result: a working technical POC with documented safety limits.

## Export and rehearsal

Run `make presentation-pptx` for the existing rendered export. Speaker notes from the canonical HTML are automatically embedded in the generated PPTX. Export a slide PDF with LibreOffice. Compare slides with the report and campaign summary, verify 14 pages/slides and rehearse at 15 minutes. Keep a recorded backup and show explanation source/LLM status when demonstrating the API. Generated deliverables stay outside Git.

### Reproduce the deliverables

```bash
pandoc docs/report.md --standalone --columns=10 --reference-doc=dist/reports/chsa-report-reference.docx --output dist/reports/chsa-school-report.docx
libreoffice -env:UserInstallation=file:///tmp/chsa-lo-report-profile --headless --convert-to pdf --outdir dist/reports dist/reports/chsa-school-report.docx
make presentation-pptx
libreoffice -env:UserInstallation=file:///tmp/chsa-lo-slides-profile --headless --convert-to pdf --outdir dist/presentations dist/presentations/chsa-current-state.pptx
pdfinfo dist/reports/chsa-school-report.pdf
pdfinfo dist/presentations/chsa-current-state.pdf
```

Create the two `dist/` directories first and restore the supplied `chsa-report-reference.docx` from the delivery archive; its Compact style keeps table rows together. The `--columns=10` option prevents zero-width table grids with the installed Pandoc 2.9 DOCX writer; it does not set a ten-character paragraph width in Word. Isolated LibreOffice profiles avoid interfering with an existing desktop session. Report and deck exports incorporate the completed user review. The generated report must stay under 20 pages; the deck must stay at 14 slides/pages.

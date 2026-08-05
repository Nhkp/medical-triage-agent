# Data sources

This registry is the source of truth for dataset provenance. The code parser expects the
table below to keep these exact columns.

| id | name | url | license | languages | intended_use | status | notes |
| --- | --- | --- | --- | --- | --- | --- | --- |
| mediqa | MediQAl | https://huggingface.co/datasets/ANR-MALADES/MediQAl | cc-by-4.0 | fr | sft,evaluation | verified | Hugging Face API reports public, ungated, French medical QA with license cc-by-4.0; v1 ingests the oeq split only. |
| frenchmedmcqa | FrenchMedMCQA | https://huggingface.co/datasets/nthngdy/frenchmedmcqa | apache-2.0 | fr | sft,evaluation | verified | License manually verified by project owner; public French medical multiple-choice QA. |
| medquad | MedQuAD Medical QnA | https://huggingface.co/datasets/keivalya/MedQuad-MedicalQnADataset | apache-2.0 | en | sft | verified | License manually verified by project owner; public English medical QA. |
| ultramedical_preference | UltraMedical Preference | https://huggingface.co/datasets/TsinghuaC3I/UltraMedical-Preference | mit | en | dpo | verified | Hugging Face API reports public, ungated, English medical preference data with license mit. |

`status=blocked` means the source must not be ingested until license and suitability are
verified.

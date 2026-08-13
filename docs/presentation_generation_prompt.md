You are working inside the repository:

`medical-triage-agent`

Your task is to **research the project from the repository itself, design the presentation narrative, and then generate a complete professional PowerPoint presentation (`.pptx`)** about the project.

The final presentation must be exceptionally polished, modern, visually sophisticated, technically accurate, and suitable for presenting the project to a mixed audience of technical, AI/ML, healthcare, academic, and decision-making stakeholders.

Do **not** create a generic AI/healthcare presentation.

The content must come from the actual repository.

# 1. RESEARCH THE PROJECT BEFORE MAKING ANY SLIDES

Before writing slide content or creating PowerPoint code, inspect the repository thoroughly.

You MUST read the available project material, with particular attention to:

* `README.md`
* `AGENTS.md`
* **every file in `input/`**

  * `input/context.md`
  * `input/ressources.md`
  * `input/step_0.md`
  * `input/step_1.md`
  * `input/step_2.md`
  * `input/step_3.md`
* relevant files in `docs/`, especially:

  * `docs/architecture.md`
  * `docs/report.md`
  * `docs/data-sources.md`
  * `docs/source-policy.md`
  * `docs/privacy-rgpd.md`
  * `docs/medical-safety.md`
  * `docs/evaluation.md`
  * Kaggle / Colab workflow documentation
  * agent/workflow documentation
* `configs/`
* relevant training scripts
* relevant evaluation scripts
* relevant serving/API code under `src/medical_triage_agent`
* `tests/` where useful to understand guaranteed behavior
* `docker-compose.yml`
* `Dockerfile`
* `Makefile`
* `.github/workflows/`
* any existing material under `presentations/`

The `input/` directory is especially important.

Treat those files as **project requirements and narrative context**, not as optional documentation.

Do not assume what Step 0, Step 1, Step 2, or Step 3 mean from their names. READ THEM.

Before generating slides, build your own internal factual project summary covering:

1. project motivation;
2. user/clinical context;
3. functional objectives;
4. explicit safety boundaries;
5. architecture;
6. dataset sources;
7. data engineering;
8. privacy/RGPD controls;
9. provenance and licensing;
10. model selection;
11. SFT strategy;
12. DPO strategy;
13. optional GRPO work;
14. infrastructure constraints;
15. Kaggle / GPU strategy;
16. evaluation methodology;
17. serving architecture;
18. API;
19. auditability;
20. reproducibility;
21. CI/testing;
22. achievements already demonstrated;
23. claims that are NOT yet demonstrated;
24. limitations;
25. clinical validation debt;
26. roadmap.

Cross-check important factual claims against multiple repository files when possible.

If documentation and implementation disagree, prefer what is demonstrably implemented and explicitly mention the difference if relevant.

NEVER invent:

* model performance;
* accuracy;
* clinical validation;
* production deployment;
* trained checkpoints;
* evaluation scores;
* hospital adoption;
* regulatory approval;
* patient outcomes.

Clearly distinguish:

**implemented / validated locally**

from

**planned / scaffolded / awaiting final training or clinical validation**.

# 2. FIRST PRODUCE A PRESENTATION PLAN

Before generating the `.pptx`, create a concise presentation-plan document or console output.

For every proposed slide specify:

* slide number;
* slide title;
* purpose;
* key message;
* repository evidence/source files supporting it;
* proposed visual concept;
* data/chart/diagram needed;
* speaker narrative.

Use this planning phase to make sure the presentation tells one coherent story.

Then generate the PowerPoint.

Do not stop after the plan.

# 3. TARGET PRESENTATION STRUCTURE

Aim for approximately **13–15 main slides**.

Adjust the exact number if the repository evidence suggests a better storytelling structure.

The preferred narrative is:

## Slide 1 — Hero / Opening

Introduce:

**Medical Triage Agent**

A bilingual, auditable and safety-oriented medical triage AI proof of concept.

Do NOT make it sound like an autonomous diagnostic system.

Create a striking opening visual.

Minimal text.

Think premium AI product keynote rather than university template.

---

## Slide 2 — The Problem

Explain the challenge of emergency/initial medical triage:

* rapid information gathering;
* symptoms + patient context;
* identifying escalation/red-flag signals;
* uncertainty;
* limited clinician time;
* need for traceability.

Then establish the opportunity:

AI as a **clinical support layer**, not a replacement for medical judgment.

Visual idea:
patient → symptoms/context → decision pressure → clinician, with the AI assistant augmenting information synthesis.

---

## Slide 3 — Objective, Scope & Safety Boundary

Clearly show:

### What the assistant SHOULD do

* collect patient-declared symptoms;
* structure relevant context;
* detect/red-flag escalation signals;
* provide triage-oriented explanations;
* retain provenance/audit metadata;
* support clinician review.

### What it MUST NOT do

* autonomous diagnosis;
* autonomous treatment decisions;
* unsupervised patient-facing final recommendations;
* replace clinician judgment.

Make this one of the clearest slides in the deck.

Use a strong “assist / never replace” visual contrast.

---

## Slide 4 — The System in One Picture

Use the actual repository architecture as the basis.

Show the complete flow:

Verified public medical datasets

→ source registry / licensing / provenance

→ ingestion

→ normalization into SFT/DPO schemas

→ PII / RGPD checks

→ deterministic train / validation / test / clinical-eval splits

→ audit manifest + dataset card + clinician-review queue

→ Qwen base model

→ QLoRA SFT

→ instruction-tuned adapter

→ DPO alignment

→ aligned medical-triage model

→ vLLM

→ FastAPI wrapper

→ safety evaluation

→ audit metadata

→ demo/API consumer.

This should be one of the visual centerpiece slides.

Redraw the architecture professionally rather than screenshotting Mermaid.

---

## Slide 5 — Step 0: Designing for Safety Before Training

Use `input/step_0.md` and associated documentation.

Explain the design philosophy:

* healthcare context first;
* safety boundaries before optimization;
* human-in-the-loop;
* bilingual EN/FR;
* provenance;
* privacy/RGPD;
* auditability;
* reproducibility;
* technical vs. clinical validation.

Show that safety is part of the architecture, not a final checkbox.

---

## Slide 6 — Step 1: Data Strategy

Show the actual source datasets identified in the repository, including:

* MediQAl;
* FrenchMedMCQA;
* MedQuAD;
* UltraMedical-Preference.

Explain the role of each dataset:

* French/English coverage;
* QA vs preference data;
* SFT vs DPO use.

Include license/provenance information where appropriate.

Design this like a modern data-source constellation or four source cards feeding the pipeline.

---

## Slide 7 — From Raw Public Data to Auditable Training Data

Explain:

* source registration;
* schema normalization;
* provenance tracking;
* license checks;
* PII screening;
* RGPD considerations;
* duplicate checks;
* deterministic splits;
* manifest generation;
* dataset card;
* clinical review queue.

Use actual repository-generated dataset numbers when verified.

Important numbers currently documented include approximately:

* 5,000 SFT records;
* 1,000 DPO records;
* bilingual English/French composition;
* train / validation / test / clinical-evaluation splits.

Verify the exact values from the current repository before using them.

Where the repo includes audit numbers, visualize them.

For example, if current repository evidence still confirms it, surface:

* zero obvious PII findings;
* zero duplicate findings;
* zero missing provenance findings.

Do not turn those technical checks into a claim of clinical validation.

Explicitly show that the clinician review queue represents **validation debt**, not completed clinician sign-off.

This distinction matters.

Use charts rather than dense tables wherever possible.

---

## Slide 8 — Step 2: Model Adaptation Strategy

Explain why the project uses a staged alignment pipeline.

Visually show:

**Qwen3-1.7B-Base**

↓

**4-bit QLoRA SFT**

teach task format / instruction behavior

↓

**Instruction-tuned adapter**

↓

**DPO**

align preferred medical-triage behavior / safety preferences

↓

**Aligned triage model**

Optionally show GRPO separately as an experimental extension if supported by the repository.

Explain LoRA/QLoRA intuitively rather than dumping equations.

Make clear which stages are implemented as scripts/configuration versus which have actually completed full training.

---

## Slide 9 — Training Under Real-World Compute Constraints

Explain the engineering decision to make training reproducible on accessible hardware.

Cover as supported by the repository:

* Kaggle free GPU path;
* low-VRAM QLoRA;
* configuration-driven training;
* YAML configs;
* CLI overrides;
* checkpoint/resume capabilities;
* CPU-safe smoke tests;
* optional Colab workflow;
* Hugging Face publication strategy.

This slide should communicate thoughtful engineering constraints rather than “we used Kaggle because it was free.”

Use an elegant compute/experiment workflow diagram.

---

## Slide 10 — Safety & Evaluation Framework

Read `docs/evaluation.md`, `docs/medical-safety.md`, tests and relevant scripts carefully.

Explain evaluation categories such as, where actually supported:

* red-flag escalation;
* unsafe-response prevention;
* hallucination behavior;
* bilingual robustness;
* empty/invalid input handling;
* privacy behavior;
* audit trace behavior;
* latency;
* response size;
* model-backed quality metrics;
* clinical evaluation set.

Clearly distinguish:

existing deterministic/local technical checks

from

future/model-backed/clinician-backed evaluation.

If no final metric exists, show the **evaluation framework**, not a fabricated score.

A radar chart with invented values is forbidden.

---

## Slide 11 — Step 3: Serving & Auditability

Explain the deployed/scaffolded serving architecture.

Use repository code and docs to show:

Model / rule fallback

→ vLLM OpenAI-compatible backend

→ thin FastAPI CHSA wrapper

→ domain validation

→ safety disclaimers

→ metadata-only audit trace

→ stable API endpoints.

Include actual endpoints where useful:

* `/health`
* `/triage`
* `/audit/{id}`

Explain why audit metadata belongs in the wrapper layer.

A clean request/response sequence diagram would work well.

---

## Slide 12 — Engineering for Reproducibility

Show that the project is more than a notebook.

Highlight as verified in the repo:

* project package structure;
* Makefile commands;
* unit/integration tests;
* linting;
* formatting;
* typing;
* CI;
* source-registry validation;
* deterministic data generation;
* audit artifacts;
* Docker;
* configuration-driven experiments;
* smoke tests.

Potential visual:

a central “make check” / reproducibility gate with branches for:

DATA
TRAINING
EVALUATION
API
CI.

Avoid displaying huge blocks of terminal text.

Use a few carefully selected commands as visual proof points.

---

## Slide 13 — Where the POC Stands Today

Create a highly legible maturity/status slide.

Use three categories:

### DONE / DEMONSTRATED

Only include things actually supported by repository evidence.

Potential examples after verification:

* data source registry;
* normalized dataset pipeline;
* local SFT/DPO dataset generation;
* privacy/provenance audits;
* private Hugging Face dataset publication;
* SFT/DPO/GRPO training scaffolds;
* smoke tests;
* FastAPI wrapper;
* vLLM integration path;
* robustness/latency evaluation tooling;
* CI / quality gate.

### READY FOR NEXT EXECUTION

Examples:

* full GPU SFT;
* DPO training;
* adapter publication;
* model-backed evaluation.

### REQUIRES CLINICAL VALIDATION

* clinician review;
* CHSA-specific triage labels;
* real-world safety validation;
* clinical acceptance;
* pilot governance.

The audience should understand project maturity within five seconds.

---

## Slide 14 — Limitations & Go/No-Go Gates

Healthcare AI requires intellectual honesty.

Explicitly discuss:

* public medical datasets ≠ CHSA clinician-validated triage labels;
* technical evaluation ≠ clinical validation;
* no autonomous diagnosis;
* no production medical-device claim;
* no invented performance claims;
* final model-backed metrics depend on trained adapters;
* human review is mandatory.

Then show the gates required before pilot exposure.

Make this visually confident rather than apologetic.

This slide demonstrates responsible engineering.

---

## Slide 15 — Roadmap / Closing

End with the path forward:

DATA FOUNDATION
→ TRAINED ADAPTERS
→ MODEL-BACKED EVALUATION
→ CLINICIAN REVIEW
→ SAFETY GATES
→ CONTROLLED PILOT

Close with one memorable statement such as:

**AI should accelerate triage reasoning without removing human accountability.**

or another sentence better supported by the repository.

Finish with a powerful but restrained visual.

# 4. VISUAL DIRECTION

The visual quality is a critical requirement.

The presentation should feel like a fusion of:

* a premium AI research keynote;
* a modern health-tech product presentation;
* an architecture/design presentation;
* a top-tier technology conference deck.

Avoid the appearance of:

* default PowerPoint;
* consulting-template overload;
* university lecture slides;
* generic corporate healthcare templates;
* Canva-like decorative layouts;
* bullet-heavy AI-generated slides.

## Overall aesthetic

Use a sophisticated healthcare/AI visual system.

Suggested direction:

* dark navy / near-black foundation OR sophisticated off-white backgrounds;
* luminous cyan / turquoise / electric blue accents;
* occasional subtle medical green;
* high contrast typography;
* large negative space;
* subtle gradients;
* translucent/glass-like panels used sparingly;
* fine grid / data / neural motifs;
* clean geometric connectors;
* elegant technical diagrams;
* premium iconography;
* restrained medical symbolism.

Avoid excessive red.

Red may be used deliberately for:

* escalation;
* unsafe behavior;
* red flags;
* boundaries.

Do NOT fill the presentation with generic stock photos of doctors.

Prefer:

* diagrams;
* abstract medical/AI imagery;
* data visualization;
* UI-inspired compositions;
* process graphics;
* architecture graphics;
* meaningful generated/illustrative elements if tooling allows.

# 5. SLIDE DESIGN PRINCIPLES

Each slide should have:

* ONE dominant idea;
* a clear visual hierarchy;
* a strong title;
* minimal but meaningful text;
* intentional whitespace;
* no accidental empty areas;
* balanced composition.

Prefer:

headline

* visual
  + 2–4 concise supporting points

over long bullet lists.

Do not copy paragraphs from documentation.

Translate technical documentation into presentation language.

Use detailed explanations in **speaker notes**, not on-slide body text.

# 6. SPEAKER NOTES

Add meaningful speaker notes to each slide.

Speaker notes should explain:

* what the presenter should say;
* what repository evidence supports the slide;
* important nuance;
* caveats;
* transitions into the next slide.

The deck should therefore function both as:

1. a visually impressive presentation; and
2. a complete presentation aid for the speaker.

# 7. DIAGRAMS

Create native editable PowerPoint diagrams whenever reasonable.

Important diagrams include:

* end-to-end architecture;
* data pipeline;
* SFT → DPO alignment;
* serving architecture;
* safety/evaluation loop;
* roadmap.

Do not simply paste screenshots of Mermaid diagrams when they can be professionally recreated.

Use alignment, consistent spacing, connectors and grouping carefully.

All diagrams should remain readable when projected.

# 8. DATA VISUALIZATION

Where real data exists in the repository, visualize it.

Possible charts include:

* SFT vs DPO sample volume;
* English vs French composition;
* records by source;
* train/validation/test/clinical-evaluation distribution;
* project maturity / milestone view;
* evaluation framework.

Do not create misleading charts.

Do not fabricate values.

For qualitative frameworks, use diagrams instead of arbitrary numeric charts.

# 9. TECHNICAL ACCURACY

Before finalizing, perform a factual audit of every slide.

For each claim ask:

“Which repository file proves this?”

If there is no evidence, remove or qualify the statement.

Specific healthcare safety rule:

Never present the Medical Triage Agent as:

* a physician;
* diagnostic authority;
* treatment recommender;
* autonomous triage decision maker;
* clinically validated medical device,

unless repository evidence explicitly supports such a claim.

Use wording such as:

* proof of concept;
* clinical decision-support exploration;
* triage-oriented assistant;
* clinician-facing support;
* human-reviewed workflow;
* safety-oriented architecture.

# 10. IMPLEMENTATION

Generate an actual `.pptx` file programmatically.

Use the best PowerPoint generation tooling available in the environment.

If repository-specific presentation-generation instructions or skills are available, follow them.

Create reusable helpers/components for:

* typography;
* titles;
* subtitles;
* cards;
* badges;
* architecture nodes;
* connectors;
* page numbers;
* footers;
* citations/source labels;
* diagrams;
* charts.

Keep design tokens centralized.

Use a widescreen **16:9** layout.

Ensure every element remains editable whenever practical.

# 11. SOURCES / FOOTNOTES

For important factual slides, include unobtrusive source references in the footer, such as:

`Source: docs/report.md · docs/architecture.md`

Do not clutter slides with raw URLs.

The final slide may include a concise repository reference.

# 12. QUALITY ASSURANCE

After generating the presentation:

1. render every slide to images;
2. visually inspect every slide;
3. detect:

   * text overflow;
   * clipped objects;
   * overlaps;
   * tiny text;
   * alignment issues;
   * unreadable diagrams;
   * inconsistent margins;
   * inconsistent typography;
   * poor contrast;
   * awkward whitespace;
4. fix all issues;
5. render again;
6. repeat until clean.

Also inspect the PPTX programmatically where possible for objects outside slide boundaries or overlapping unintentionally.

# 13. DELIVERABLES

Produce:

1. the final `.pptx`;
2. the source code used to generate it;
3. any generated visual assets in an organized folder;
4. a slide-plan / content-outline file;
5. optionally a PDF preview if the environment supports it.

Suggested output structure:

`presentations/medical-triage-agent-final/`

containing for example:

* `medical-triage-agent.pptx`
* `generate_presentation.py` or equivalent
* `presentation_plan.md`
* `assets/`
* `renders/`
* optional `medical-triage-agent.pdf`

# 14. FINAL CHECK

Before considering the task finished, verify that someone seeing only the presentation can clearly answer:

1. What clinical problem are we solving?
2. Who is the assistant for?
3. What can it do?
4. What is it explicitly forbidden from doing?
5. Why is the project bilingual?
6. Where does the data come from?
7. How is provenance handled?
8. How is privacy/RGPD handled?
9. What data has actually been produced?
10. How is that data audited?
11. Why SFT?
12. Why DPO?
13. Why QLoRA?
14. Why Qwen?
15. What is actually trained today?
16. What is still pending?
17. How is safety evaluated?
18. How does the API work?
19. How is auditability implemented?
20. What is reproducible today?
21. What remains before a clinical pilot?
22. Why does human clinical oversight remain mandatory?

If any of these answers is unclear, revise the deck.

The desired end result is not merely “a presentation about the repository.”

It should feel like a **carefully art-directed visual narrative of how a safety-first medical AI system is being engineered from data provenance all the way to controlled clinical deployment**.

Research first.

Plan second.

Design third.

Generate the `.pptx`.

Render and inspect it.

Revise until presentation-grade.

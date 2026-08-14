from __future__ import annotations

from typing import Any

from medical_triage_agent.triage import TriageResponse, assess_triage, audit_metadata
from medical_triage_agent.vllm_client import configured_model, generate_explanation, is_configured

_AUDIT_STORE: dict[str, dict[str, Any]] = {}


def triage(payload: dict[str, Any]) -> dict[str, str]:
    response = assess_triage(payload)
    generated_explanation = generate_explanation(payload, response)
    if generated_explanation is not None:
        response = TriageResponse(
            priority=response.priority,
            explanation=generated_explanation,
            disclaimer=response.disclaimer,
            audit_id=response.audit_id,
        )
    _AUDIT_STORE[response.audit_id] = audit_metadata(payload, response, model=configured_model())
    return response.to_dict()


def audit(audit_id: str) -> dict[str, Any] | None:
    return _AUDIT_STORE.get(audit_id)


def health() -> dict[str, str]:
    return {
        "status": "ok",
        "model": configured_model(),
        "vllm": "configured" if is_configured() else "disabled",
    }


def demo_html() -> str:
    return """<!doctype html>
<html lang="en">
<head>
  <meta charset="utf-8">
  <meta name="viewport" content="width=device-width, initial-scale=1">
  <title>CHSA Medical Triage Demo</title>
  <style>
    :root {
      --bg: #f6f8fa;
      --surface: #ffffff;
      --ink: #102033;
      --muted: #5b6b7a;
      --accent: #0ea5b7;
      --critical: #c24135;
      --critical-soft: #fdecec;
      --border: #dce5ea;
    }
    body {
      margin: 0;
      background: var(--bg);
      color: var(--ink);
      font: 15px/1.5 Inter, ui-sans-serif, system-ui, -apple-system, BlinkMacSystemFont, "Segoe UI", sans-serif;
    }
    main {
      max-width: 1040px;
      margin: 0 auto;
      padding: 36px 20px 48px;
    }
    header, section {
      background: var(--surface);
      border: 1px solid var(--border);
      border-radius: 12px;
      padding: 22px;
      box-shadow: 0 12px 40px rgba(16, 32, 51, 0.05);
      margin-bottom: 18px;
    }
    h1, h2 { margin: 0 0 8px; }
    p { color: var(--muted); }
    label { display: block; font-weight: 650; margin: 16px 0 6px; }
    textarea, input {
      box-sizing: border-box;
      width: 100%;
      border: 1px solid var(--border);
      border-radius: 8px;
      padding: 12px;
      font: inherit;
      background: #fbfdfe;
    }
    button {
      border: 0;
      border-radius: 8px;
      padding: 11px 14px;
      margin-top: 14px;
      background: var(--accent);
      color: white;
      font-weight: 700;
      cursor: pointer;
    }
    code {
      background: #eef5f6;
      border-radius: 6px;
      padding: 3px 6px;
    }
    .grid { display: grid; grid-template-columns: 1.3fr 0.9fr; gap: 18px; }
    .status { display: flex; gap: 8px; flex-wrap: wrap; }
    .chip {
      display: inline-flex;
      border: 1px solid var(--border);
      border-radius: 999px;
      padding: 4px 10px;
      color: var(--muted);
      background: #fbfdfe;
    }
    .result {
      border-left: 6px solid var(--accent);
      background: #eef8f5;
    }
    .urgent {
      border-left-color: var(--critical);
      background: var(--critical-soft);
    }
    .hidden { display: none; }
    .metadata { color: var(--muted); font-size: 13px; }
    @media (max-width: 760px) { .grid { grid-template-columns: 1fr; } }
  </style>
</head>
<body>
<main>
  <header>
    <h1>CHSA Medical Triage Demo</h1>
    <p>Proof-of-concept support for clinical staff. Human clinical review remains authoritative.</p>
    <div class="status" id="health"><span class="chip">Checking backend...</span></div>
  </header>
  <div class="grid">
    <section>
      <h2>New triage</h2>
      <p>Enter one symptom per line. Do not enter identifiable patient data in this demo.</p>
      <label for="symptoms">Symptoms</label>
      <textarea id="symptoms" rows="8" placeholder="douleur thoracique&#10;difficulte respiratoire"></textarea>
      <button id="analyze">Analyze triage</button>
      <p class="metadata">This UI calls <code>POST /triage</code>.</p>
    </section>
    <section>
      <h2>Audit lookup</h2>
      <p>Audit output is metadata-only and does not expose raw patient text.</p>
      <label for="audit-id">Audit ID</label>
      <input id="audit-id" placeholder="audit_...">
      <button id="lookup">View audit trace</button>
      <pre id="audit-output" class="metadata"></pre>
    </section>
  </div>
  <section id="result" class="result hidden">
    <h2 id="priority"></h2>
    <p id="explanation"></p>
    <p id="disclaimer"></p>
    <p>Audit ID: <code id="audit"></code></p>
  </section>
</main>
<script>
const health = document.getElementById("health");
const result = document.getElementById("result");
const auditInput = document.getElementById("audit-id");

async function refreshHealth() {
  try {
    const response = await fetch("/health");
    const data = await response.json();
    health.innerHTML = `<span class="chip">API: ${data.status}</span><span class="chip">Model: ${data.model}</span><span class="chip">vLLM: ${data.vllm}</span>`;
  } catch (_error) {
    health.innerHTML = '<span class="chip">API offline</span>';
  }
}

document.getElementById("analyze").addEventListener("click", async () => {
  const symptoms = document.getElementById("symptoms").value.split("\\n").map(s => s.trim()).filter(Boolean);
  if (!symptoms.length) {
    alert("Add at least one symptom.");
    return;
  }
  const response = await fetch("/triage", {
    method: "POST",
    headers: {"Content-Type": "application/json"},
    body: JSON.stringify({symptoms})
  });
  const data = await response.json();
  if (!response.ok) {
    alert(data.detail || "Triage request failed.");
    return;
  }
  result.classList.toggle("urgent", data.priority === "urgence_maximale");
  result.classList.remove("hidden");
  document.getElementById("priority").textContent = data.priority === "urgence_maximale" ? `Urgent escalation: ${data.priority}` : `Priority: ${data.priority}`;
  document.getElementById("explanation").textContent = data.explanation;
  document.getElementById("disclaimer").textContent = data.disclaimer;
  document.getElementById("audit").textContent = data.audit_id;
  auditInput.value = data.audit_id;
});

document.getElementById("lookup").addEventListener("click", async () => {
  const auditId = auditInput.value.trim();
  if (!auditId) return;
  const response = await fetch(`/audit/${encodeURIComponent(auditId)}`);
  const data = await response.json();
  document.getElementById("audit-output").textContent = JSON.stringify(data, null, 2);
});

refreshHealth();
</script>
</body>
</html>"""


def create_app() -> Any:
    from fastapi import FastAPI, HTTPException
    from fastapi.responses import HTMLResponse

    app = FastAPI(title="CHSA medical triage POC")

    @app.get("/", response_class=HTMLResponse)
    def root_endpoint() -> str:
        return demo_html()

    @app.get("/demo", response_class=HTMLResponse)
    def demo_endpoint() -> str:
        return demo_html()

    @app.get("/health")
    def health_endpoint() -> dict[str, str]:
        return health()

    @app.post("/triage")
    def triage_endpoint(payload: dict[str, Any]) -> dict[str, str]:
        try:
            return triage(payload)
        except ValueError as exc:
            raise HTTPException(status_code=400, detail=str(exc)) from exc

    @app.get("/audit/{audit_id}")
    def audit_endpoint(audit_id: str) -> dict[str, Any]:
        record = audit(audit_id)
        if record is None:
            raise HTTPException(status_code=404, detail="audit record not found")
        return record

    return app

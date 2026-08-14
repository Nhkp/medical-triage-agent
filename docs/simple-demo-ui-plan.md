# Simple FastAPI demo UI

The demo interface is intentionally small and served by the existing FastAPI process.

## Purpose

- Provide a browser-accessible console for `/health`, `/triage`, and `/audit/{id}`.
- Work locally at `http://127.0.0.1:8080/demo`.
- Work through Kaggle/ngrok at `https://<ngrok-host>/demo`.
- Avoid a separate frontend server, CORS configuration, Streamlit dependency, or duplicated
  clinical logic.

## Boundaries

- The page is a proof-of-concept endpoint console, not a clinical product UI.
- It does not make diagnoses or treatment recommendations.
- It sends only a `symptoms` list to the existing `/triage` endpoint.
- Audit lookup displays only metadata returned by the backend.

## Safety

Urgent priorities are rendered with a visible escalation state. The backend disclaimer is shown
with each result, and users are warned not to enter identifiable patient data in the demo.

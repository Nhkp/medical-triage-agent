---
title: CHSA Medical Triage Streamlit
colorFrom: blue
colorTo: green
sdk: streamlit
sdk_version: 1.50.0
app_file: app.py
pinned: false
---

# CHSA Medical Triage Streamlit

This Space hosts only the Streamlit UI. It calls the Modal-hosted FastAPI triage API and
does not load vLLM or model weights locally.

Configure these Space settings before launch:

- `MODAL_API_URL`: variable pointing to the Modal app URL.
- `TRIAGE_API_TOKEN`: secret used as the bearer token for protected API calls.

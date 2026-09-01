# InsightPilot — AI Analytics Copilot

A portfolio-ready AI-first product prototype that converts natural-language business questions into safe SQL and evidence-backed insights.

**Workflow:** User question → OpenAI LLM → structured SQL plan → SQL guardrail → DuckDB → result → chart.

## Features
- Live OpenAI LLM mode
- Structured JSON output
- Read-only SQL guardrails
- CSV upload
- DuckDB execution
- Tables and charts
- Deterministic demo fallback
- Evaluation harness

## Local setup
```bash
pip install -r requirements.txt
streamlit run app.py
```

Create `.streamlit/secrets.toml` locally:
```toml
OPENAI_API_KEY = "sk-..."
OPENAI_MODEL = "gpt-5.6-luna"
```

Do not commit secrets.

## Deployment
See `LIVE_DEPLOYMENT.md`.

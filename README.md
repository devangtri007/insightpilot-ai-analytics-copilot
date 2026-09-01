
# InsightPilot — AI-First Analytics Copilot

A portfolio-ready Associate Product Manager prototype that turns natural-language business questions into **read-only SQL**, executes the query against a dataset, and presents an evidence-backed result.

## Why this project

The product demonstrates the workflow an AI-first PM should be able to own:

**User problem → AI prototype → structured output → evaluation → safe execution → measurable result**

It intentionally uses **synthetic pharma-commercial data**, so no real customer or PHI data is involved.

## Product capabilities

- Natural-language analytics interface
- LLM-generated structured plan: SQL + answer + assumptions + chart specification
- Read-only SQL guardrails
- CSV upload
- Automatic query execution with DuckDB
- Result table + chart
- Deterministic demo mode without an API key
- Evaluation harness for analytical intent and SQL safety

## Run locally

```bash
python -m venv .venv
# Windows:
.venv\Scripts\activate
# macOS/Linux:
source .venv/bin/activate

pip install -r requirements.txt
streamlit run app.py
```

The app works immediately in **Demo mode**.

To enable live AI:

```bash
# Windows PowerShell
$env:OPENAI_API_KEY="your_key_here"

# macOS/Linux
export OPENAI_API_KEY="your_key_here"
```

Then restart Streamlit.

## Evaluation

```bash
python evals/run_evals.py
```

The evaluation suite checks whether generated SQL:
- is read-only
- reflects the requested analytical dimension
- contains the intended aggregation
- returns a non-empty result

## Suggested demo questions

1. Which products generated the most revenue?
2. Compare revenue by region.
3. Which product has the highest weighted gross margin?

## Product thinking

### Target user
A business analyst / PM / sales leader who needs quick answers from structured data but does not want to write SQL.

### Core success metrics
- Task success rate
- SQL execution success rate
- Answer correctness
- Time-to-insight
- User correction rate
- % of questions requiring manual analyst intervention

### Next experiments
1. **Baseline:** compare manual SQL workflow vs InsightPilot.
2. **Accuracy:** test 50–100 benchmark questions with expected outputs.
3. **Trust:** expose SQL + assumptions and measure user acceptance.
4. **Safety:** add a semantic SQL validator and query-cost limits.
5. **Product:** add saved questions and feedback buttons.
6. **AI:** add a RAG layer for metric definitions / business glossary.
7. **Agent:** allow the system to decompose multi-step questions into several read-only queries.

## Portfolio positioning

Use this as a project demonstrating:
- AI-assisted product prototyping
- LLM structured outputs
- SQL + analytics
- AI evaluation
- safety/guardrails
- user workflow design
- experimentation mindset

Do not claim production scale or real customer usage. It is a working portfolio prototype built on synthetic data.


## Live LLM deployment

InsightPilot supports live LLM mode through the OpenAI Responses API. Set `OPENAI_API_KEY` and optionally `OPENAI_MODEL=gpt-5.6-luna`. Without a key, the app falls back to deterministic demo mode.

See `LIVE_DEPLOYMENT.md` for local setup and Streamlit deployment instructions.

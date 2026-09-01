# InsightPilot — AI Analytics Copilot

A Prototype that turns natural-language business questions into **read-only SQL**, executes the query against a dataset, and presents an evidence-backed result.

## Product Architecture

```text
                   USER
                    │
                    ▼
        ┌─────────────────────┐
        │  Streamlit Frontend │
        │                     │
        │ "Which product had  │
        │  highest revenue?"  │
        └──────────┬──────────┘
                   │
                   ▼
        ┌─────────────────────┐
        │    OpenAI LLM       │
        │                     │
        │ Natural language    │
        │       ↓             │
        │ Structured plan     │
        │       ↓             │
        │ SQL + assumptions   │
        └──────────┬──────────┘
                   │
                   ▼
        ┌─────────────────────┐
        │   SQL Guardrails    │
        │                     │
        │ SELECT/WITH only    │
        │ No mutations        │
        └──────────┬──────────┘
                   │
                   ▼
        ┌─────────────────────┐
        │       DuckDB        │
        │                     │
        │ Synthetic dataset   │
        └──────────┬──────────┘
                   │
                   ▼
        ┌─────────────────────┐
        │ Result + Chart +    │
        │ Explanation         │
        └─────────────────────┘
```

## Architecture Roadmap

### Phase 1 — MVP / Current
**Natural language → SQL → evidence**
- Streamlit product UI
- OpenAI Responses API
- Structured JSON planning
- Read-only SQL validation
- DuckDB execution
- Results + charts
- Visible assumptions and SQL
- Synthetic dataset
- Demo fallback

### Phase 2 — Trust & Evaluation
```text
User question → AI plan → Evaluation
                         ├─ Intent match
                         ├─ SQL safety
                         ├─ Execution success
                         └─ Answer correctness
                                      ↓
                              Feedback + logs
```

### Phase 3 — Business Context / RAG
```text
User → Intent → Business Glossary / RAG → SQL Planner → Guardrails → Data
```

### Phase 4 — Agentic Analytics
```text
User → Planner Agent
        ├─ Query data
        ├─ Validate result
        ├─ Run follow-up analysis
        ├─ Compare segments
        └─ Generate recommendation
```

### Phase 5 — Production
```text
USER → Web UI → API/Auth → AI Orchestrator
                         ↙      ↓       ↘
                       RAG   Evaluator  Memory
                         ↘      ↓       ↙
                       SQL Validator
                            ↓
                     Read-only DB Role
                            ↓
                      Data Warehouse
                            ↓
                 Answer + Evidence + Chart
                            ↓
                       Feedback/Logs
```

Production controls: authentication, RBAC, rate limits, query timeouts, stronger AST validation, observability and audit logs.

## Product Thinking

**Target user:** business analysts, PMs, sales leaders and operations users.

**Problem:** routine analytical questions often require analyst/SQL support.

**MVP hypothesis:** natural-language analytics plus visible SQL/evidence can reduce time-to-insight while preserving trust.

**Primary metric:** task success rate.

**Secondary metrics:** answer correctness, SQL execution success, median time-to-insight, correction rate and analyst intervention rate.

## Experiments

1. Compare time-to-answer with manual SQL.
2. Build a 100+ question benchmark with human-verified answers.
3. Test answer-only vs answer + assumptions + SQL for trust.
4. Add RAG metric definitions and measure ambiguity-related errors.

## Safety

The prototype rejects mutating/admin SQL and permits only `SELECT`/`WITH`. This is a portfolio-level guardrail, not a production security boundary. Production should add restricted database roles, stronger AST validation, query limits, authentication and logging.

## Data

Synthetic commercial data only. No real customer data, PHI or confidential client data.

## Local setup

```bash
pip install -r requirements.txt
streamlit run app.py
```

Create `.streamlit/secrets.toml`:

```toml
OPENAI_API_KEY = "sk-..."
OPENAI_MODEL = "gpt-5.6-luna"
```

Never commit this file.

## Deployment

See `LIVE_DEPLOYMENT.md` for GitHub + Streamlit Community Cloud deployment.


# InsightPilot

AI analytics copilot for turning business questions into
safe, executable data analysis.

[Live Demo] [GitHub]

## Why I built it

Most analytics workflows still require users to know:
- where the data lives
- how it is structured
- how to write SQL
- which metrics to calculate

InsightPilot explores a different interaction:
ask a business question → get an analysis.

## Product

User Question
      ↓
LLM reasoning
      ↓
Structured analytical plan
      ↓
SQL guardrails
      ↓
DuckDB
      ↓
Result + visualization + explanation

## What it can do

- Natural-language analytical queries
- Aggregation and metric calculation
- Grouping and comparisons
- Ranking and filtering
- Automatic chart selection
- Schema-aware reasoning
- Read-only SQL execution

## Safety by design

InsightPilot does not blindly execute generated SQL.

Generated queries pass through guardrails that:
- allow SELECT / WITH queries
- reject destructive operations
- prevent multi-statement execution
- validate identifiers against the dataset schema

## Evaluation

40 test cases
30 analytical
10 adversarial

100% overall task success
100% analytical cases
100% adversarial safety
0 failures

The evaluation suite covers:
- aggregation
- grouping
- ranking
- filtering
- comparison
- schema grounding
- unsupported requests
- ambiguous requests
- destructive SQL attempts
- prompt injection

## Architecture

...

## Product decisions

### 1. Why SQL instead of dataframe code?

...

### 2. Why DuckDB?

...

### 3. Why read-only?

...

## What I learned

...

## Next iteration

...

## Tech Stack

Python · Streamlit · OpenAI · DuckDB · Pandas
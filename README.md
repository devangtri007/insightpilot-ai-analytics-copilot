# InsightPilot

### AI analytics copilot for turning business questions into safe, executable data analysis.

InsightPilot lets users explore structured datasets using natural language instead of writing SQL or manually building analysis.

**Ask a question → InsightPilot plans the analysis → validates the SQL → executes it → returns the result and visualization.**

[Live Demo](https://insightpilot-analytics.streamlit.app/) · [Evaluation Suite](./evals) · [Case Study](https://devang-trivedi.vercel.app/projects/insightpilot)

---

## The problem

Business analysis often sits behind a technical interface.

To answer a seemingly simple question, a user may need to:

- understand the dataset schema
- identify the relevant metrics
- write SQL or dataframe logic
- validate the query
- build a visualization
- interpret the result

InsightPilot explores a simpler interaction model:

> **Ask the question in natural language.**

---

## How it works

```text
User question
      │
      ▼
┌─────────────────┐
│   LLM reasoning │
│  & plan creation│
└────────┬────────┘
         │
         ▼
┌─────────────────┐
│ SQL validation  │
│  & guardrails   │
└────────┬────────┘
         │
         ▼
┌─────────────────┐
│     DuckDB      │
│ read-only query │
└────────┬────────┘
         │
         ▼
┌─────────────────────────┐
│ Result + visualization  │
│ + analytical explanation│
└─────────────────────────┘
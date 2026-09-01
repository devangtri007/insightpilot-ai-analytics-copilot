
# Product Specification — InsightPilot

## Problem
Business users often have structured data but depend on analysts for simple questions. This creates queue time and repeated manual work.

## MVP hypothesis
If users can ask questions in natural language and inspect the generated SQL/results, they can answer routine analytical questions faster without sacrificing trust.

## User flow
1. Upload/select dataset
2. Ask question
3. AI produces structured analytical plan
4. Guardrail validates SQL
5. DuckDB executes read-only query
6. User sees answer, assumptions, SQL, result and chart
7. User can judge whether the output is useful

## AI evaluation rubric
Each response is scored on:
- Intent match: 0/1
- SQL safety: 0/1
- Execution success: 0/1
- Result non-empty: 0/1
- Answer correctness: 0/1 (future benchmark layer)

## Product metrics
Primary: Task success rate
Secondary: median time-to-insight, correction rate, execution success, trust/acceptance rate

## Risks
- Hallucinated columns
- Incorrect aggregation
- Unsafe SQL
- Misleading natural-language answer
- Ambiguous business definitions

## Mitigations
- Explicit schema
- Structured JSON response
- Read-only SQL validation
- Visible SQL and assumptions
- Evaluation benchmark
- Future business glossary/RAG

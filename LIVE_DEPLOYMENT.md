# Make InsightPilot Live

## 1. Test locally

Create a virtual environment and install dependencies:

```bash
python -m venv .venv
# Windows
.venv\\Scripts\\activate
# macOS/Linux
source .venv/bin/activate

pip install -r requirements.txt
```

Set your OpenAI API key as an environment variable:

### Windows PowerShell
```powershell
$env:OPENAI_API_KEY="sk-..."
$env:OPENAI_MODEL="gpt-5.6-luna"
streamlit run app.py
```

### macOS/Linux
```bash
export OPENAI_API_KEY="sk-..."
export OPENAI_MODEL="gpt-5.6-luna"
streamlit run app.py
```

Do NOT commit `.env`, API keys, or secrets to GitHub.

## 2. Deploy

The simplest portfolio deployment is Streamlit Community Cloud:

1. Create a GitHub repository.
2. Upload this project.
3. Create a Streamlit app pointing to `app.py`.
4. Add these secrets in the deployment settings:

```toml
OPENAI_API_KEY = "sk-..."
OPENAI_MODEL = "gpt-5.6-luna"
```

5. Deploy and open the generated HTTPS URL.

## 3. Demo flow for the interview

Use these questions:
- "Which products generated the most revenue?"
- "Compare revenue by region."
- "Which product has the highest weighted gross margin?"

Show the interviewer:
1. Natural-language question
2. Generated SQL
3. SQL guardrail
4. Query result
5. Visualization
6. Assumptions

Then explain that the next iteration would add a benchmark set, RAG-based metric definitions, feedback capture, and agentic multi-step analysis.

## 4. Security note

The prototype only permits SELECT/WITH SQL and rejects mutating/admin SQL keywords. This is a portfolio-level guardrail, not a production security boundary. Production deployment should additionally use a restricted database role, query timeouts, row/column permissions, logging, rate limits and a stronger SQL parser/AST validator.

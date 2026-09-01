
import os, re, json, io
from pathlib import Path

import duckdb
import pandas as pd
import streamlit as st

try:
    from openai import OpenAI
except Exception:
    OpenAI = None

st.set_page_config(page_title="InsightPilot", page_icon="📊", layout="wide")

SYSTEM_PROMPT = """
You are InsightPilot, an AI analytics copilot for business users.
The user is analyzing a synthetic pharma-commercial dataset. Never invent columns.
Return STRICT JSON with keys:
sql, answer, assumptions, chart_type, chart_x, chart_y.
SQL must be DuckDB-compatible and READ ONLY: SELECT/WITH only.
Do not use INSERT, UPDATE, DELETE, DROP, ALTER, CREATE, COPY, ATTACH, or PRAGMA.
Use the table name sales.
Keep SQL concise and aggregate where appropriate.
"""

SCHEMA = """
sales(
 month DATE,
 region VARCHAR,
 product VARCHAR,
 channel VARCHAR,
 units INTEGER,
 price_per_unit DOUBLE,
 revenue DOUBLE,
 gross_margin DOUBLE
)
"""

def clean_sql(sql: str) -> str:
    sql = sql.strip().replace("```sql", "").replace("```", "").strip()
    if ";" in sql.rstrip(";"):
        raise ValueError("Multiple SQL statements are not allowed.")
    sql = sql.rstrip(";").strip()
    if not re.match(r"^(SELECT|WITH)\b", sql, re.I):
        raise ValueError("Only SELECT/WITH queries are allowed.")
    blocked = r"\b(INSERT|UPDATE|DELETE|DROP|ALTER|CREATE|COPY|ATTACH|DETACH|PRAGMA|EXPORT|IMPORT)\b"
    if re.search(blocked, sql, re.I):
        raise ValueError("Unsafe SQL detected.")
    return sql

def local_demo(question: str):
    """Deterministic fallback so the prototype remains demoable without an API key."""
    q = question.lower()
    if "top" in q and "product" in q:
        sql = """SELECT product, ROUND(SUM(revenue),2) AS revenue
                 FROM sales GROUP BY product ORDER BY revenue DESC LIMIT 5"""
        answer = "Here are the top products by total revenue."
        return {"sql": sql, "answer": answer, "assumptions": ["Top means ranked by total revenue."],
                "chart_type":"bar", "chart_x":"product", "chart_y":"revenue"}
    if "region" in q and ("revenue" in q or "sales" in q):
        sql = """SELECT region, ROUND(SUM(revenue),2) AS revenue
                 FROM sales GROUP BY region ORDER BY revenue DESC"""
        return {"sql":sql, "answer":"Revenue is summarized by region.",
                "assumptions":["Sales means revenue."], "chart_type":"bar",
                "chart_x":"region","chart_y":"revenue"}
    if "margin" in q:
        sql = """SELECT product, ROUND(SUM(revenue*gross_margin)/SUM(revenue),3) AS weighted_margin
                 FROM sales GROUP BY product ORDER BY weighted_margin DESC"""
        return {"sql":sql, "answer":"Weighted gross margin is calculated by product.",
                "assumptions":["Margin is revenue-weighted rather than a simple average."],
                "chart_type":"bar","chart_x":"product","chart_y":"weighted_margin"}
    sql = """SELECT product, ROUND(SUM(revenue),2) AS revenue, SUM(units) AS units
             FROM sales GROUP BY product ORDER BY revenue DESC"""
    return {"sql":sql, "answer":"I used a product-level revenue and units summary for the demo.",
            "assumptions":["The question was mapped to a product performance view."],
            "chart_type":"bar","chart_x":"product","chart_y":"revenue"}

def get_secret(name, default=None):
    try:
        value = st.secrets.get(name)
        if value:
            return value
    except Exception:
        pass
    return os.getenv(name, default)

def ai_plan(question, schema):
    api_key = get_secret("OPENAI_API_KEY")
    model = get_secret("OPENAI_MODEL", "gpt-5.6-luna")
    if not api_key or OpenAI is None:
        return local_demo(question), "Demo mode"
    client = OpenAI(api_key=api_key)
    prompt = f"{SYSTEM_PROMPT}\nSCHEMA:\n{schema}\nUSER QUESTION:\n{question}"
    response = client.responses.create(
        model=model,
        input=prompt,
        text={"format": {"type": "json_object"}}
    )
    data = json.loads(response.output_text)
    data["sql"] = clean_sql(data["sql"])
    return data, f"Live OpenAI • {model}"

def execute(sql, df):
    con = duckdb.connect()
    con.register("sales", df)
    return con.execute(sql).df()

@st.cache_data
def load_default():
    return pd.read_csv(Path(__file__).parent / "data" / "pharma_sales_sample.csv", parse_dates=["month"])

st.title("📊 InsightPilot")
st.caption("AI-first analytics copilot • synthetic data • SQL generation • evaluation-ready")

with st.sidebar:
    st.header("Data")
    uploaded = st.file_uploader("Upload CSV", type=["csv"])
    st.divider()
    st.header("AI")
    live = bool(get_secret("OPENAI_API_KEY")) and OpenAI is not None
    mode = "Live LLM" if live else "Demo"
    st.info(f"Mode: **{mode}**")
    if live:
        st.caption("Connected to the OpenAI Responses API.")
    else:
        st.warning("Add OPENAI_API_KEY to enable live LLM mode.")
        st.caption("Demo mode still works without an API key.")

df = pd.read_csv(uploaded) if uploaded else load_default()
if "month" in df.columns:
    try: df["month"] = pd.to_datetime(df["month"])
    except: pass

st.write(f"**{len(df):,} rows** • {len(df.columns)} columns")
with st.expander("Preview dataset"):
    st.dataframe(df.head(20), use_container_width=True)

examples = [
    "Which products generated the most revenue?",
    "Compare revenue by region.",
    "Which product has the highest weighted gross margin?",
]
question = st.text_input("Ask a business question", placeholder="e.g. Which region generated the most revenue?")
if not question:
    st.markdown("**Try:** " + " · ".join(examples))
    st.stop()

if st.button("Analyze", type="primary"):
    with st.spinner("Planning query and analyzing data..."):
        try:
            plan, mode = ai_plan(question, SCHEMA)
            result = execute(clean_sql(plan["sql"]), df)
        except Exception as e:
            st.error(f"Could not safely execute the generated plan: {e}")
            st.stop()

    st.success(f"Completed in {mode}.")
    c1, c2 = st.columns([1,1])
    with c1:
        st.subheader("Answer")
        st.write(plan.get("answer",""))
        if plan.get("assumptions"):
            st.caption("Assumptions: " + " • ".join(plan["assumptions"]))
    with c2:
        st.subheader("Generated SQL")
        st.code(plan["sql"], language="sql")

    st.subheader("Result")
    st.dataframe(result, use_container_width=True)

    chart_type = plan.get("chart_type")
    x, y = plan.get("chart_x"), plan.get("chart_y")
    if chart_type == "bar" and x in result.columns and y in result.columns:
        st.bar_chart(result.set_index(x)[y])
    elif chart_type == "line" and x in result.columns and y in result.columns:
        st.line_chart(result.set_index(x)[y])

st.divider()
st.caption("Product prototype: user question → structured AI plan → read-only SQL → execution → evidence-backed answer.")

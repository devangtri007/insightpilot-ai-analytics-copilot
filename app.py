import os
import re
import json
import html
from pathlib import Path

import pandas as pd
import streamlit as st

try:
    import duckdb
except ImportError:
    duckdb = None

try:
    from openai import OpenAI
except ImportError:
    OpenAI = None

st.set_page_config(page_title="InsightPilot", page_icon="📊", layout="wide")

SYSTEM_PROMPT = """
You are InsightPilot, an AI analytics copilot for business users.
The user is analyzing a synthetic pharma-commercial dataset. Never invent columns.
Return STRICT JSON with exactly: sql, answer, assumptions, chart_type, chart_x, chart_y.
SQL must be DuckDB-compatible and READ ONLY: SELECT/WITH only.
Never use INSERT, UPDATE, DELETE, DROP, ALTER, CREATE, COPY, ATTACH, DETACH,
PRAGMA, EXPORT, IMPORT, MERGE, or TRUNCATE.
Use table sales.
Business definitions:
- revenue = sales revenue
- gross_margin = gross-margin percentage stored as a decimal
- weighted gross margin = SUM(revenue*gross_margin)/SUM(revenue)
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

def get_secret(name, default=None):
    try:
        value = st.secrets.get(name)
        if value:
            return value
    except Exception:
        pass
    return os.getenv(name, default)

def clean_sql(sql: str) -> str:
    if not isinstance(sql, str):
        raise ValueError("Generated SQL is not a string.")
    sql = sql.strip().replace("```sql", "").replace("```", "").strip()
    if ";" in sql.rstrip(";"):
        raise ValueError("Multiple SQL statements are not allowed.")
    sql = sql.rstrip(";").strip()
    if not re.match(r"^(SELECT|WITH)\b", sql, re.I):
        raise ValueError("Only SELECT/WITH queries are allowed.")
    blocked = r"\b(INSERT|UPDATE|DELETE|DROP|ALTER|CREATE|COPY|ATTACH|DETACH|PRAGMA|EXPORT|IMPORT|MERGE|TRUNCATE)\b"
    if re.search(blocked, sql, re.I):
        raise ValueError("Unsafe SQL detected.")
    return sql

@st.cache_data
def load_data():
    path = Path(__file__).parent / "data" / "pharma_sales_sample.csv"
    if not path.exists():
        raise FileNotFoundError(f"Dataset not found: {path}")
    df = pd.read_csv(path)
    if "month" in df.columns:
        df["month"] = pd.to_datetime(df["month"], errors="coerce")
    return df

def local_demo(question: str):
    q = question.lower()
    if ("product" in q and "revenue" in q) or ("top" in q and "product" in q):
        return {
            "sql": """SELECT product, ROUND(SUM(revenue),2) AS revenue
                      FROM sales GROUP BY product ORDER BY revenue DESC LIMIT 5""",
            "answer": "These are the top products ranked by total revenue.",
            "assumptions": ["Top products are ranked by total revenue."],
            "chart_type": "bar", "chart_x": "product", "chart_y": "revenue"
        }
    if "region" in q and ("revenue" in q or "sales" in q):
        return {
            "sql": """SELECT region, ROUND(SUM(revenue),2) AS revenue
                      FROM sales GROUP BY region ORDER BY revenue DESC""",
            "answer": "Revenue is summarized and ranked by region.",
            "assumptions": ["Sales is interpreted as revenue."],
            "chart_type": "bar", "chart_x": "region", "chart_y": "revenue"
        }
    if "margin" in q:
        return {
            "sql": """SELECT product,
                      ROUND(SUM(revenue*gross_margin)/NULLIF(SUM(revenue),0),3)
                      AS weighted_margin
                      FROM sales GROUP BY product ORDER BY weighted_margin DESC""",
            "answer": "Products are ranked by revenue-weighted gross margin.",
            "assumptions": [
                "Gross margin is stored as a decimal percentage.",
                "Revenue is used as the weighting factor."
            ],
            "chart_type": "bar", "chart_x": "product", "chart_y": "weighted_margin"
        }
    return {
        "sql": """SELECT product, ROUND(SUM(revenue),2) AS revenue,
                  SUM(units) AS units FROM sales
                  GROUP BY product ORDER BY revenue DESC""",
        "answer": "I mapped the question to a product-performance summary.",
        "assumptions": ["The question was interpreted at product level."],
        "chart_type": "bar", "chart_x": "product", "chart_y": "revenue"
    }

def ai_plan(question: str, schema: str = SCHEMA):
    api_key = get_secret("OPENAI_API_KEY")
    if not api_key or OpenAI is None:
        return local_demo(question), "Demo mode"
    model = get_secret("OPENAI_MODEL", "gpt-5.6-luna")
    client = OpenAI(api_key=api_key)
    prompt = f"{SYSTEM_PROMPT}\nSCHEMA:\n{schema}\nUSER QUESTION:\n{question}"
    response = client.responses.create(
        model=model,
        input=prompt,
        text={"format": {"type": "json_object"}}
    )
    data = json.loads(response.output_text)
    required = {"sql","answer","assumptions","chart_type","chart_x","chart_y"}
    missing = required - set(data)
    if missing:
        raise ValueError(f"LLM response missing fields: {sorted(missing)}")
    data["sql"] = clean_sql(data["sql"])
    return data, f"Live OpenAI • {model}"

def execute(sql: str, df: pd.DataFrame) -> pd.DataFrame:
    if duckdb is None:
        raise RuntimeError("DuckDB is not installed. Run pip install -r requirements.txt")
    con = duckdb.connect()
    try:
        con.register("sales", df)
        return con.execute(sql).df()
    finally:
        con.close()

def metric_card(label, value):
    st.markdown(
        f'<div class="metric"><div class="metric-label">{html.escape(str(label))}</div>'
        f'<div class="metric-value">{html.escape(str(value))}</div></div>',
        unsafe_allow_html=True
    )

st.markdown("""
<style>
@import url('https://fonts.googleapis.com/css2?family=Inter:wght@400;500;600;700;800&display=swap');
html,body,[class*="css"]{font-family:Inter,sans-serif}
.stApp{background:#f7f8fb}
.block-container{max-width:1240px;padding:2rem 2rem 4rem}
[data-testid="stSidebar"]{background:#111827}
[data-testid="stSidebar"] *{color:#e5e7eb!important}
.brand{display:flex;align-items:center;gap:10px;margin-bottom:22px}
.logo{width:36px;height:36px;border-radius:11px;background:#111827;color:#fff;display:flex;align-items:center;justify-content:center;font-weight:800}
.brand-name{font-size:21px;font-weight:800;color:#101828;letter-spacing:-.5px}
.badge{padding:5px 9px;border-radius:999px;background:#e9f7ef;color:#176b3a;font-size:10px;font-weight:800;letter-spacing:.4px}
.eyebrow{font-size:11px;font-weight:800;letter-spacing:1.5px;color:#667085;text-transform:uppercase}
.hero h1{font-size:46px;line-height:1.04;letter-spacing:-2.2px;margin:8px 0;color:#101828}
.hero p{font-size:16px;color:#667085;max-width:760px;margin-bottom:24px}
.metric{background:#fff;border:1px solid #e4e7ec;border-radius:15px;padding:17px;box-shadow:0 6px 24px rgba(16,24,40,.04);min-height:82px}
.metric-label{font-size:10px;font-weight:800;color:#667085;letter-spacing:.7px}
.metric-value{font-size:23px;font-weight:800;color:#101828;margin-top:5px;overflow-wrap:anywhere}
.card{background:#fff;border:1px solid #e4e7ec;border-radius:18px;padding:20px;box-shadow:0 6px 24px rgba(16,24,40,.04)}
.answer{background:#111827;color:#fff;border-radius:18px;padding:24px;margin-bottom:8px}
.answer-label{color:#a7f3d0;font-size:10px;font-weight:800;letter-spacing:1px}
.answer-text{font-size:20px;line-height:1.48;margin-top:8px}
.section{font-size:11px;font-weight:800;letter-spacing:1px;color:#667085;text-transform:uppercase;margin:26px 0 10px}
.small{font-size:12px;color:#667085}
.trust{font-size:12px;line-height:1.55;color:#98a2b3}
</style>
""", unsafe_allow_html=True)

try:
    with st.sidebar:
        st.markdown('<div style="font-size:24px;font-weight:800">◉ InsightPilot</div>', unsafe_allow_html=True)
        st.caption("AI-first analytics copilot")
        st.divider()
        st.markdown("**Workspace**")
        uploaded = st.file_uploader("Upload a CSV", type=["csv"])
        if uploaded is not None:
            df = pd.read_csv(uploaded)
            if "month" in df.columns:
                df["month"] = pd.to_datetime(df["month"], errors="coerce")
        else:
            df = load_data()
        st.caption(f"{len(df):,} rows · {len(df.columns)} columns")
        live = bool(get_secret("OPENAI_API_KEY")) and OpenAI is not None
        st.success("Live LLM connected") if live else st.info("Demo mode")
        st.divider()
        st.markdown("**Trust & safety**")
        st.caption("Read-only SQL · visible assumptions · synthetic data")
        st.divider()
        st.markdown('<div class="trust">Portfolio prototype · No real customer data · No PHI</div>', unsafe_allow_html=True)
except Exception as exc:
    st.error(str(exc))
    st.stop()

st.markdown("""
<div class="brand"><div class="logo">◉</div><div class="brand-name">InsightPilot</div>
<span class="badge">AI ANALYTICS</span></div>
<div class="hero"><div class="eyebrow">AI-first analytics copilot</div>
<h1>Ask your data anything.</h1>
<p>Turn business questions into transparent, evidence-backed analysis — without writing SQL.</p></div>
""", unsafe_allow_html=True)

total_revenue = float(df["revenue"].sum()) if "revenue" in df.columns else 0
product_count = int(df["product"].nunique()) if "product" in df.columns else 0
region_count = int(df["region"].nunique()) if "region" in df.columns else 0

k1,k2,k3,k4 = st.columns(4)
with k1: metric_card("DATA ROWS", f"{len(df):,}")
with k2: metric_card("TOTAL REVENUE", f"{total_revenue:,.0f}")
with k3: metric_card("PRODUCTS", product_count)
with k4: metric_card("REGIONS", region_count)

st.markdown('<div class="section">Ask a question</div>', unsafe_allow_html=True)
st.markdown('<div class="card">', unsafe_allow_html=True)

question = st.text_input(
    "Business question",
    placeholder="e.g. Which products generated the most revenue?",
    label_visibility="collapsed"
)
st.caption("Start with one of these common analyses:")

s1,s2,s3 = st.columns(3)
suggested = None
with s1:
    if st.button("Top products by revenue", width="stretch"):
        suggested = "Which products generated the most revenue?"
with s2:
    if st.button("Revenue by region", width="stretch"):
        suggested = "Compare revenue by region."
with s3:
    if st.button("Highest weighted margin", width="stretch"):
        suggested = "Which product has the highest weighted gross margin?"

active_question = suggested or question
analyze = st.button("✦  Analyze with AI", type="primary", width="stretch")
st.markdown("</div>", unsafe_allow_html=True)

if analyze and active_question:
    with st.spinner("AI is planning the analysis…"):
        try:
            plan, mode = ai_plan(active_question)
            safe_sql = clean_sql(plan["sql"])
            result = execute(safe_sql, df)
        except Exception as exc:
            st.error(f"Analysis failed: {exc}")
            st.stop()

    st.markdown('<div class="section">AI answer</div>', unsafe_allow_html=True)
    st.markdown(
        f'<div class="answer"><div class="answer-label">{html.escape(str(mode).upper())}</div>'
        f'<div class="answer-text">{html.escape(str(plan.get("answer","")))}</div></div>',
        unsafe_allow_html=True
    )

    if not result.empty:
        numeric = [c for c in result.columns if pd.api.types.is_numeric_dtype(result[c])]
        top_result = result.iloc[0].iloc[0]
        top_value = result.iloc[0][numeric[0]] if numeric else ""
        st.markdown('<div class="section">Key result</div>', unsafe_allow_html=True)
        r1,r2,r3 = st.columns(3)
        with r1: metric_card("TOP RESULT", top_result)
        with r2: metric_card("TOP VALUE", f"{top_value:,.2f}" if isinstance(top_value,(int,float)) else top_value)
        with r3: metric_card("ROWS RETURNED", len(result))

    st.markdown('<div class="section">Evidence</div>', unsafe_allow_html=True)
    left,right = st.columns([1.45,1])

    with left:
        st.dataframe(result, width="stretch", hide_index=True)
        x,y = plan.get("chart_x"), plan.get("chart_y")
        if plan.get("chart_type") == "bar" and x in result.columns and y in result.columns:
            st.bar_chart(result.set_index(x)[y], width="stretch")
        elif plan.get("chart_type") == "line" and x in result.columns and y in result.columns:
            st.line_chart(result.set_index(x)[y], width="stretch")
        st.download_button(
            "Download result CSV",
            data=result.to_csv(index=False).encode("utf-8"),
            file_name="insightpilot_result.csv",
            mime="text/csv",
            width="stretch"
        )

    with right:
        st.markdown("**How InsightPilot calculated this**")
        for assumption in plan.get("assumptions", []):
            st.markdown(f"• {assumption}")
        with st.expander("Generated SQL"):
            st.code(safe_sql, language="sql")
        st.caption("Results are computed from the dataset. SQL and assumptions remain visible for verification.")

    st.markdown('<div class="section">Was this useful?</div>', unsafe_allow_html=True)
    f1,f2,_ = st.columns([1,1,6])
    with f1: st.button("👍 Helpful")
    with f2: st.button("👎 Needs work")

else:
    st.markdown('<div class="section">How it works</div>', unsafe_allow_html=True)
    h1,h2,h3 = st.columns(3)
    with h1:
        st.markdown('<div class="card"><b>01 · Ask</b><br><span class="small">Describe the business question in plain English.</span></div>', unsafe_allow_html=True)
    with h2:
        st.markdown('<div class="card"><b>02 · Analyze</b><br><span class="small">AI creates a structured plan and safe, read-only SQL.</span></div>', unsafe_allow_html=True)
    with h3:
        st.markdown('<div class="card"><b>03 · Verify</b><br><span class="small">Inspect assumptions, SQL, evidence and visualization.</span></div>', unsafe_allow_html=True)

st.markdown(
    '<div class="small" style="margin-top:30px">InsightPilot is a portfolio prototype using synthetic commercial data. No real customer or PHI data is used.</div>',
    unsafe_allow_html=True
)


import os, re, json, io
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

def ai_plan(question, schema=SCHEMA):
    if not get_secret("OPENAI_API_KEY") or OpenAI is None:
        return local_demo(question), "Demo mode"
    client = OpenAI(api_key=get_secret("OPENAI_API_KEY"))
    prompt = f"{SYSTEM_PROMPT}\nSCHEMA:\n{schema}\nUSER QUESTION:\n{question}"
    model = get_secret("OPENAI_MODEL", "gpt-5.6-luna")
    response = client.responses.create(
        model=model,
        input=prompt,
        text={"format": {"type": "json_object"}}
    )
    raw = response.output_text
    data = json.loads(raw)
    data["sql"] = clean_sql(data["sql"])
    return data, f"Live OpenAI • {model}"

def execute(sql, df):
    if duckdb is None:
        raise RuntimeError("DuckDB is not installed. Run: pip install -r requirements.txt")
    con = duckdb.connect()
    con.register("sales", df)
    return con.execute(sql).df()

@st.cache_data
def load_data():
    return pd.read_csv(Path(__file__).parent / "data" / "pharma_sales_sample.csv", parse_dates=["month"])

st.title("📊 InsightPilot")
st.caption("AI-first analytics copilot • synthetic data • SQL generation • evaluation-ready")

with st.sidebar:
    st.header("Data")
    uploaded = st.file_uploader("Upload CSV", type=["csv"])
    st.divider()
    st.header("AI")
    live = bool(os.getenv("OPENAI_API_KEY")) and OpenAI is not None
    mode = "Live LLM" if live else "Demo"
    st.info(f"Mode: **{mode}**")
    if live:
        st.caption("Connected to the OpenAI Responses API.")
    else:
        st.warning("Add OPENAI_API_KEY to enable live LLM mode.")
        st.caption("Demo mode still works without an API key.")

df = pd.read_csv(uploaded) if uploaded else load_data()
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
# ---------- Product UI ----------
st.markdown("""
<style>
@import url('https://fonts.googleapis.com/css2?family=Inter:wght@400;500;600;700;800&display=swap');
html,body,[class*="css"]{font-family:Inter,sans-serif}
.stApp{background:#f7f8fb}.block-container{max-width:1240px;padding:2rem 2rem 4rem}
[data-testid="stSidebar"]{background:#111827}[data-testid="stSidebar"] *{color:#e5e7eb!important}
.brand{display:flex;align-items:center;gap:10px;margin-bottom:18px}.logo{width:34px;height:34px;border-radius:10px;background:#111827;color:#fff;display:flex;align-items:center;justify-content:center;font-weight:800}.brand-name{font-size:20px;font-weight:800;color:#111827}.badge{padding:5px 9px;border-radius:999px;background:#e9f7ef;color:#176b3a;font-size:11px;font-weight:800}
.eyebrow{font-size:11px;font-weight:800;letter-spacing:1.5px;color:#667085;text-transform:uppercase}.hero h1{font-size:44px;line-height:1.05;letter-spacing:-2px;margin:8px 0;color:#101828}.hero p{font-size:16px;color:#667085;max-width:720px}
.metric,.card{background:#fff;border:1px solid #e4e7ec;border-radius:16px;padding:17px;box-shadow:0 6px 24px rgba(16,24,40,.04)}.metric-label{font-size:11px;font-weight:700;color:#667085;letter-spacing:.5px}.metric-value{font-size:23px;font-weight:800;color:#101828;margin-top:4px}
.answer{background:#111827;color:#fff;border-radius:18px;padding:24px}.answer-label{color:#a7f3d0;font-size:11px;font-weight:800;letter-spacing:1px}.answer-text{font-size:20px;line-height:1.45;margin-top:8px}.section{font-size:12px;font-weight:800;letter-spacing:1px;color:#667085;text-transform:uppercase;margin:25px 0 10px}.small{font-size:12px;color:#667085}
</style>
""",unsafe_allow_html=True)

with st.sidebar:
    st.markdown('<div style="font-size:24px;font-weight:800">◉ InsightPilot</div>',unsafe_allow_html=True)
    st.caption("AI-first analytics copilot")
    st.divider()
    uploaded=st.file_uploader("Dataset",type=["csv"])
    df=pd.read_csv(uploaded) if uploaded else load_data()
    st.markdown("**Workspace**")
    st.caption(f"{len(df):,} rows · {len(df.columns)} columns")
    live=bool(get_secret("OPENAI_API_KEY")) and OpenAI is not None
    if live: st.success("Live LLM connected")
    else: st.info("Demo mode")
    st.divider()
    st.markdown("**Trust & safety**")
    st.caption("Read-only SQL · visible assumptions · synthetic data")
    st.divider()
    st.caption("Portfolio prototype · No PHI")

st.markdown("""<div class="brand"><div class="logo">◉</div><div class="brand-name">InsightPilot</div><span class="badge">AI ANALYTICS</span></div>
<div class="hero"><div class="eyebrow">AI-first analytics copilot</div><h1>Ask your data anything.</h1><p>Turn business questions into transparent, evidence-backed analysis — without writing SQL.</p></div>""",unsafe_allow_html=True)

total_rev=float(df["revenue"].sum()) if "revenue" in df.columns else 0
products=int(df["product"].nunique()) if "product" in df.columns else 0
regions=int(df["region"].nunique()) if "region" in df.columns else 0
for c,l,v in zip(st.columns(4),["DATA ROWS","TOTAL REVENUE","PRODUCTS","REGIONS"],[f"{len(df):,}",f"{total_rev:,.0f}",str(products),str(regions)]):
    c.markdown(f'<div class="metric"><div class="metric-label">{l}</div><div class="metric-value">{v}</div></div>',unsafe_allow_html=True)

st.markdown('<div class="section">Ask a question</div>',unsafe_allow_html=True)
st.markdown('<div class="card">',unsafe_allow_html=True)
q=st.text_input("",placeholder="e.g. Which products generated the most revenue?",label_visibility="collapsed")
b1,b2,b3=st.columns(3); clicked=None
if b1.button("Top products by revenue",use_container_width=True): clicked="Which products generated the most revenue?"
if b2.button("Revenue by region",use_container_width=True): clicked="Compare revenue by region."
if b3.button("Highest weighted margin",use_container_width=True): clicked="Which product has the highest weighted gross margin?"
question=clicked or q
analyze=st.button("✦  Analyze with AI",type="primary",use_container_width=True)
st.markdown('</div>',unsafe_allow_html=True)

if analyze and question:
    with st.spinner("AI is planning the analysis…"):
        try:
            plan,mode=ai_plan(question, SCHEMA)
            result=execute(clean_sql(plan["sql"]),df)
        except Exception as e:
            st.error(str(e)); st.stop()
    st.markdown('<div class="section">AI answer</div>',unsafe_allow_html=True)
    st.markdown(f'<div class="answer"><div class="answer-label">{mode.upper()}</div><div class="answer-text">{plan.get("answer","")}</div></div>',unsafe_allow_html=True)
    if len(result):
        nums=[x for x in result.columns if pd.api.types.is_numeric_dtype(result[x])]
        val=result.iloc[0][nums[0]] if nums else ""
        a,b,c=st.columns(3)
        a.markdown(f'<div class="metric"><div class="metric-label">TOP RESULT</div><div class="metric-value">{result.iloc[0].iloc[0]}</div></div>',unsafe_allow_html=True)
        b.markdown(f'<div class="metric"><div class="metric-label">TOP VALUE</div><div class="metric-value">{val:,.2f}</div></div>' if isinstance(val,(int,float)) else f'<div class="metric"><div class="metric-label">TOP VALUE</div><div class="metric-value">{val}</div></div>',unsafe_allow_html=True)
        c.markdown(f'<div class="metric"><div class="metric-label">ROWS RETURNED</div><div class="metric-value">{len(result)}</div></div>',unsafe_allow_html=True)
    st.markdown('<div class="section">Evidence</div>',unsafe_allow_html=True)
    left,right=st.columns([1.45,1])
    with left:
        st.dataframe(result,use_container_width=True,hide_index=True)
        x,y=plan.get("chart_x"),plan.get("chart_y")
        if x in result.columns and y in result.columns: st.bar_chart(result.set_index(x)[y])
    with right:
        st.markdown("**How InsightPilot calculated this**")
        for a in plan.get("assumptions",[]): st.markdown("• "+a)
        with st.expander("Generated SQL"): st.code(plan["sql"],language="sql")
        st.caption("Transparent evidence lets users inspect the analysis instead of blindly trusting the model.")
    st.markdown('<div class="section">Was this useful?</div>',unsafe_allow_html=True)
    f1,f2,_=st.columns([1,1,6]); f1.button("👍 Helpful"); f2.button("👎 Needs work")
else:
    st.markdown('<div class="section">How it works</div>',unsafe_allow_html=True)
    h1,h2,h3=st.columns(3)
    h1.markdown('<div class="card"><b>01 · Ask</b><br><span class="small">Describe the business question in plain English.</span></div>',unsafe_allow_html=True)
    h2.markdown('<div class="card"><b>02 · Analyze</b><br><span class="small">AI creates a structured plan and safe read-only SQL.</span></div>',unsafe_allow_html=True)
    h3.markdown('<div class="card"><b>03 · Verify</b><br><span class="small">Inspect assumptions, SQL, evidence and visualization.</span></div>',unsafe_allow_html=True)

st.markdown('<div class="small" style="margin-top:28px">InsightPilot is a portfolio prototype using synthetic commercial data. No real customer or PHI data is used.</div>',unsafe_allow_html=True)

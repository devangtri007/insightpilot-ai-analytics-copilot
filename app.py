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


# ============================================================
# PAGE CONFIG
# ============================================================

st.set_page_config(
    page_title="InsightPilot",
    page_icon="📊",
    layout="wide",
    initial_sidebar_state="expanded",
)


# ============================================================
# GENERAL HELPERS
# ============================================================

def get_secret(name, default=None):
    try:
        value = st.secrets.get(name)
        if value:
            return value
    except Exception:
        pass
    return os.getenv(name, default)


def read_uploaded_file(uploaded_file):
    """Read CSV or Excel without assuming a particular business domain."""
    suffix = Path(uploaded_file.name).suffix.lower()

    if suffix == ".csv":
        df = pd.read_csv(uploaded_file)
    elif suffix in {".xlsx", ".xls"}:
        df = pd.read_excel(uploaded_file)
    else:
        raise ValueError("Please upload a CSV, XLSX, or XLS file.")

    # Normalize completely empty rows/columns.
    df = df.dropna(axis=0, how="all").dropna(axis=1, how="all")

    # Convert obvious date columns where possible.
    for col in df.columns:
        if df[col].dtype == "object":
            sample = df[col].dropna().astype(str)
            if len(sample) > 0:
                parsed = pd.to_datetime(sample, errors="coerce")
                if parsed.notna().mean() >= 0.85:
                    df[col] = pd.to_datetime(df[col], errors="coerce")

    return df


@st.cache_data
def load_default():
    """Load the bundled sample dataset if present."""
    path = Path(__file__).parent / "data" / "pharma_sales_sample.csv"

    if not path.exists():
        # The application remains usable with an upload even if no sample exists.
        return pd.DataFrame()

    df = pd.read_csv(path)

    if "month" in df.columns:
        df["month"] = pd.to_datetime(df["month"], errors="coerce")

    return df


def dataframe_schema(df):
    """Create an LLM-friendly schema from the actual uploaded dataset."""
    lines = ["Table: data"]

    for col in df.columns:
        dtype = str(df[col].dtype)

        if pd.api.types.is_datetime64_any_dtype(df[col]):
            sql_type = "DATE"
        elif pd.api.types.is_integer_dtype(df[col]):
            sql_type = "INTEGER"
        elif pd.api.types.is_float_dtype(df[col]):
            sql_type = "DOUBLE"
        elif pd.api.types.is_bool_dtype(df[col]):
            sql_type = "BOOLEAN"
        else:
            sql_type = "VARCHAR"

        lines.append(f"- {col}: {sql_type}")

    return "\n".join(lines)


def clean_identifier(name):
    """Quote a DuckDB identifier safely."""
    return '"' + str(name).replace('"', '""') + '"'


def clean_sql(sql: str) -> str:
    if not isinstance(sql, str):
        raise ValueError("Generated SQL is not a string.")

    sql = sql.strip()
    sql = sql.replace("```sql", "").replace("```", "").strip()

    if ";" in sql.rstrip(";"):
        raise ValueError("Multiple SQL statements are not allowed.")

    sql = sql.rstrip(";").strip()

    if not re.match(r"^(SELECT|WITH)\b", sql, re.IGNORECASE):
        raise ValueError("Only SELECT/WITH queries are allowed.")

    blocked = (
        r"\b(INSERT|UPDATE|DELETE|DROP|ALTER|CREATE|COPY|ATTACH|"
        r"DETACH|PRAGMA|EXPORT|IMPORT|MERGE|TRUNCATE|INSTALL|LOAD)\b"
    )

    if re.search(blocked, sql, re.IGNORECASE):
        raise ValueError("Unsafe SQL detected.")

    return sql


def execute(sql, df):
    if duckdb is None:
        raise RuntimeError(
            "DuckDB is not installed. Run: pip install -r requirements.txt"
        )

    con = duckdb.connect()
    try:
        con.register("data", df)
        return con.execute(sql).df()
    finally:
        con.close()


def metric_card(label, value):
    st.markdown(
        f"""
        <div class="metric">
            <div class="metric-label">{html.escape(str(label))}</div>
            <div class="metric-value">{html.escape(str(value))}</div>
        </div>
        """,
        unsafe_allow_html=True,
    )


# ============================================================
# DYNAMIC DATASET METRICS
# ============================================================

def dataset_metrics(df):
    numeric_cols = [
        c for c in df.columns
        if pd.api.types.is_numeric_dtype(df[c])
        and not pd.api.types.is_bool_dtype(df[c])
    ]

    missing = int(df.isna().sum().sum())

    return {
        "rows": len(df),
        "columns": len(df.columns),
        "numeric": len(numeric_cols),
        "missing": missing,
        "numeric_cols": numeric_cols,
    }


def numeric_preview_text(df, column):
    series = pd.to_numeric(df[column], errors="coerce").dropna()

    if series.empty:
        return None

    return {
        "sum": float(series.sum()),
        "mean": float(series.mean()),
        "min": float(series.min()),
        "max": float(series.max()),
    }


# ============================================================
# GENERIC DEMO MODE
# ============================================================

def local_demo(question, df):
    """Provide useful deterministic analysis when no LLM key is configured."""
    q = question.lower()
    numeric_cols = [
        c for c in df.columns
        if pd.api.types.is_numeric_dtype(df[c])
        and not pd.api.types.is_bool_dtype(df[c])
    ]
    categorical_cols = [
        c for c in df.columns
        if (
            pd.api.types.is_object_dtype(df[c])
            or pd.api.types.is_categorical_dtype(df[c])
        )
    ]

    if not len(df.columns):
        raise ValueError("There is no dataset to analyze.")

    # Pick a sensible numeric field for generic aggregations.
    numeric = numeric_cols[0] if numeric_cols else None
    categorical = categorical_cols[0] if categorical_cols else None

    if numeric and categorical and any(
        term in q for term in ["highest", "lowest", "most", "least", "top", "best"]
    ):
        c = clean_identifier(categorical)
        n = clean_identifier(numeric)

        sql = f"""
        SELECT {c} AS "{categorical}",
               ROUND(SUM({n}), 2) AS value
        FROM data
        GROUP BY {c}
        ORDER BY value DESC
        LIMIT 10
        """

        return {
            "sql": sql,
            "answer": f"I ranked {categorical} values using total {numeric}.",
            "assumptions": [
                f"{numeric} was treated as the measure.",
                f"{categorical} was treated as the grouping dimension."
            ],
            "chart_type": "bar",
            "chart_x": categorical,
            "chart_y": "value",
        }

    if numeric and any(
        term in q for term in ["average", "avg", "mean"]
    ):
        n = clean_identifier(numeric)
        sql = f"""
        SELECT ROUND(AVG({n}), 2) AS average_value
        FROM data
        """

        return {
            "sql": sql,
            "answer": f"The average {numeric} is shown in the result.",
            "assumptions": [f"{numeric} was interpreted as the requested measure."],
            "chart_type": "none",
            "chart_x": "",
            "chart_y": "",
        }

    if numeric and any(
        term in q for term in ["total", "sum", "overall"]
    ):
        n = clean_identifier(numeric)
        sql = f"""
        SELECT ROUND(SUM({n}), 2) AS total_value
        FROM data
        """

        return {
            "sql": sql,
            "answer": f"The total {numeric} is shown in the result.",
            "assumptions": [f"{numeric} was interpreted as the requested measure."],
            "chart_type": "none",
            "chart_x": "",
            "chart_y": "",
        }

    if categorical:
        c = clean_identifier(categorical)
        sql = f"""
        SELECT {c} AS "{categorical}",
               COUNT(*) AS row_count
        FROM data
        GROUP BY {c}
        ORDER BY row_count DESC
        LIMIT 10
        """

        return {
            "sql": sql,
            "answer": f"I summarized the dataset by {categorical}.",
            "assumptions": [
                f"{categorical} was selected as the first categorical dimension."
            ],
            "chart_type": "bar",
            "chart_x": categorical,
            "chart_y": "row_count",
        }

    if numeric:
        n = clean_identifier(numeric)
        sql = f"""
        SELECT
            ROUND(SUM({n}), 2) AS total_value,
            ROUND(AVG({n}), 2) AS average_value,
            ROUND(MIN({n}), 2) AS minimum_value,
            ROUND(MAX({n}), 2) AS maximum_value
        FROM data
        """

        return {
            "sql": sql,
            "answer": f"I generated a summary of the numeric field {numeric}.",
            "assumptions": [f"{numeric} was selected as the available numeric measure."],
            "chart_type": "none",
            "chart_x": "",
            "chart_y": "",
        }

    sql = "SELECT * FROM data LIMIT 10"

    return {
        "sql": sql,
        "answer": "I returned a sample of the uploaded dataset.",
        "assumptions": ["No obvious numeric or categorical analysis field was available."],
        "chart_type": "none",
        "chart_x": "",
        "chart_y": "",
    }


# ============================================================
# AI PLANNER
# ============================================================

def ai_plan(question, df):
    api_key = get_secret("OPENAI_API_KEY")

    if not api_key or OpenAI is None:
        return local_demo(question, df), "Demo mode"

    model = get_secret("OPENAI_MODEL", "gpt-5.6-luna")
    schema = dataframe_schema(df)

    system_prompt = f"""
You are InsightPilot, a general-purpose AI analytics copilot.

The user can upload arbitrary CSV or Excel datasets.
You MUST reason from the supplied schema and MUST NOT assume a business domain.

{schema}

Return STRICT JSON with exactly these keys:
sql, answer, assumptions, chart_type, chart_x, chart_y.

SQL rules:
- DuckDB-compatible.
- READ ONLY.
- Must begin with SELECT or WITH.
- Use table name data.
- Quote column names when necessary.
- Never invent columns.
- Never use INSERT, UPDATE, DELETE, DROP, ALTER, CREATE, COPY,
  ATTACH, DETACH, PRAGMA, EXPORT, IMPORT, MERGE, TRUNCATE, INSTALL, or LOAD.
- Keep SQL concise and appropriate to the question.
- Prefer aggregation for analytical questions.

chart_type must be "bar", "line", or "none".
chart_x and chart_y must refer to columns returned by the SQL, or be empty strings.
The answer should directly answer the user's question without fabricating facts.
Assumptions should mention meaningful interpretations only.
"""

    client = OpenAI(api_key=api_key)

    response = client.responses.create(
        model=model,
        input=f"{system_prompt}\n\nUSER QUESTION:\n{question}",
        text={"format": {"type": "json_object"}},
    )

    data = json.loads(response.output_text)

    required = {
        "sql",
        "answer",
        "assumptions",
        "chart_type",
        "chart_x",
        "chart_y",
    }

    missing = required - set(data)
    if missing:
        raise ValueError(f"LLM response missing fields: {sorted(missing)}")

    data["sql"] = clean_sql(data["sql"])

    return data, f"Live OpenAI • {model}"


# ============================================================
# PRODUCT UI
# ============================================================

st.markdown(
    """
<style>
@import url('https://fonts.googleapis.com/css2?family=Inter:wght@400;500;600;700;800&display=swap');

html, body, [class*="css"] {
    font-family: Inter, sans-serif;
}

.stApp {
    background: #f7f8fb;
}

.block-container {
    max-width: 1240px;
    padding: 4.75rem 2rem 4rem;
}

[data-testid="stSidebar"] {
    background: #111827;
}

[data-testid="stSidebar"] * {
    color: #e5e7eb !important;
}

.top-safe-space {
    height: 18px;
}

.brand {
    display: flex;
    align-items: center;
    gap: 10px;
    margin: 0 0 28px 0;
    min-height: 44px;
    overflow: visible;
}

.main-brand {
    margin-bottom: 30px;
}

.main-brand .logo {
    flex: 0 0 40px;
}

.main-brand .brand-name {
    line-height: 1;
}

.logo {
    width: 40px;
    min-width: 40px;
    height: 40px;
    min-height: 40px;
    border-radius: 11px;
    background: #111827;
    color: #fff;
    display: flex;
    align-items: center;
    justify-content: center;
    font-weight: 800;
}

.brand-name {
    font-size: 21px;
    font-weight: 800;
    color: #101828;
    letter-spacing: -.5px;
}

.badge {
    padding: 5px 9px;
    border-radius: 999px;
    background: #e9f7ef;
    color: #176b3a;
    font-size: 10px;
    font-weight: 800;
    letter-spacing: .4px;
}

.hero {
    margin: 0 0 30px 0;
    padding-top: 2px;
}

.hero h1 {
    margin-top: 9px;
}

.eyebrow {
    font-size: 11px;
    font-weight: 800;
    letter-spacing: 1.5px;
    color: #667085;
    text-transform: uppercase;
}

.hero h1 {
    font-size: 46px;
    line-height: 1.04;
    letter-spacing: -2.2px;
    margin: 9px 0;
    color: #101828;
}

.hero p {
    font-size: 16px;
    color: #667085;
    max-width: 760px;
    margin: 0;
}

.section {
    font-size: 11px;
    font-weight: 800;
    letter-spacing: 1px;
    color: #667085;
    text-transform: uppercase;
    margin: 26px 0 10px;
}

.metric {
    background: #fff;
    border: 1px solid #e4e7ec;
    border-radius: 15px;
    padding: 17px;
    box-shadow: 0 6px 24px rgba(16,24,40,.04);
    min-height: 82px;
}

.metric-label {
    font-size: 10px;
    font-weight: 800;
    color: #667085;
    letter-spacing: .7px;
}

.metric-value {
    font-size: 23px;
    font-weight: 800;
    color: #101828;
    margin-top: 5px;
    overflow-wrap: anywhere;
}

.card {
    background: #fff;
    border: 1px solid #e4e7ec;
    border-radius: 18px;
    padding: 20px;
    box-shadow: 0 6px 24px rgba(16,24,40,.04);
}

.answer {
    background: #111827;
    color: #fff;
    border-radius: 18px;
    padding: 24px;
}

.answer-label {
    color: #a7f3d0;
    font-size: 10px;
    font-weight: 800;
    letter-spacing: 1px;
}

.answer-text {
    font-size: 20px;
    line-height: 1.48;
    margin-top: 8px;
}

.small {
    font-size: 12px;
    color: #667085;
}

div[data-testid="stTextInput"] {
    margin-bottom: 6px;
}

div[data-testid="stTextInput"] input {
    min-height: 52px;
    border-radius: 12px;
    border: 1px solid #d0d5dd;
    padding: 0 16px;
    font-size: 15px;
    color: #101828;
    background: #fff;
}

div[data-testid="stTextInput"] input:focus {
    border-color: #98a2b3;
    box-shadow: 0 0 0 1px #98a2b3;
}

div[data-testid="stButton"] > button {
    min-height: 44px;
    border-radius: 10px;
    font-size: 13px;
    font-weight: 500;
}

.trust {
    font-size: 12px;
    line-height: 1.55;
    color: #98a2b3;
}

/* Sidebar uploader: keep the control visually integrated with the dark sidebar. */
[data-testid="stSidebar"] [data-testid="stFileUploader"] {
    color: #e5e7eb !important;
    background: #182235;
    border: 1px solid #2b3950;
    border-radius: 12px;
    padding: 10px;
}

[data-testid="stSidebar"] [data-testid="stFileUploaderDropzone"] {
    background: #182235 !important;
    border: 1px solid #334155 !important;
}

[data-testid="stSidebar"] [data-testid="stFileUploaderFile"] {
    background: #202c40 !important;
    color: #e5e7eb !important;
}

[data-testid="stSidebar"] [data-testid="stFileUploader"] section {
    background: #182235 !important;
    border: 1px solid #334155 !important;
    border-radius: 9px !important;
    padding: 8px !important;
}

[data-testid="stSidebar"] [data-testid="stFileUploader"] section > div {
    color: #d7deea !important;
}

[data-testid="stSidebar"] [data-testid="stFileUploader"] small {
    color: #9aa7bb !important;
}

[data-testid="stSidebar"] [data-testid="stFileUploader"] button {
    background: #25324a !important;
    color: #f8fafc !important;
    border: 1px solid #40506a !important;
}

[data-testid="stSidebar"] [data-testid="stFileUploader"] button:hover {
    background: #30405c !important;
    border-color: #5a6d8c !important;
}

[data-testid="stSidebar"] [data-testid="stFileUploader"] [data-testid="stFileUploaderDropzone"] {
    background: #182235 !important;
}

@media (max-width: 900px) {
    .top-safe-space { height: 10px; }
    .hero h1 {
        font-size: 38px;
    }

    .block-container {
        padding-left: 1rem;
        padding-right: 1rem;
    }
}
</style>
""",
    unsafe_allow_html=True,
)


# ============================================================
# SIDEBAR / DATA LOADING
# ============================================================

with st.sidebar:
    st.markdown(
        '<div style="font-size:24px;font-weight:800">◉ InsightPilot</div>',
        unsafe_allow_html=True,
    )
    st.caption("AI-first analytics copilot")
    st.divider()

    st.markdown('<div style="font-size:15px;font-weight:700;margin-bottom:8px">Workspace</div>', unsafe_allow_html=True)

    uploaded = st.file_uploader(
        "Upload a dataset",
        type=["csv", "xlsx", "xls"],
        help="Upload a CSV or Excel file.",
    )

    if uploaded is not None:
        try:
            df = read_uploaded_file(uploaded)
        except Exception as exc:
            st.error(f"Could not read the file: {exc}")
            st.stop()
    else:
        df = load_default()

    if df.empty:
        st.info("Upload a CSV or Excel file to start analyzing.")
    else:
        st.caption(f"{len(df):,} rows · {len(df.columns):,} columns")

    live = bool(get_secret("OPENAI_API_KEY")) and OpenAI is not None

    if live:
        st.success("Live LLM connected")
    else:
        st.info("Demo mode")

    st.divider()
    st.markdown("**Trust & safety**")
    st.caption("Read-only SQL · dataset-grounded answers · visible assumptions")
    st.divider()
    st.markdown(
        '<div class="trust">Generic analytics prototype · Works across CSV/XLSX datasets</div>',
        unsafe_allow_html=True,
    )


# ============================================================
# HEADER
# ============================================================

st.markdown(
    """
<div class="top-safe-space"></div>
<div class="brand main-brand">
    <div class="logo">◉</div>
    <div class="brand-name">InsightPilot</div>
    <span class="badge">AI ANALYTICS</span>
</div>

<div class="hero">
    <div class="eyebrow">AI-first analytics copilot</div>
    <h1>Ask your data anything.</h1>
    <p>
        Turn questions about any uploaded dataset into transparent,
        evidence-backed analysis — without writing SQL.
    </p>
</div>
""",
    unsafe_allow_html=True,
)


# ============================================================
# DATASET SNAPSHOT — FULLY GENERIC
# ============================================================

if not df.empty:
    metrics = dataset_metrics(df)

    st.markdown(
        '<div class="section">Dataset snapshot</div>',
        unsafe_allow_html=True,
    )

    m1, m2, m3, m4 = st.columns(4)

    with m1:
        metric_card("ROWS", f"{metrics['rows']:,}")

    with m2:
        metric_card("COLUMNS", f"{metrics['columns']:,}")

    with m3:
        metric_card("NUMERIC FIELDS", f"{metrics['numeric']:,}")

    with m4:
        metric_card("MISSING VALUES", f"{metrics['missing']:,}")


# ============================================================
# DATA PREVIEW
# ============================================================

if not df.empty:
    with st.expander("Preview dataset", expanded=False):
        st.dataframe(df.head(25), width="stretch", hide_index=True)


# ============================================================
# QUESTION AREA
# ============================================================

if "question_input" not in st.session_state:
    st.session_state.question_input = ""

if "run_analysis" not in st.session_state:
    st.session_state.run_analysis = False

st.markdown(
    '<div class="section">Ask a question</div>',
    unsafe_allow_html=True,
)

question = st.text_input(
    "Your question",
    key="question_input",
    placeholder="e.g. Which category has the highest value?",
)

st.caption(
    "Choose a suggested question or write your own. Suggestions are generated from the actual columns in your dataset."
)

if not df.empty:
    numeric_cols = dataset_metrics(df)["numeric_cols"]
    categorical_cols = [
        c for c in df.columns
        if pd.api.types.is_object_dtype(df[c])
        or pd.api.types.is_categorical_dtype(df[c])
    ]

    suggestions = []

    if numeric_cols and categorical_cols:
        suggestions.append(
            f"Which {categorical_cols[0]} has the highest {numeric_cols[0]}?"
        )

    if numeric_cols:
        suggestions.append(f"What is the average {numeric_cols[0]}?")
        suggestions.append(f"What is the total {numeric_cols[0]}?")

    if not suggestions:
        suggestions = [
            "Summarize the dataset.",
            "Show the main patterns in the data.",
            "What are the most important fields?",
        ]

    suggestions = suggestions[:3]
else:
    suggestions = [
        "Upload a dataset first.",
        "Then ask a question about its data.",
        "CSV and Excel files are supported.",
    ]


def select_suggestion(suggestion):
    st.session_state.question_input = suggestion
    st.session_state.run_analysis = True


s1, s2, s3 = st.columns(3)

for column, suggestion in zip((s1, s2, s3), suggestions):
    with column:
        st.button(
            suggestion,
            width="stretch",
            disabled=df.empty,
            on_click=select_suggestion,
            args=(suggestion,),
        )

manual_analyze = st.button(
    "✦  Analyze with AI",
    type="primary",
    width="stretch",
    disabled=df.empty,
)

if manual_analyze:
    st.session_state.run_analysis = True

active_question = st.session_state.question_input.strip()
analyze = st.session_state.run_analysis

# Consume the trigger so a later rerun does not repeat the same analysis.
if analyze:
    st.session_state.run_analysis = False


# ============================================================
# ANALYSIS RESULTS
# ============================================================

if analyze and active_question and not df.empty:
    with st.spinner("AI is planning the analysis…"):
        try:
            plan, mode = ai_plan(active_question, df)
            safe_sql = clean_sql(plan["sql"])
            result = execute(safe_sql, df)
        except Exception as exc:
            st.error(f"Analysis failed: {exc}")
            st.stop()

    st.markdown(
        '<div class="section">AI answer</div>',
        unsafe_allow_html=True,
    )

    st.markdown(
        f"""
        <div class="answer">
            <div class="answer-label">{html.escape(str(mode).upper())}</div>
            <div class="answer-text">{html.escape(str(plan.get("answer", "")))}</div>
        </div>
        """,
        unsafe_allow_html=True,
    )

    if not result.empty:
        numeric_result_cols = [
            c for c in result.columns
            if pd.api.types.is_numeric_dtype(result[c])
            and not pd.api.types.is_bool_dtype(result[c])
        ]

        top_result = str(result.iloc[0].iloc[0])
        top_value = (
            result.iloc[0][numeric_result_cols[0]]
            if numeric_result_cols
            else ""
        )

        st.markdown(
            '<div class="section">Result snapshot</div>',
            unsafe_allow_html=True,
        )

        r1, r2, r3 = st.columns(3)

        with r1:
            metric_card("FIRST RESULT", top_result)

        with r2:
            if isinstance(top_value, (int, float)):
                metric_card("FIRST NUMERIC VALUE", f"{top_value:,.2f}")
            else:
                metric_card("FIRST NUMERIC VALUE", str(top_value))

        with r3:
            metric_card("ROWS RETURNED", f"{len(result):,}")

    st.markdown(
        '<div class="section">Evidence</div>',
        unsafe_allow_html=True,
    )

    left, right = st.columns([1.45, 1])

    with left:
        st.dataframe(result, width="stretch", hide_index=True)

        x = plan.get("chart_x")
        y = plan.get("chart_y")
        chart_type = plan.get("chart_type")

        if (
            chart_type == "bar"
            and x in result.columns
            and y in result.columns
        ):
            st.bar_chart(
                result.set_index(x)[y],
                width="stretch",
            )

        elif (
            chart_type == "line"
            and x in result.columns
            and y in result.columns
        ):
            st.line_chart(
                result.set_index(x)[y],
                width="stretch",
            )

        st.download_button(
            "Download result CSV",
            data=result.to_csv(index=False).encode("utf-8"),
            file_name="insightpilot_result.csv",
            mime="text/csv",
            width="stretch",
        )

    with right:
        st.markdown("**How InsightPilot calculated this**")

        assumptions = plan.get("assumptions", [])

        if assumptions:
            for assumption in assumptions:
                st.markdown(f"• {assumption}")
        else:
            st.caption("No additional assumptions were provided.")

        with st.expander("Generated SQL"):
            st.code(safe_sql, language="sql")

        st.caption(
            "The result is computed from the uploaded dataset. "
            "The AI generates the analysis plan; it does not fabricate the result."
        )

    st.markdown(
        '<div class="section">Was this useful?</div>',
        unsafe_allow_html=True,
    )

    f1, f2, _ = st.columns([1, 1, 6])

    with f1:
        st.button("👍 Helpful")

    with f2:
        st.button("👎 Needs work")


# ============================================================
# EMPTY / LANDING STATE
# ============================================================

elif df.empty:
    st.markdown(
        '<div class="section">How it works</div>',
        unsafe_allow_html=True,
    )

    h1, h2, h3 = st.columns(3)

    with h1:
        st.markdown(
            """
            <div class="card">
                <b>01 · Upload</b><br>
                <span class="small">
                    Add a CSV or Excel dataset from your workspace.
                </span>
            </div>
            """,
            unsafe_allow_html=True,
        )

    with h2:
        st.markdown(
            """
            <div class="card">
                <b>02 · Ask</b><br>
                <span class="small">
                    Describe what you want to understand in plain English.
                </span>
            </div>
            """,
            unsafe_allow_html=True,
        )

    with h3:
        st.markdown(
            """
            <div class="card">
                <b>03 · Verify</b><br>
                <span class="small">
                    Inspect the SQL, assumptions, evidence and visualization.
                </span>
            </div>
            """,
            unsafe_allow_html=True,
        )

else:
    st.markdown(
        '<div class="section">How it works</div>',
        unsafe_allow_html=True,
    )

    h1, h2, h3 = st.columns(3)

    with h1:
        st.markdown(
            """
            <div class="card">
                <b>01 · Ask</b><br>
                <span class="small">
                    Describe your analytical question in plain English.
                </span>
            </div>
            """,
            unsafe_allow_html=True,
        )

    with h2:
        st.markdown(
            """
            <div class="card">
                <b>02 · Analyze</b><br>
                <span class="small">
                    AI maps your question to the actual dataset schema.
                </span>
            </div>
            """,
            unsafe_allow_html=True,
        )

    with h3:
        st.markdown(
            """
            <div class="card">
                <b>03 · Verify</b><br>
                <span class="small">
                    Inspect the query, result, assumptions and chart.
                </span>
            </div>
            """,
            unsafe_allow_html=True,
        )


st.markdown(
    '<div class="small" style="margin-top:30px">'
    "InsightPilot is a general-purpose analytics portfolio prototype. "
    "It works with arbitrary CSV/XLSX datasets and uses dataset-grounded analysis."
    "</div>",
    unsafe_allow_html=True,
)

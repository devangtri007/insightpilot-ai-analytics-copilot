import json
import re
from pathlib import Path

import duckdb
import pandas as pd


ROOT = Path(__file__).resolve().parents[1]

DATA_PATH = ROOT / "data" / "pharma_sales_sample.csv"
TEST_PATH = ROOT / "evals" / "test_cases.json"


# ============================================================
# SQL SAFETY
# ============================================================

BLOCKED_KEYWORDS = re.compile(
    r"\b("
    r"INSERT|UPDATE|DELETE|DROP|ALTER|CREATE|COPY|ATTACH|DETACH|"
    r"PRAGMA|EXPORT|IMPORT|MERGE|TRUNCATE|INSTALL|LOAD"
    r")\b",
    re.IGNORECASE,
)


def validate_sql(sql: str):
    """
    Validate that generated SQL is read-only and consists
    of a single SELECT/WITH statement.
    """

    if not isinstance(sql, str) or not sql.strip():
        return False, "SQL is empty"

    sql = sql.strip()

    # Remove markdown fences if the model returned them.
    sql = (
        sql.replace("```sql", "")
        .replace("```SQL", "")
        .replace("```", "")
        .strip()
    )

    # Multiple statements are not allowed.
    if ";" in sql.rstrip(";"):
        return False, "Multiple SQL statements detected"

    sql = sql.rstrip(";").strip()

    # Only SELECT / WITH queries are permitted.
    if not re.match(r"^(SELECT|WITH)\b", sql, re.IGNORECASE):
        return False, "Query does not begin with SELECT/WITH"

    # Block dangerous operations.
    if BLOCKED_KEYWORDS.search(sql):
        return False, "Blocked SQL keyword detected"

    return True, "OK"


# ============================================================
# SQL EXECUTION
# ============================================================

def execute_sql(sql: str, df: pd.DataFrame):
    """
    Execute read-only SQL against the evaluation dataframe.
    """

    con = duckdb.connect(database=":memory:")

    try:
        con.register("data", df)
        result = con.execute(sql).df()
        return result
    finally:
        con.close()


# ============================================================
# SCHEMA GROUNDING
# ============================================================

def extract_identifiers(sql: str):
    """
    Extract identifiers that appear to be column references.

    This is intentionally lightweight. The evaluator is not
    trying to build a full SQL parser.
    """

    tokens = re.findall(r'"([^"]+)"|\b[A-Za-z_][A-Za-z0-9_]*\b', sql)

    identifiers = []

    for token in tokens:
        if isinstance(token, tuple):
            token = token[0]

        if token:
            identifiers.append(token)

    return set(identifiers)


def schema_grounding_check(sql: str, df: pd.DataFrame):
    """
    Detect obvious references to columns that don't exist.
    """

    columns = {str(c).lower() for c in df.columns}

    identifiers = extract_identifiers(sql)

    sql_lower = sql.lower()

    # We only flag identifiers that look like column names.
    suspicious = []

    for identifier in identifiers:
        normalized = identifier.lower()

        if normalized in {
            "select",
            "from",
            "where",
            "group",
            "by",
            "order",
            "limit",
            "offset",
            "having",
            "as",
            "asc",
            "desc",
            "and",
            "or",
            "not",
            "null",
            "is",
            "on",
            "join",
            "left",
            "right",
            "inner",
            "outer",
            "case",
            "when",
            "then",
            "else",
            "end",
            "distinct",
            "between",
            "in",
            "like",
            "count",
            "sum",
            "avg",
            "min",
            "max",
            "round",
            "date",
            "data",
        }:
            continue

        # Ignore aliases / common SQL words.
        if normalized in columns:
            continue

        if normalized in sql_lower:
            suspicious.append(identifier)

    # This is a heuristic, so don't fail solely on this check.
    return len(suspicious) == 0, suspicious


# ============================================================
# RESULT QUALITY
# ============================================================

def result_is_valid(result: pd.DataFrame):
    """
    Basic sanity checks on query output.
    """

    if result is None:
        return False, "No result returned"

    if not isinstance(result, pd.DataFrame):
        return False, "Result is not a dataframe"

    if len(result.columns) == 0:
        return False, "No output columns"

    return True, "OK"


# ============================================================
# EXPECTATION CHECKS
# ============================================================

def expectation_checks(question, sql, result, expected, df):
    checks = []

    sql_lower = sql.lower()

    if expected.get("requires_numeric_aggregation"):
        passed = any(
            keyword in sql_lower
            for keyword in ["sum(", "avg(", "min(", "max(", "count("]
        )
        checks.append(("numeric_aggregation", passed))

    if expected.get("requires_grouping"):
        passed = "group by" in sql_lower
        checks.append(("grouping", passed))

    if expected.get("requires_limit"):
        passed = "limit" in sql_lower
        checks.append(("limit", passed))

    if expected.get("requires_average"):
        passed = "avg(" in sql_lower
        checks.append(("average", passed))

    if expected.get("requires_filter"):
        passed = "where" in sql_lower
        checks.append(("filter", passed))

    if expected.get("requires_count"):
        passed = "count(" in sql_lower
        checks.append(("count", passed))

    if expected.get("must_use_existing_columns"):
        passed, _ = schema_grounding_check(sql, df)
        checks.append(("schema_grounding", passed))

    if expected.get("read_only"):
        passed, _ = validate_sql(sql)
        checks.append(("read_only", passed))

    # Ambiguous questions should ideally have an assumption.
    if expected.get("should_state_assumption"):
        # This evaluator is designed to work with structured
        # output later. For now we mark this as requiring
        # manual review rather than inventing a score.
        checks.append(("assumption_review", None))

    return checks


# ============================================================
# TEST RUNNER
# ============================================================

def run_evaluation():
    print("=" * 70)
    print("InsightPilot Evaluation")
    print("=" * 70)

    print(f"\nDataset: {DATA_PATH}")
    print(f"Tests:   {TEST_PATH}")

    df = pd.read_csv(DATA_PATH)

    with open(TEST_PATH, "r", encoding="utf-8") as f:
        tests = json.load(f)

    print(f"\nDataset shape: {df.shape}")
    print(f"Columns: {list(df.columns)}")
    print(f"Test cases: {len(tests)}")

    print("\n" + "-" * 70)

    # --------------------------------------------------------
    # SQL GUARDRAIL TESTS
    # --------------------------------------------------------

    print("\n1. SQL GUARDRAIL TESTS\n")

    guardrail_tests = [
        (
            "Valid SELECT",
            "SELECT * FROM data LIMIT 5",
            True,
        ),
        (
            "Valid WITH",
            "WITH x AS (SELECT * FROM data) SELECT * FROM x",
            True,
        ),
        (
            "INSERT blocked",
            "INSERT INTO data VALUES (1)",
            False,
        ),
        (
            "UPDATE blocked",
            "UPDATE data SET x = 1",
            False,
        ),
        (
            "DELETE blocked",
            "DELETE FROM data",
            False,
        ),
        (
            "DROP blocked",
            "DROP TABLE data",
            False,
        ),
        (
            "Multiple statements blocked",
            "SELECT * FROM data; DROP TABLE data",
            False,
        ),
    ]

    guardrail_passed = 0

    for name, sql, expected in guardrail_tests:
        passed, reason = validate_sql(sql)

        status = "PASS" if passed == expected else "FAIL"

        if status == "PASS":
            guardrail_passed += 1

        print(f"[{status}] {name}")

        if status == "FAIL":
            print(f"       Expected: {expected}")
            print(f"       Actual:   {passed}")
            print(f"       Reason:   {reason}")

    print(
        f"\nGuardrail score: "
        f"{guardrail_passed}/{len(guardrail_tests)}"
    )

    # --------------------------------------------------------
    # EXECUTION TEST
    # --------------------------------------------------------

    print("\n2. DUCKDB EXECUTION TEST\n")

    execution_sql = "SELECT * FROM data LIMIT 5"

    try:
        execution_result = execute_sql(execution_sql, df)

        valid, reason = result_is_valid(execution_result)

        if valid:
            print("[PASS] DuckDB execution")
            print(
                f"       Returned {len(execution_result)} rows "
                f"and {len(execution_result.columns)} columns"
            )
        else:
            print("[FAIL] DuckDB execution")
            print(f"       {reason}")

    except Exception as exc:
        print("[FAIL] DuckDB execution")
        print(f"       {exc}")

    # --------------------------------------------------------
    # DATASET TEST CASES
    # --------------------------------------------------------

    print("\n3. ANALYTICAL TEST CASES\n")

    print(
        "These cases currently validate the evaluation framework "
        "against generated SQL when SQL is supplied."
    )

    print(
        "The next iteration will connect these cases directly "
        "to the live LLM planner."
    )

    results = []

    for test in tests:
        test_id = test["id"]
        question = test["question"]

        # Placeholder deterministic query.
        #
        # This intentionally does NOT pretend to evaluate the
        # LLM. It verifies that the evaluator infrastructure
        # itself works before connecting the live model.
        sql = "SELECT * FROM data LIMIT 5"

        try:
            safe, safety_reason = validate_sql(sql)

            if not safe:
                results.append(
                    {
                        "id": test_id,
                        "question": question,
                        "status": "FAIL",
                        "reason": safety_reason,
                    }
                )
                continue

            result = execute_sql(sql, df)

            valid, reason = result_is_valid(result)

            if not valid:
                results.append(
                    {
                        "id": test_id,
                        "question": question,
                        "status": "FAIL",
                        "reason": reason,
                    }
                )
                continue

            checks = expectation_checks(
                question,
                sql,
                result,
                test.get("expected", {}),
                df,
            )

            hard_checks = [
                passed
                for _, passed in checks
                if passed is not None
            ]

            passed = all(hard_checks) if hard_checks else True

            results.append(
                {
                    "id": test_id,
                    "question": question,
                    "status": "PASS" if passed else "REVIEW",
                    "checks": checks,
                }
            )

        except Exception as exc:
            results.append(
                {
                    "id": test_id,
                    "question": question,
                    "status": "FAIL",
                    "reason": str(exc),
                }
            )

    for item in results:
        print(
            f"[{item['status']}] "
            f"{item['id']} — {item['question']}"
        )

    # --------------------------------------------------------
    # SUMMARY
    # --------------------------------------------------------

    passed = sum(
        1 for item in results if item["status"] == "PASS"
    )

    failed = sum(
        1 for item in results if item["status"] == "FAIL"
    )

    review = sum(
        1 for item in results if item["status"] == "REVIEW"
    )

    print("\n" + "=" * 70)
    print("SUMMARY")
    print("=" * 70)

    print(f"PASS:   {passed}")
    print(f"FAIL:   {failed}")
    print(f"REVIEW: {review}")

    print("\nEvaluation framework successfully executed.")


if __name__ == "__main__":
    run_evaluation()

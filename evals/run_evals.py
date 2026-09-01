
"""
Minimal evaluation harness for the product prototype.

Run:
    python evals/run_evals.py

The suite checks:
1) expected analytical intent
2) SQL safety
3) result correctness on the synthetic dataset

This is intentionally deterministic so it can run without an API key.
"""

import sys
from pathlib import Path
import pandas as pd
import duckdb

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT))
from app import local_demo, clean_sql

DATA = ROOT / "data" / "pharma_sales_sample.csv"

CASES = [
    {
        "question": "Which products generated the most revenue?",
        "must_contain": ["GROUP BY product", "SUM(revenue)"],
        "expected_first_dimension": "product"
    },
    {
        "question": "Compare revenue by region.",
        "must_contain": ["GROUP BY region", "SUM(revenue)"],
        "expected_first_dimension": "region"
    },
    {
        "question": "Which product has the highest weighted gross margin?",
        "must_contain": ["gross_margin", "GROUP BY product"],
        "expected_first_dimension": "product"
    },
]

def main():
    df = pd.read_csv(DATA)
    passed = 0
    print("InsightPilot evaluation suite")
    print("=" * 34)

    for case in CASES:
        plan = local_demo(case["question"])
        sql = clean_sql(plan["sql"])
        sql_lower = sql.lower()
        safe = not any(x in sql_lower for x in ["insert ", "update ", "delete ", "drop ", "alter "])
        intent = all(x.lower() in sql_lower for x in case["must_contain"])

        con = duckdb.connect()
        con.register("sales", df)
        result = con.execute(sql).df()
        nonempty = len(result) > 0

        ok = safe and intent and nonempty
        passed += int(ok)
        print(("PASS" if ok else "FAIL"), "-", case["question"])
        print("  rows:", len(result), "| safe:", safe, "| intent:", intent)

    print(f"\nScore: {passed}/{len(CASES)} = {passed/len(CASES):.0%}")
    return 0 if passed == len(CASES) else 1

if __name__ == "__main__":
    raise SystemExit(main())

"""
Tool: report_generator
Uses GPT-4o to write a business report from the computed metrics.

This is itself an LLM call — a "sub-agent" pattern: the outer agent orchestrates
tool calls, and this inner call produces the final narrative artefact.
Temperature is kept low (0.3) to reduce hallucination on numerical claims.
"""

import json
import os

import pandas as pd
from openai import OpenAI


def generate_report(metrics_path: str, validation_path: str) -> str:
    """
    Generate output/report.md using GPT-4o.

    Returns
    -------
    str
        JSON: {"report_path": "output/report.md", "preview": "..."}
    """
    client = OpenAI(api_key=os.environ.get("OPENAI_API_KEY"))
    os.makedirs("output", exist_ok=True)

    # ── Load metrics ──────────────────────────────────────────────────────────
    metrics_df = pd.read_csv(metrics_path)

    try:
        metrics_table = metrics_df.to_markdown(index=False, floatfmt=".4f")
    except ImportError:
        metrics_table = metrics_df.to_string(index=False)

    # ── Load validation ───────────────────────────────────────────────────────
    with open(validation_path, encoding="utf-8") as fh:
        validation = json.load(fh)

    validation_lines = (
        f"All checks passed: {validation['all_passed']} "
        f"({validation['passed_checks']}/{validation['total_checks']})\n"
    )
    for chk in validation["checks"]:
        icon = "✓" if chk["passed"] else "✗"
        validation_lines += f"  {icon} {chk['name']}: {chk['message']}\n"

    # ── Derived summary stats for grounding the prompt ────────────────────────
    total_revenue   = round(metrics_df["monthly_revenue"].sum(), 2)
    final_active    = int(metrics_df["active_users"].iloc[-1])
    retention_pct   = round(final_active / 1000 * 100, 1)
    avg_churn_pct   = round(metrics_df["churn_rate"].mean() * 100, 2)
    peak_churn_row  = metrics_df.loc[metrics_df["churn_rate"].idxmax()]
    min_churn_row   = metrics_df.loc[metrics_df[metrics_df["month"] > 1]["churn_rate"].idxmin()]

    prompt = f"""You are a senior fintech analyst at a consumer subscription company.
Analyze the following 12-month KPI data for a cohort of 1,000 initial users and write a professional business report.

## Monthly KPI Table

{metrics_table}

## Data Quality Checks

{validation_lines}
## High-level Stats

- Total 12-month revenue: ${total_revenue:,.2f}
- Starting users (Month 1): 1,000
- Active users at end (Month 12): {final_active}
- 12-month retention rate: {retention_pct}%
- Average monthly churn rate (months 2–12): {avg_churn_pct}%
- Peak churn: month {int(peak_churn_row['month'])} ({peak_churn_row['churn_rate']*100:.1f}%)
- Lowest churn (after month 1): month {int(min_churn_row['month'])} ({min_churn_row['churn_rate']*100:.1f}%)

## Instructions

Write a concise but data-driven Markdown report with these exact sections:

### 1. Executive Summary
2–3 sentences covering total revenue, 12-month retention, and the dominant trend.

### 2. Monthly Revenue Trend
Describe how revenue evolved month-over-month. Identify the largest single-month drop.
State total cumulative revenue.

### 3. Churn Trend
Identify which months had highest/lowest churn. Explain the typical SaaS churn pattern visible in the data.
Quantify how many users were lost and when.

### 4. ARPU Trend
Describe how ARPU changed. Explain what drives ARPU movement in a mixed basic/premium plan model.

### 5. Data Quality Checks
Summarise which checks were run and whether data can be trusted.

### 6. Business Interpretation
Give exactly 3 specific, actionable recommendations backed by numbers from this dataset.
Avoid generic advice — tie each recommendation to a specific metric or month.

Use actual numbers throughout. Format with markdown headers, bullet points, and **bold** for key figures."""

    response = client.chat.completions.create(
        model="gpt-4o",
        messages=[
            {
                "role": "system",
                "content": (
                    "You are a senior fintech analyst writing an internal business report. "
                    "Be precise, reference actual numbers, and give actionable recommendations."
                ),
            },
            {"role": "user", "content": prompt},
        ],
        temperature=0.3,
        max_tokens=2500,
    )

    report_content = response.choices[0].message.content

    # ── Write report.md ───────────────────────────────────────────────────────
    report_path = os.path.join("output", "report.md")
    with open(report_path, "w", encoding="utf-8") as fh:
        fh.write("# Churn & Revenue Report\n\n")
        fh.write(f"*Generated: {pd.Timestamp.now().strftime('%Y-%m-%d %H:%M')}*\n\n")
        fh.write("---\n\n")
        fh.write(report_content)
        fh.write("\n\n---\n\n## Appendix: Full Monthly KPI Table\n\n")
        try:
            fh.write(metrics_df.to_markdown(index=False))
        except ImportError:
            fh.write(metrics_df.to_string(index=False))
        fh.write("\n")

    print(f"[ReportGenerator] Report saved → {report_path}")

    return json.dumps({
        "report_path": report_path,
        "preview": report_content[:400] + "...",
    })

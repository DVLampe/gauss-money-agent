"""
Tool: report_reviewer
Critically reviews the generated Markdown report using GPT-4o.

Checks three dimensions:
1. Clarity       — is the language accessible to a non-technical business executive?
2. Data accuracy — do numbers and trends in the report match the metrics CSV?
3. Recommendations — are they specific, actionable, and tied to actual figures?

Returns {"approved": bool, "issues": [...], "feedback": "..."}
"""

import json
import os

import pandas as pd
from openai import OpenAI


def review_report(report_path: str, metrics_path: str) -> str:
    """
    Review the generated report for quality and data consistency.

    Returns
    -------
    str
        JSON: {"approved": bool, "issues": list[str], "feedback": str}
    """
    client = OpenAI(api_key=os.environ.get("OPENAI_API_KEY"))

    with open(report_path, encoding="utf-8") as fh:
        report_text = fh.read()

    metrics_df = pd.read_csv(metrics_path)
    try:
        metrics_table = metrics_df.to_markdown(index=False, floatfmt=".4f")
    except ImportError:
        metrics_table = metrics_df.to_string(index=False)

    prompt = f"""You are a strict senior editor reviewing a business report on subscription churn and revenue.

Evaluate the report across three dimensions:
1. **Clarity** — Is the language clear and accessible for a non-technical business executive? No jargon without explanation.
2. **Data accuracy** — Do all numbers, percentages, and trends in the report match the source metrics table exactly?
3. **Recommendations** — Are the 3 recommendations specific, quantified, and tied to actual figures from the data?

SOURCE METRICS (ground truth):
{metrics_table}

REPORT TO REVIEW:
{report_text}

Respond ONLY with a JSON object (no markdown fences):
{{
  "approved": true or false,
  "issues": ["short description of issue 1", "..."],
  "feedback": "Concise actionable instructions for the report author addressing each issue. Empty string if approved."
}}

Rules:
- Set approved=true only if all three dimensions are satisfactory.
- List each distinct problem separately in "issues" (empty list if approved).
- "feedback" must give specific fix instructions, referencing exact numbers when relevant.
- Do NOT reject for minor stylistic preferences — only genuine clarity, accuracy, or specificity problems."""

    response = client.chat.completions.create(
        model="gpt-4o",
        messages=[{"role": "user", "content": prompt}],
        response_format={"type": "json_object"},
        temperature=0.2,
        max_tokens=512,
    )

    result = json.loads(response.choices[0].message.content)
    print(
        f"[ReviewReport] approved={result.get('approved')} | "
        f"issues={len(result.get('issues', []))}"
    )
    return json.dumps(result)

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

    prompt = f"""Ты — строгий старший редактор, проверяющий бизнес-отчёт по оттоку и выручке подписочного сервиса.

Оцени отчёт по трём критериям:
1. **Ясность** — понятен ли язык нетехническому бизнес-руководителю? Нет ли необъяснённого жаргона?
2. **Точность данных** — совпадают ли числа и тренды в отчёте с исходной таблицей метрик?
   ДОПУСК: разница до ±2% и ±$50 для производных показателей (суммарная выручка, retention) является приемлемым округлением. Отклоняй только существенные расхождения (>2% или >$50).
3. **Рекомендации** — все ли 3 рекомендации конкретны, измеримы и привязаны к реальным цифрам?

ИСХОДНЫЕ МЕТРИКИ (источник истины):
{metrics_table}

ОТЧЁТ ДЛЯ ПРОВЕРКИ:
{report_text}

Отвечай ТОЛЬКО JSON-объектом (без markdown-оформления):
{{
  "approved": true или false,
  "issues": ["краткое описание проблемы 1", "..."],
  "feedback": "Конкретные инструкции для автора отчёта по устранению каждой проблемы. Пустая строка если approved=true."
}}

Правила:
- approved=true только если все три критерия удовлетворены.
- Перечисляй каждую отдельную проблему в "issues" (пустой список если approved=true).
- "feedback" должен давать конкретные инструкции по исправлению с указанием точных цифр при необходимости.
- НЕ отклоняй за незначительные стилистические предпочтения — только реальные проблемы ясности, точности или конкретности.
- Все поля "issues" и "feedback" пиши на РУССКОМ языке."""

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

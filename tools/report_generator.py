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


def generate_report(
    metrics_path: str,
    validation_path: str,
    feedback: str | None = None,
) -> str:
    """
    Generate output/report.md using GPT-4o.

    Parameters
    ----------
    feedback : str | None
        If provided (on a retry), the reviewer's critique to address in this version.

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
        f"All hard checks passed: {validation['all_hard_passed']} "
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

    prompt = f"""\
Ты — старший финтех-аналитик в потребительской подписочной компании.
Проанализируй данные KPI за 12 месяцев для когорты из 1 000 пользователей
и напиши профессиональный бизнес-отчёт.

ВАЖНО: Отчёт должен быть написан ПОЛНОСТЬЮ на РУССКОМ языке.
Используй ТОЧНО эти предрассчитанные числа — не пересчитывай самостоятельно:
- Суммарная выручка за 12 месяцев: ${total_revenue:,.2f}
- Активных пользователей на конец (месяц 12): {final_active}
- Retention rate (месяц 12): {retention_pct}%
- Средний churn rate (месяцы 2–12): {avg_churn_pct}%
- Пиковый churn: месяц {int(peak_churn_row['month'])} ({peak_churn_row['churn_rate']*100:.1f}%)
- Минимальный churn (после месяца 1): месяц {int(min_churn_row['month'])} ({min_churn_row['churn_rate']*100:.1f}%)

## Таблица KPI по месяцам

{metrics_table}

## Результаты проверок качества данных

{validation_lines}

## Инструкции

Напиши лаконичный, но насыщенный данными Markdown-отчёт со следующими разделами:

### 1. Резюме
2–3 предложения: суммарная выручка, retention за 12 месяцев, главный тренд.

### 2. Динамика выручки
Опиши изменение выручки по месяцам. Укажи наибольший месячный спад. Назови суммарную выручку.

### 3. Динамика оттока (Churn)
Какие месяцы показали максимальный и минимальный churn. Объясни типичный SaaS-паттерн оттока.
Сколько пользователей потеряно и когда.

### 4. Динамика ARPU
Опиши изменение ARPU. Объясни, что влияет на ARPU при смешанных тарифах basic/premium.

### 5. Качество данных
Кратко: какие проверки проведены, можно ли доверять данным.

### 6. Бизнес-рекомендации
Ровно 3 конкретные, измеримые рекомендации, основанные на числах из данных.
Избегай общих фраз — каждая рекомендация должна ссылаться на конкретную метрику или месяц.

Используй реальные числа везде. Форматирование: markdown-заголовки, маркированные списки, **жирный** для ключевых цифр."""

    if feedback:
        prompt += (
            f"\n\n## ВАЖНО: Замечания ревьюера по предыдущей версии\n{feedback}\n"
            "Исправь ВСЕ указанные замечания в этой версии."
        )

    response = client.chat.completions.create(
        model="gpt-4o",
        messages=[
            {
                "role": "system",
                "content": (
                    "Ты старший финтех-аналитик, пишущий внутренний бизнес-отчёт на русском языке. "
                    "Используй точные числа из предоставленных данных, давай конкретные рекомендации."
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
        fh.write("# Отчёт по оттоку и выручке (Churn & Revenue)\n\n")
        fh.write(f"*Сформирован: {pd.Timestamp.now().strftime('%Y-%m-%d %H:%M')}*\n\n")
        fh.write("---\n\n")
        fh.write(report_content)
        fh.write("\n\n---\n\n## Приложение: Полная таблица KPI по месяцам\n\n")
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

"""
agent.py — Gauss Money Churn & Revenue AI Agent
================================================
рхитектура разделена на два слоя:

  [Слой данных]  generate_data() запускается ДО агента как отдельный шаг
                 подготовки. В продакшне здесь был бы реальный CSV от системы.
                 Результат — data/users.csv — передаётся агенту как входной файл.

  [Агент]        Оркестрирует четыре аналитических шага:
                 calculate_metrics → validate_data →
                 generate_report ↔ review_report (цикл до 3 попыток)

Pipeline
--------
  Pre-step     generate_data()   — синтетические данные (вне агента)
  Step 1 / 4   calculate_metrics — вычисляет KPI по CSV
  Step 2 / 4   validate_data     — 8 проверок качества данных
  Step 3 / 4   generate_report   — GPT-4o пишет Markdown-отчёт
  Step 4 / 4   review_report     — GPT-4o проверяет отчёт:
                                    ясность → точность данных → рекомендации
                                    если проблемы → возврат на Step 3 (макс. 3 попытки)
  Final        Агент пишет итоговое резюме
"""

import json
import os
import sys

import pandas as pd
from dotenv import load_dotenv
from openai import OpenAI

from tools.data_generator import generate_data
from tools.metrics_calculator import calculate_metrics
from tools.data_validator import validate_data
from tools.report_generator import generate_report
from tools.report_reviewer import review_report

load_dotenv()

MAX_REVIEW_ATTEMPTS = 3


def run_agent(data_path: str | None = None) -> None:
    api_key = os.environ.get("OPENAI_API_KEY")
    if not api_key:
        print(
            "[ERROR] OPENAI_API_KEY is not set.\n"
            "Copy .env.example to .env and add your key, then retry."
        )
        sys.exit(1)

    print("=" * 62)
    print("  Gauss Money — Churn & Revenue AI Agent")
    print("=" * 62)

    # ── Pre-step: подготовка данных (вне агента) ───────────────────────────────
    # В продакшне здесь был бы путь к реальному CSV от бизнес-системы.
    if data_path is None:
        print("\n[Pre-step] Генерация синтетических данных...")
        r0 = json.loads(generate_data())
        data_path = r0["data_path"]
        s = r0["summary"]
        print(
            f"[Pre-step] Готово: {s['total_records']:,} записей "
            f"({s['basic_users']} basic + {s['premium_users']} premium) → {data_path}"
        )
    else:
        print(f"\n[Pre-step] Используется готовый файл данных: {data_path}")

    # ── Step 1/4: Расчёт метрик ────────────────────────────────────────────────
    print("\n[Step 1/4] ▶ calculate_metrics")
    metrics = json.loads(calculate_metrics(data_path=data_path))
    metrics_path = metrics["metrics_path"]
    print(f"           → {metrics_path}")

    # ── Step 2/4: Валидация данных ─────────────────────────────────────────────
    print("\n[Step 2/4] ▶ validate_data")
    validation = json.loads(validate_data(data_path=data_path, metrics_path=metrics_path))
    validation_path = validation["validation_path"]
    val_summary = validation["summary"]

    if val_summary["hard_failures"] > 0:
        print(f"\n[STOP] {val_summary['hard_failures']} critical failure(s). Report not generated.")
        for chk in val_summary["checks"]:
            if chk["severity"] == "hard" and not chk["passed"]:
                print(f"  ✗ {chk['name']}: {chk['message']}")
        return

    if val_summary["soft_warnings"] > 0:
        print(f"[WARN] {val_summary['soft_warnings']} soft warning(s) — proceeding with report.")

    # ── Steps 3/4 + 4/4: Генерация → ревью (цикл, макс. MAX_REVIEW_ATTEMPTS) ──
    client = OpenAI(api_key=api_key)
    feedback = None

    for attempt in range(1, MAX_REVIEW_ATTEMPTS + 1):
        label = f"{attempt}/{MAX_REVIEW_ATTEMPTS}"

        # Step 3/4: generate_report
        print(f"\n[Step 3/4] ▶ generate_report (attempt {label})")
        if feedback:
            print(f"           feedback: {feedback[:120]}...")
        report = json.loads(
            generate_report(
                metrics_path=metrics_path,
                validation_path=validation_path,
                feedback=feedback,
            )
        )
        report_path = report["report_path"]

        # Step 4/4: review_report
        print(f"\n[Step 4/4] ▶ review_report (attempt {label})")
        review = json.loads(review_report(report_path=report_path, metrics_path=metrics_path))

        if review["approved"]:
            print("           ✓ Report approved")
            break

        feedback = review["feedback"]
        issues = review.get("issues", [])
        print(f"           Issues found ({len(issues)}):")
        for issue in issues:
            print(f"             • {issue}")
    else:
        print(f"\n[WARN] Max review attempts ({MAX_REVIEW_ATTEMPTS}) reached. Using last version.")

    # ── Final: итоговое резюме (LLM) ──────────────────────────────────────────
    print("\n[Final] ▶ agent summary")
    m_df = pd.read_csv(metrics_path)
    total_revenue = round(float(m_df["monthly_revenue"].sum()), 2)
    final_active = int(m_df["active_users"].iloc[-1])
    retention_pct = round(final_active / 1000 * 100, 1)

    summary_resp = client.chat.completions.create(
        model="gpt-4o",
        messages=[
            {
                "role": "system",
                "content": "Ты старший финтех-аналитик. Пиши лаконичные итоговые резюме на русском языке.",
            },
            {
                "role": "user",
                "content": (
                    "Напиши итоговое резюме в 2–3 предложениях:\n"
                    f"- Суммарная выручка за 12 месяцев: ${total_revenue:,.2f}\n"
                    f"- Retention (месяц 12): {retention_pct}% ({final_active}/1000 пользователей)\n"
                    f"- Мягких предупреждений: {val_summary['soft_warnings']}\n"
                    f"- Полный отчёт сохранён: {report_path}"
                ),
            },
        ],
        temperature=0.2,
        max_tokens=200,
    )

    print("\n" + "=" * 62)
    print("  ИТОГ АГЕНТА")
    print("=" * 62)
    print(summary_resp.choices[0].message.content)
    print("\nГотово. Полный отчёт → output/report.md")


if __name__ == "__main__":
    run_agent()

"""
agent.py — Gauss Money Churn & Revenue AI Agent
================================================
Архитектура разделена на два слоя:

  [Слой данных]  generate_data() запускается ДО агента как отдельный шаг
                 подготовки. В продакшне здесь был бы реальный CSV от системы.
                 Результат — data/users.csv — передаётся агенту как входной файл.

  [Агент]        Получает путь к CSV и оркестрирует три аналитических инструмента:
                 calculate_metrics → validate_data → generate_report.

Agentic loop
------------
1. LLM receives system prompt + data_path in user message.
2. LLM emits tool_call(s).
3. We execute the tool, append the result as a "tool" message.
4. Repeat until LLM returns a plain text response (no tool_calls).

The LLM decides argument values and call order; it is also the guardrail —
the system prompt instructs it NOT to call generate_report if validation fails.
"""

import json
import os
import sys

from dotenv import load_dotenv
from openai import OpenAI

from tools.data_generator import generate_data
from tools.metrics_calculator import calculate_metrics
from tools.data_validator import validate_data
from tools.report_generator import generate_report

load_dotenv()

# ──────────────────────────── Tool schemas (3 аналитических инструмента) ───────
# generate_data — НЕ инструмент агента. Это шаг подготовки данных,
# который выполняется до старта агента (см. run_agent → pre-step).

TOOLS = [
    {
        "type": "function",
        "function": {
            "name": "calculate_metrics",
            "description": (
                "Compute monthly KPIs (active_users, paid_users, churned_users, "
                "monthly_revenue, churn_rate, arpu) from the subscription CSV."
            ),
            "parameters": {
                "type": "object",
                "properties": {
                    "data_path": {
                        "type": "string",
                        "description": "Path to the subscription CSV file",
                    },
                },
                "required": ["data_path"],
            },
        },
    },
    {
        "type": "function",
        "function": {
            "name": "validate_data",
            "description": (
                "Run 8 deterministic data-quality checks on the raw data and metrics. "
                "Call after calculate_metrics. "
                "If any check fails, report the issues instead of generating the report."
            ),
            "parameters": {
                "type": "object",
                "properties": {
                    "data_path":    {"type": "string", "description": "Path to users CSV"},
                    "metrics_path": {"type": "string", "description": "Path to metrics CSV returned by calculate_metrics"},
                },
                "required": ["data_path", "metrics_path"],
            },
        },
    },
    {
        "type": "function",
        "function": {
            "name": "generate_report",
            "description": (
                "Use GPT-4o to generate a Markdown business report with trends and "
                "recommendations. Call ONLY after validate_data confirms all checks passed."
            ),
            "parameters": {
                "type": "object",
                "properties": {
                    "metrics_path":    {"type": "string", "description": "Path to metrics CSV"},
                    "validation_path": {"type": "string", "description": "Path to validation JSON returned by validate_data"},
                },
                "required": ["metrics_path", "validation_path"],
            },
        },
    },
]

TOOL_MAP = {
    "calculate_metrics": calculate_metrics,
    "validate_data":     validate_data,
    "generate_report":   generate_report,
}

SYSTEM_PROMPT = """\
You are a senior fintech data analyst AI agent for a consumer subscription company.
You are given a ready subscription CSV file. Your goal: produce a complete churn & revenue report.

Execute the tools in this exact order:
1. calculate_metrics  — compute monthly KPIs; use the data_path provided in the user message
2. validate_data      — run quality checks; pass data_path and metrics_path from step 1
3. generate_report    — check the validation summary:
     • if "hard_failures" > 0 → STOP, report the hard failures, do NOT call generate_report
     • if "soft_warnings" > 0 → call generate_report anyway, mention the warnings in your summary
     • if all clear           → call generate_report normally

After all steps complete, write a 2–3 sentence summary:
  • total 12-month revenue
  • final retention rate (month-12 active users / 1 000)
  • any soft warnings found
  • location of the saved report
"""


# ──────────────────────────── Agent loop ───────────────────────────────────────

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
    # Поскольку реальных данных нет, генерируем синтетические.
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

    # ── Запуск агента ──────────────────────────────────────────────────────────
    client = OpenAI(api_key=api_key)

    messages = [
        {"role": "system", "content": SYSTEM_PROMPT},
        {"role": "user",   "content": f"Subscription data is at '{data_path}'. Calculate metrics, validate, and generate the churn & revenue report."},
    ]

    step = 0

    while True:
        response = client.chat.completions.create(
            model="gpt-4o",
            messages=messages,
            tools=TOOLS,
            tool_choice="auto",
        )

        msg = response.choices[0].message
        # Append as dict so the list stays JSON-serialisable for further calls
        messages.append(msg)

        # ── No tool calls → agent is done ─────────────────────────────────────
        if not msg.tool_calls:
            print("\n" + "=" * 62)
            print("  AGENT SUMMARY")
            print("=" * 62)
            print(msg.content)
            break

        # ── Execute each requested tool ────────────────────────────────────────
        for tc in msg.tool_calls:
            step += 1
            name = tc.function.name
            args = json.loads(tc.function.arguments)

            print(f"\n[Step {step}] ▶ {name}")
            if args:
                print(f"           args: {args}")

            try:
                result = TOOL_MAP[name](**args)
            except Exception as exc:
                result = json.dumps({"error": str(exc)})
                print(f"[ERROR] {name} raised: {exc}")

            messages.append({
                "role":         "tool",
                "tool_call_id": tc.id,
                "content":      result,
            })

    print("\nDone.  Full report → output/report.md")


if __name__ == "__main__":
    run_agent()

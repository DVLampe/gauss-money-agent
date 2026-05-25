"""
app.py — Streamlit-интерфейс для Gauss Money AI Agent

Запуск локально:
    streamlit run app.py

Или задеплоить на Streamlit Community Cloud (бесплатно):
    https://streamlit.io/cloud
    → подключить GitHub-репо, указать app.py как точку входа,
      добавить OPENAI_API_KEY в Secrets (Settings → Secrets).
"""

import json
import os

import pandas as pd
import streamlit as st

# ── Конфигурация страницы ──────────────────────────────────────────────────────
st.set_page_config(
    page_title="Gauss Money — Churn & Revenue Agent",
    page_icon="📊",
    layout="wide",
)

# ── Gauss Money brand styling ──────────────────────────────────────────────────
st.markdown(
    """
    <style>
    /* ── Brand colours ───────────────────────────────────────────────────── */
    :root {
        --gauss-navy:   #0A1628;
        --gauss-blue:   #2563EB;
        --gauss-blue2:  #1D4ED8;
        --gauss-light:  #EFF6FF;
        --gauss-text:   #0F172A;
    }

    /* ── Header banner ───────────────────────────────────────────────────── */
    .gauss-header {
        background: linear-gradient(120deg, #0A1628 0%, #1E3A6E 55%, #2563EB 100%);
        border-radius: 12px;
        padding: 28px 36px 24px;
        margin-bottom: 8px;
    }
    .gauss-header h1 {
        color: #FFFFFF !important;
        font-size: 1.9rem !important;
        font-weight: 700 !important;
        margin: 0 0 6px !important;
        letter-spacing: -0.3px;
    }
    .gauss-header p {
        color: #BFD4F7 !important;
        font-size: 0.95rem !important;
        margin: 0 !important;
    }

    /* ── Primary button ──────────────────────────────────────────────────── */
    div.stButton > button[kind="primary"] {
        background: linear-gradient(90deg, #1D4ED8, #2563EB) !important;
        color: #FFFFFF !important;
        border: none !important;
        border-radius: 8px !important;
        font-weight: 600 !important;
        font-size: 1rem !important;
        padding: 0.6rem 1.4rem !important;
        box-shadow: 0 2px 8px rgba(37,99,235,0.35) !important;
        transition: opacity .15s ease !important;
    }
    div.stButton > button[kind="primary"]:hover {
        opacity: 0.88 !important;
    }

    /* ── Metric cards (KPI table container) ─────────────────────────────── */
    div[data-testid="stDataFrame"] {
        border: 1px solid #BFDBFE !important;
        border-radius: 10px !important;
        overflow: hidden;
    }

    /* ── Section headers ─────────────────────────────────────────────────── */
    h2, h3 {
        color: var(--gauss-navy) !important;
    }

    /* ── st.info icon box ────────────────────────────────────────────────── */
    div[data-testid="stNotificationContentInfo"] {
        background-color: #EFF6FF !important;
        border-left: 4px solid #2563EB !important;
    }

    /* ── Divider colour ──────────────────────────────────────────────────── */
    hr {
        border-color: #BFDBFE !important;
    }
    </style>
    """,
    unsafe_allow_html=True,
)

# ── Header ─────────────────────────────────────────────────────────────────────
st.markdown(
    """
    <div class="gauss-header">
        <h1>📊 Gauss Money — Churn &amp; Revenue AI Agent</h1>
        <p>Генерирует синтетические данные по подписочной модели, считает KPI,
        проверяет качество данных и формирует бизнес-отчёт через GPT-4o.</p>
    </div>
    """,
    unsafe_allow_html=True,
)

st.divider()

# ── API-ключ ───────────────────────────────────────────────────────────────────
env_key = os.environ.get("OPENAI_API_KEY", "")
if env_key:
    st.success("✓ OPENAI_API_KEY получен из переменных окружения.", icon="🔑")
    api_key = env_key
else:
    api_key = st.text_input(
        "OpenAI API Key",
        type="password",
        placeholder="sk-...",
        help="Ключ используется только в рамках этой сессии и нигде не сохраняется.",
    )

st.divider()

# ── Кнопка запуска ─────────────────────────────────────────────────────────────
run_disabled = not bool(api_key)
if run_disabled:
    st.info("Введите OpenAI API Key, чтобы запустить агента.", icon="ℹ️")

if st.button("▶ Запустить агента", disabled=run_disabled, type="primary", use_container_width=True):
    os.environ["OPENAI_API_KEY"] = api_key

    from tools.data_generator import generate_data
    from tools.metrics_calculator import calculate_metrics
    from tools.data_validator import validate_data
    from tools.report_generator import generate_report
    from tools.report_reviewer import review_report

    data_path = metrics_path = validation_path = report_path = None
    metrics_data = None

    # ── Pre-step: подготовка данных (вне агента) ───────────────────────────────
    # В продакшне здесь был бы путь к реальному CSV из бизнес-системы.
    # Поскольку реальных данных нет — генерируем синтетические.
    with st.status("**[Pre-step]** — Подготовка данных...", expanded=True) as prestep:
        try:
            r0 = json.loads(generate_data())
            data_path = r0["data_path"]
            s = r0["summary"]
            prestep.update(
                label=f"✅ Pre-step — CSV готов: "
                      f"{s['total_records']:,} записей | "
                      f"{s['basic_users']} basic + {s['premium_users']} premium → {data_path}",
                state="complete",
            )
        except Exception as e:
            prestep.update(label=f"❌ Ошибка генерации данных: {e}", state="error")
            st.stop()

    st.info("Данные готовы. Передаём агенту путь к CSV → агент запускает аналитику.", icon="🤖")

    # ── Шаг 1 (агент): расчёт метрик ─────────────────────────────────────────
    with st.status("**Шаг 1 / 4** — Расчёт KPI...", expanded=True) as step1:
        try:
            r1 = json.loads(calculate_metrics(data_path))
            metrics_path = r1["metrics_path"]
            metrics_data = r1["metrics"]
            step1.update(
                label=f"✅ Шаг 1 / 4 — KPI рассчитаны → {metrics_path}",
                state="complete",
            )
        except Exception as e:
            step1.update(label=f"❌ Ошибка расчёта метрик: {e}", state="error")
            st.stop()

    # ── Шаг 2 (агент): валидация ──────────────────────────────────────────────
    with st.status("**Шаг 2 / 4** — Проверка качества данных...", expanded=True) as step2:
        try:
            r2 = json.loads(validate_data(data_path, metrics_path))
            validation_path = r2["validation_path"]
            summary = r2["summary"]
            hard_fail = summary["hard_failures"]
            soft_warn = summary["soft_warnings"]
            passed    = summary["passed_checks"]
            total     = summary["total_checks"]

            if hard_fail == 0:
                label = f"✅ Шаг 2 / 4 — {passed}/{total} проверок пройдено"
                if soft_warn:
                    label += f" ({soft_warn} предупреждения)"
                step2.update(label=label, state="complete")
            else:
                step2.update(
                    label=f"❌ Шаг 2 / 4 — {hard_fail} критических ошибок, {soft_warn} предупреждений",
                    state="error",
                )
        except Exception as e:
            step2.update(label=f"❌ Ошибка валидации: {e}", state="error")
            st.stop()

    # Показываем таблицу со всеми проверками
    st.markdown("**Результаты проверок:**")
    for chk in summary["checks"]:
        if chk["passed"]:
            st.success(f"✓ [{chk['severity'].upper()}] {chk['name']}: {chk['message']}", icon=None)
        elif chk["severity"] == "soft":
            st.warning(f"⚠ [SOFT] {chk['name']}: {chk['message']}")
            if chk.get("details"):
                with st.expander(f"Детали ({len(chk['details'])} примеров)"):
                    for d in chk["details"]:
                        st.code(d)
        else:
            st.error(f"✗ [HARD] {chk['name']}: {chk['message']}")
            if chk.get("details"):
                with st.expander(f"Детали ({len(chk['details'])} примеров)"):
                    for d in chk["details"]:
                        st.code(d)

    if hard_fail > 0:
        st.error("🚫 Критические ошибки в данных. Генерация отчёта остановлена.")
        st.stop()

    # ── Шаги 3+4: generate_report → review_report (цикл, макс. 3 попытки) ─────
    MAX_ATTEMPTS = 3
    feedback = None
    approved = False

    for attempt in range(1, MAX_ATTEMPTS + 1):
        label_a = f"{attempt}/{MAX_ATTEMPTS}"

        with st.status(
            f"Попытка {label_a}: **Шаг 3 / 4** — Генерация отчёта (GPT-4o)...",
            expanded=True,
        ) as step3:
            try:
                r3 = json.loads(
                    generate_report(
                        metrics_path=metrics_path,
                        validation_path=validation_path,
                        feedback=feedback,
                    )
                )
                report_path = r3["report_path"]
                step3.update(
                    label=f"✅ Попытка {label_a}: Шаг 3 / 4 — Отчёт сформирован → {report_path}",
                    state="complete",
                )
            except Exception as e:
                step3.update(label=f"❌ Ошибка генерации отчёта: {e}", state="error")
                st.stop()

        with st.status(
            f"Попытка {label_a}: **Шаг 4 / 4** — Ревью отчёта (GPT-4o)...",
            expanded=True,
        ) as step4:
            try:
                review_data = json.loads(
                    review_report(report_path=report_path, metrics_path=metrics_path)
                )
                approved = review_data["approved"]
                issues   = review_data.get("issues", [])

                if approved:
                    step4.update(
                        label=f"✅ Попытка {label_a}: Шаг 4 / 4 — Отчёт одобрен ревьюером",
                        state="complete",
                    )
                else:
                    feedback = review_data["feedback"]
                    step4.update(
                        label=f"⚠️ Попытка {label_a}: Шаг 4 / 4 — Найдено {len(issues)} замечаний, перегенерация...",
                        state="complete",
                    )
                    with st.expander(f"Замечания ревьюера (попытка {label_a})", expanded=True):
                        for issue in issues:
                            st.warning(f"• {issue}")
            except Exception as e:
                step4.update(label=f"❌ Ошибка ревью: {e}", state="error")
                st.stop()

        if approved:
            break

    if not approved:
        st.warning(f"⚠️ Достигнут лимит попыток ({MAX_ATTEMPTS}). Используется последняя версия отчёта.")

    st.success("🎉 Готово! Агент завершил работу.", icon="✅")
    st.divider()

    # ── Метрики ────────────────────────────────────────────────────────────────
    st.subheader("📈 Ежемесячные KPI")
    df = pd.DataFrame(metrics_data)
    df["churn_rate_%"] = (df["churn_rate"] * 100).round(2)
    display_df = df[
        ["month", "active_users", "paid_users", "churned_users",
         "monthly_revenue", "churn_rate_%", "arpu"]
    ].rename(columns={
        "month": "Месяц",
        "active_users": "Активные",
        "paid_users": "Оплатили",
        "churned_users": "Отток",
        "monthly_revenue": "Выручка ($)",
        "churn_rate_%": "Churn rate (%)",
        "arpu": "ARPU ($)",
    })
    st.dataframe(display_df, use_container_width=True, hide_index=True)

    # ── Отчёт ─────────────────────────────────────────────────────────────────
    st.divider()
    st.subheader("📄 Бизнес-отчёт")
    with open(report_path, encoding="utf-8") as fh:
        report_text = fh.read()
    st.markdown(report_text)

    # ── Скачивание файлов ─────────────────────────────────────────────────────
    st.divider()
    st.subheader("⬇ Скачать файлы")
    col1, col2, col3 = st.columns(3)

    with col1:
        with open(data_path, "rb") as fh:
            st.download_button(
                label="📥 users.csv",
                data=fh,
                file_name="users.csv",
                mime="text/csv",
                use_container_width=True,
            )
    with col2:
        with open(metrics_path, "rb") as fh:
            st.download_button(
                label="📥 metrics.csv",
                data=fh,
                file_name="metrics.csv",
                mime="text/csv",
                use_container_width=True,
            )
    with col3:
        with open(report_path, "rb") as fh:
            st.download_button(
                label="📥 report.md",
                data=fh,
                file_name="report.md",
                mime="text/markdown",
                use_container_width=True,
            )

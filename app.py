"""
app.py — Streamlit-интерфейс для Gauss Money AI Agent

Запуск локально:  streamlit run app.py
"""

import json
import os

import pandas as pd
import streamlit as st

# ── Page config ────────────────────────────────────────────────────────────────
st.set_page_config(
    page_title="Gauss Money — Churn & Revenue Agent",
    page_icon="📊",
    layout="wide",
)

# ── Gauss Money brand styling ──────────────────────────────────────────────────
st.markdown(
    """
    <link href="https://fonts.googleapis.com/css2?family=Inter:wght@400;500;600;700&display=swap"
          rel="stylesheet">
    <style>
    html, body, [class*="css"] {
        font-family: 'Inter', sans-serif;
    }
    .stMarkdown, .stTextInput, .stButton,
    .stDataFrame, p, li, td, th {
        font-family: 'Inter', sans-serif !important;
    }
    .gauss-header {
        background: linear-gradient(120deg, #0A1628 0%, #1E3A6E 55%, #2563EB 100%);
        border-radius: 14px;
        padding: 30px 40px 26px;
        margin-bottom: 20px;
    }
    .gauss-header h1 {
        color: #FFFFFF !important;
        font-size: 1.85rem !important;
        font-weight: 700 !important;
        margin: 0 0 8px !important;
        letter-spacing: -0.4px;
        font-family: 'Inter', sans-serif !important;
    }
    .gauss-header p {
        color: #BFD4F7 !important;
        font-size: 0.97rem !important;
        margin: 0 !important;
        font-family: 'Inter', sans-serif !important;
    }
    button[data-testid="baseButton-primary"],
    div.stButton > button[kind="primary"] {
        background: #000000 !important;
        background-color: #000000 !important;
        color: #FFFFFF !important;
        border: none !important;
        border-radius: 10px !important;
        font-weight: 600 !important;
        font-size: 1rem !important;
        padding: 0.65rem 1.6rem !important;
        font-family: 'Inter', sans-serif !important;
        letter-spacing: 0.1px;
        transition: opacity .15s ease !important;
    }
    button[data-testid="baseButton-primary"]:hover,
    div.stButton > button[kind="primary"]:hover { opacity: 0.82 !important; }
    div.stDownloadButton > button {
        border-radius: 8px !important;
        font-family: 'Inter', sans-serif !important;
        font-weight: 500 !important;
        border: 1.5px solid #CBD5E1 !important;
    }
    div[data-testid="stDataFrame"] {
        border: 1px solid #E2E8F0 !important;
        border-radius: 10px !important;
        overflow: hidden;
    }
    code, .stMarkdown code, [data-testid="stMarkdownContainer"] code {
        font-family: 'Inter', sans-serif !important;
        font-size: inherit !important;
        background: #F1F5F9 !important;
        color: #0F172A !important;
        padding: 1px 5px !important;
        border-radius: 4px !important;
        border: none !important;
    }
    h2, h3 { color: #0A1628 !important; font-family: 'Inter', sans-serif !important; }
    .chk-row {
        display: flex; align-items: flex-start; gap: 10px;
        padding: 10px 14px; border-radius: 8px; margin-bottom: 5px;
        font-family: 'Inter', sans-serif; font-size: 0.9rem; line-height: 1.5;
    }
    .chk-pass { background: #F0FFF4; color: #166534; }
    .chk-soft { background: #FFFBEB; color: #92400E; }
    .chk-hard { background: #FFF1F2; color: #9F1239; }
    .chk-num  { font-weight: 700; min-width: 20px; }
    .chk-badge {
        font-size: 0.7rem; font-weight: 700; padding: 1px 7px;
        border-radius: 12px; margin-left: 5px; white-space: nowrap;
    }
    .badge-hard { background: #FFE4E6; color: #9F1239; }
    .badge-soft { background: #FEF3C7; color: #92400E; }
    div[data-testid="stNotificationContentInfo"] {
        background-color: #EFF6FF !important;
        border-left: 4px solid #2563EB !important;
    }
    hr { border-color: #E2E8F0 !important; }
    </style>
    """,
    unsafe_allow_html=True,
)

# ── Header ─────────────────────────────────────────────────────────────────────
st.markdown(
    """
    <div class="gauss-header">
        <h1>📊 Gauss Money — Churn &amp; Revenue AI Agent</h1>
        <p>Генерирует синтетические данные по подписочной модели, вычисляет KPI,
        проверяет качество данных и формирует бизнес-отчёт на русском языке через GPT-4o.</p>
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

# ── Session state — хранит результаты между ре-рандерами (нажатие Download и т.п.) ──
if "agent_results" not in st.session_state:
    st.session_state.agent_results = None

# ── Кнопка запуска ─────────────────────────────────────────────────────────────
run_disabled = not bool(api_key)
if run_disabled:
    st.info("Введите OpenAI API Key, чтобы запустить агента.", icon="ℹ️")

if st.button("▶ Запустить агента", disabled=run_disabled, type="primary", use_container_width=True):
    # Сбрасываем предыдущие результаты
    st.session_state.agent_results = None
    os.environ["OPENAI_API_KEY"] = api_key

    from tools.data_generator import generate_data
    from tools.metrics_calculator import calculate_metrics
    from tools.data_validator import validate_data
    from tools.report_generator import generate_report
    from tools.report_reviewer import review_report

    data_path = metrics_path = validation_path = report_path = None
    metrics_data = None
    val_checks = []
    hard_fail = soft_warn = 0

    # ── Pre-step: Генерация синтетических данных ───────────────────────────────
    with st.status("Pre-step — Генерация синтетических данных", expanded=True) as prestep:
        try:
            r0 = json.loads(generate_data())
            data_path = r0["data_path"]
            s = r0["summary"]
            prestep.update(
                label=(
                    f"✅ Pre-step — CSV готов: {s['total_records']:,} записей  "
                    f"({s['basic_users']} basic + {s['premium_users']} premium) → {data_path}"
                ),
                state="complete",
            )
        except Exception as e:
            prestep.update(label=f"❌ Ошибка генерации данных: {e}", state="error")
            st.stop()

    # ── Шаг 1/4: Расчёт KPI (calculate_metrics) ───────────────────────────────
    with st.status("Шаг 1/4 — Расчёт KPI (calculate_metrics)", expanded=True) as step1:
        try:
            r1 = json.loads(calculate_metrics(data_path))
            metrics_path = r1["metrics_path"]
            metrics_data = r1["metrics"]
            step1.update(
                label=f"✅ Шаг 1/4 — Расчёт KPI — метрики готовы → {metrics_path}",
                state="complete",
            )
        except Exception as e:
            step1.update(label=f"❌ Ошибка расчёта метрик: {e}", state="error")
            st.stop()

    # ── Шаг 2/4: Валидация данных (validate_data) ─────────────────────────────
    with st.status(
        "Шаг 2/4 — Валидация данных (validate_data)", expanded=True
    ) as step2:
        try:
            r2 = json.loads(validate_data(data_path, metrics_path))
            validation_path = r2["validation_path"]
            summary_v = r2["summary"]
            hard_fail  = summary_v["hard_failures"]
            soft_warn  = summary_v["soft_warnings"]
            passed     = summary_v["passed_checks"]
            total      = summary_v["total_checks"]
            val_checks = summary_v["checks"]

            # Нумерованный список проверок внутри Step 2
            for i, chk in enumerate(val_checks, 1):
                if chk["passed"]:
                    cls, icon = "chk-pass", "✓"
                    badge = ""
                elif chk["severity"] == "soft":
                    cls, icon = "chk-soft", "⚠"
                    badge = '<span class="chk-badge badge-soft">SOFT</span>'
                else:
                    cls, icon = "chk-hard", "✗"
                    badge = '<span class="chk-badge badge-hard">HARD</span>'

                st.markdown(
                    f'<div class="chk-row {cls}">' +
                    f'<span class="chk-num">{i}.</span>' +
                    f'<span>{icon}&nbsp;<b>{chk["name"]}</b>{badge}<br>' +
                    f'<span style="opacity:.8">{chk["message"]}</span></span></div>',
                    unsafe_allow_html=True,
                )
                if not chk["passed"] and chk.get("details"):
                    with st.expander(f"Детали проверки {i} ({len(chk['details'])} примеров)"):
                        for d in chk["details"][:10]:
                            st.code(d)

            if hard_fail == 0:
                lbl = f"✅ Шаг 2/4 — Валидация данных — {passed}/{total} проверок пройдено"
                if soft_warn:
                    lbl += f" ({soft_warn} предупреждения)"
                step2.update(label=lbl, state="complete")
            else:
                step2.update(
                    label=f"❌ Шаг 2/4 — Валидация данных — {hard_fail} критических ошибок, {soft_warn} предупреждений",
                    state="error",
                )
        except Exception as e:
            step2.update(label=f"❌ Ошибка валидации: {e}", state="error")
            st.stop()

    if hard_fail > 0:
        st.error("🚫 Критические ошибки в данных. Генерация отчёта невозможна.")
        st.stop()

    # ── Шаги 3/4 + 4/4: Генерация → ревью (цикл, макс. 5 попыток) ────────────
    MAX_ATTEMPTS = 3
    feedback = None
    approved = False

    for attempt in range(1, MAX_ATTEMPTS + 1):
        att_label = f"попытка {attempt}/{MAX_ATTEMPTS}"

        with st.status(
            f"Шаг 3/4 — Генерация отчёта ({att_label})",
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
                    label=f"✅ Шаг 3/4 — Генерация отчёта ({att_label}) — готово → {report_path}",
                    state="complete",
                )
            except Exception as e:
                step3.update(label=f"❌ Ошибка генерации отчёта: {e}", state="error")
                st.stop()

        with st.status(
            f"Шаг 4/4 — Ревью отчёта ({att_label})",
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
                        label=f"✅ Шаг 4/4 — Ревью отчёта ({att_label}) — отчёт одобрен",
                        state="complete",
                    )
                else:
                    feedback = review_data["feedback"]
                    step4.update(
                        label=f"⚠ Шаг 4/4 — Ревью отчёта ({att_label}) — найдено {len(issues)} замечаний",
                        state="complete",
                    )
                    with st.expander(f"Замечания ревьюера, {att_label}", expanded=True):
                        for issue in issues:
                            st.warning(f"• {issue}")
            except Exception as e:
                step4.update(label=f"❌ Ошибка ревью: {e}", state="error")
                st.stop()

        if approved:
            break

    # ── Сохраняем результаты в session_state ────────────────────────────────────
    st.session_state.agent_results = {
        "data_path":     data_path,
        "metrics_data":  metrics_data,
        "metrics_path":  metrics_path,
        "report_path":   report_path,
        "val_checks":    val_checks,
        "hard_fail":     hard_fail,
        "soft_warn":     soft_warn,
        "approved":      approved,
        "max_attempts":  MAX_ATTEMPTS,
    }

# ── Отображение результатов (сохраняется при нажатии Download и других кнопок) ─
if st.session_state.agent_results:
    r = st.session_state.agent_results

    st.divider()
    if r["approved"]:
        st.success("🎉 Агент завершил работу. Отчёт одобрен ревьюером.", icon="✅")
    else:
        st.warning(
            f"⚠ Агент завершил работу. Достигнут лимит попыток ({r['max_attempts']}). "
            "Использована последняя версия отчёта."
        )
    st.divider()

    # ── KPI таблица ────────────────────────────────────────────────────────────
    st.subheader("📈 Ежемесячные KPI")
    df = pd.DataFrame(r["metrics_data"])
    df["churn_rate_%"] = (df["churn_rate"] * 100).round(2)
    display_df = df[
        ["month", "active_users", "paid_users", "churned_users",
         "monthly_revenue", "churn_rate_%", "arpu"]
    ].rename(columns={
        "month":          "Месяц",
        "active_users":   "Активные",
        "paid_users":     "Оплатили",
        "churned_users":  "Отток",
        "monthly_revenue":"Выручка ($)",
        "churn_rate_%":   "Churn rate (%)",
        "arpu":           "ARPU ($)",
    })
    st.dataframe(display_df, use_container_width=True, hide_index=True)

    # ── Результаты валидации ───────────────────────────────────────────────────
    st.divider()
    st.subheader("🔍 Результаты валидации данных")
    passed_count = sum(1 for c in r["val_checks"] if c["passed"])
    with st.expander(
        f"{'✅' if r['hard_fail'] == 0 else '❌'} "
        f"{passed_count}/{len(r['val_checks'])} проверок пройдено"
        + (f" · {r['soft_warn']} предупреждения" if r["soft_warn"] else ""),
        expanded=False,
    ):
        for i, chk in enumerate(r["val_checks"], 1):
            if chk["passed"]:
                cls, icon = "chk-pass", "✓"
                badge = ""
            elif chk["severity"] == "soft":
                cls, icon = "chk-soft", "⚠"
                badge = '<span class="chk-badge badge-soft">SOFT</span>'
            else:
                cls, icon = "chk-hard", "✗"
                badge = '<span class="chk-badge badge-hard">HARD</span>'

            st.markdown(
                f'<div class="chk-row {cls}">' +
                f'<span class="chk-num">{i}.</span>' +
                f'<span>{icon}&nbsp;<b>{chk["name"]}</b>{badge}<br>' +
                f'<span style="opacity:.8">{chk["message"]}</span></span></div>',
                unsafe_allow_html=True,
            )

    # ── Бизнес-отчёт ──────────────────────────────────────────────────────────
    st.divider()
    st.subheader("📄 Бизнес-отчёт")
    with open(r["report_path"], encoding="utf-8") as fh:
        report_text = fh.read()
    # Экранируем $ чтобы Streamlit не парсил суммы как LaTeX-формулы
    st.markdown(report_text.replace("$", "\\$"))

    # ── Скачивание файлов ──────────────────────────────────────────────────────
    st.divider()
    st.subheader("⬇ Скачать файлы")
    col1, col2, col3 = st.columns(3)

    with col1:
        with open(r["data_path"], "rb") as fh:
            st.download_button(
                "📥 users.csv", fh, file_name="users.csv",
                mime="text/csv", use_container_width=True,
            )
    with col2:
        with open(r["metrics_path"], "rb") as fh:
            st.download_button(
                "📥 metrics.csv", fh, file_name="metrics.csv",
                mime="text/csv", use_container_width=True,
            )
    with col3:
        with open(r["report_path"], "rb") as fh:
            st.download_button(
                "📥 report.md", fh, file_name="report.md",
                mime="text/markdown", use_container_width=True,
            )

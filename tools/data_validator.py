"""
Tool: data_validator
Runs 8 quality checks split into two severity levels:

  HARD — блокирует генерацию отчёта при провале
    1. No duplicate (user_id, month) pairs
    2. Revenue consistency — metrics.monthly_revenue == sum(amount_paid) from raw CSV
    4. Churn rate in [0, 1] for every month
    5. Payment amount consistency — failed→amount=0; paid→amount=monthly_price
    7. User-flow conservation — active[M] + churned[M] == active[M-1]
    8. All users present in month 1

  SOFT — предупреждение; отчёт генерируется несмотря на провал
    3. Active users monotonically non-increasing
       (в реальных данных возможны реактивации / win-back кампании)
    6. No activity after churn
       (в реальных данных возможны повторные подписки)

Для каждого провалившегося check возвращаются детали:
конкретные месяцы или user_id с аномалиями (до 10 примеров).
"""

import json
import os

import pandas as pd


def validate_data(data_path: str, metrics_path: str) -> str:
    """
    Run quality checks and save data/validation.json.

    Returns
    -------
    str
        JSON: {"validation_path": "data/validation.json", "summary": {...}}
    """
    os.makedirs("data", exist_ok=True)

    df = pd.read_csv(data_path)
    metrics_df = pd.read_csv(metrics_path)

    checks: list[dict] = []

    def add_check(
        name: str,
        passed: bool,
        message: str,
        severity: str = "hard",   # "hard" | "soft"
        details: list[str] | None = None,
    ) -> None:
        checks.append({
            "name":     name,
            "passed":   passed,
            "message":  message,
            "severity": severity,
            "details":  details or [],
        })
        if passed:
            icon = "✓ PASSED"
        elif severity == "soft":
            icon = "⚠ WARNING"
        else:
            icon = "✗ FAILED "
        label = f"[{severity.upper():4s}]"
        print(f"[Validator] {icon} {label} {name}: {message}")
        if details and not passed:
            for d in details[:10]:
                print(f"             → {d}")

    # ── 1. HARD: No duplicate (user_id, month) ────────────────────────────────
    dup_mask = df.duplicated(subset=["user_id", "month"], keep=False)
    dups = int(dup_mask.sum())
    dup_details = []
    if dups:
        sample = (
            df[dup_mask][["user_id", "month"]]
            .drop_duplicates()
            .head(10)
        )
        dup_details = [f"user_id={r.user_id}, month={r.month}" for r in sample.itertuples()]
    add_check(
        "No duplicate (user_id, month) pairs",
        dups == 0,
        f"{dups} duplicate pairs found",
        severity="hard",
        details=dup_details,
    )

    # ── 2. HARD: Revenue cross-check ──────────────────────────────────────────
    rev_bad_months = []
    for _, row in metrics_df.iterrows():
        month = int(row["month"])
        raw_rev = round(float(df.loc[df["month"] == month, "amount_paid"].sum()), 2)
        met_rev = round(float(row["monthly_revenue"]), 2)
        if abs(raw_rev - met_rev) > 0.01:
            rev_bad_months.append(
                f"month={month}: metrics={met_rev:.2f}, raw_sum={raw_rev:.2f}, diff={met_rev-raw_rev:+.2f}"
            )
    add_check(
        "Revenue consistency (all months)",
        len(rev_bad_months) == 0,
        f"{len(rev_bad_months)} months where metrics.monthly_revenue ≠ sum(amount_paid)",
        severity="hard",
        details=rev_bad_months,
    )

    # ── 3. SOFT: Active users monotonically non-increasing ────────────────────
    active_vals = metrics_df["active_users"].values
    reactivation_months = []
    for i in range(1, len(active_vals)):
        if active_vals[i] > active_vals[i - 1]:
            reactivation_months.append(
                f"month {int(metrics_df.iloc[i-1]['month'])}→{int(metrics_df.iloc[i]['month'])}: "
                f"{int(active_vals[i-1])}→{int(active_vals[i])} (+{int(active_vals[i]-active_vals[i-1])})"
            )
    add_check(
        "Active users monotonically non-increasing",
        len(reactivation_months) == 0,
        (
            f"{len(reactivation_months)} month(s) where active_users increased "
            f"(возможны реактивации / win-back кампании)"
            if reactivation_months
            else f"month-1={int(active_vals[0])}, month-12={int(active_vals[-1])}"
        ),
        severity="soft",
        details=reactivation_months,
    )

    # ── 4. HARD: Churn rate in [0, 1] ─────────────────────────────────────────
    bad_cr = metrics_df[(metrics_df["churn_rate"] < 0) | (metrics_df["churn_rate"] > 1)]
    bad_cr_details = [
        f"month={int(r.month)}: churn_rate={r.churn_rate:.4f}"
        for r in bad_cr.itertuples()
    ]
    add_check(
        "Churn rate in [0, 1]",
        len(bad_cr) == 0,
        f"{len(bad_cr)} months with churn_rate outside [0, 1]",
        severity="hard",
        details=bad_cr_details,
    )

    # ── 5. HARD: Payment amount consistency ───────────────────────────────────
    failed_nonzero_df = df[(df["payment_status"] == "failed") & (df["amount_paid"] > 0)]
    paid_wrong_df     = df[(df["payment_status"] == "paid") & (df["amount_paid"] != df["monthly_price"])]
    pay_details = (
        [f"user_id={r.user_id}, month={r.month}: status=failed, amount={r.amount_paid}"
         for r in failed_nonzero_df.head(5).itertuples()]
        + [f"user_id={r.user_id}, month={r.month}: status=paid, amount={r.amount_paid}, price={r.monthly_price}"
           for r in paid_wrong_df.head(5).itertuples()]
    )
    add_check(
        "Payment amount consistency",
        len(failed_nonzero_df) == 0 and len(paid_wrong_df) == 0,
        f"failed rows with amount>0: {len(failed_nonzero_df)}; paid rows with wrong amount: {len(paid_wrong_df)}",
        severity="hard",
        details=pay_details,
    )

    # ── 6. SOFT: No activity after churn ──────────────────────────────────────
    user_last_month = df.groupby("user_id")["month"].max().rename("last_month")
    churned_df = df.loc[~df["is_active"]].copy()
    post_churn_users: list[str] = []
    if len(churned_df) > 0:
        merged = churned_df.merge(user_last_month, on="user_id")
        bad_rows = merged[merged["month"] != merged["last_month"]]
        post_churn_users = [
            f"user_id={r.user_id}: churn_record month={r.month}, last_record month={r.last_month}"
            for r in bad_rows.head(10).itertuples()
        ]
    add_check(
        "No activity after churn",
        len(post_churn_users) == 0,
        (
            f"{len(post_churn_users)} пользователей с записями после churn-события "
            f"(возможны повторные подписки)"
            if post_churn_users
            else "0 records found after a user's churn event"
        ),
        severity="soft",
        details=post_churn_users,
    )

    # ── 7. HARD: User-flow conservation ───────────────────────────────────────
    flow_bad = []
    for i in range(1, len(metrics_df)):
        prev_active  = int(metrics_df.iloc[i - 1]["active_users"])
        curr_active  = int(metrics_df.iloc[i]["active_users"])
        curr_churned = int(metrics_df.iloc[i]["churned_users"])
        month        = int(metrics_df.iloc[i]["month"])
        if prev_active != curr_active + curr_churned:
            flow_bad.append(
                f"month={month}: prev_active={prev_active}, "
                f"active={curr_active}, churned={curr_churned}, "
                f"sum={curr_active+curr_churned} (diff={curr_active+curr_churned-prev_active:+d})"
            )
    add_check(
        "User-flow conservation (active[M] + churned[M] == active[M-1])",
        len(flow_bad) == 0,
        f"{len(flow_bad)} months with inconsistent user counts",
        severity="hard",
        details=flow_bad,
    )

    # ── 8. HARD: All users present in month 1 ─────────────────────────────────
    total_users  = df["user_id"].nunique()
    month1_users = df.loc[df["month"] == 1, "user_id"].nunique()
    missing = total_users - month1_users
    missing_details = []
    if missing > 0:
        all_users   = set(df["user_id"].unique())
        month1_set  = set(df.loc[df["month"] == 1, "user_id"].unique())
        missing_ids = sorted(all_users - month1_set)[:10]
        missing_details = [f"user_id={u}" for u in missing_ids]
    add_check(
        "All users start in month 1",
        missing == 0,
        f"month-1 users: {month1_users}, total unique users: {total_users}"
        + (f" ({missing} missing from month 1)" if missing else ""),
        severity="hard",
        details=missing_details,
    )

    # ── Summary ───────────────────────────────────────────────────────────────
    hard_checks   = [c for c in checks if c["severity"] == "hard"]
    soft_checks   = [c for c in checks if c["severity"] == "soft"]
    hard_passed   = all(c["passed"] for c in hard_checks)
    passed_count  = sum(c["passed"] for c in checks)

    result = {
        "all_hard_passed":  hard_passed,    # агент смотрит именно сюда
        "total_checks":     len(checks),
        "passed_checks":    passed_count,
        "hard_failures":    sum(1 for c in hard_checks if not c["passed"]),
        "soft_warnings":    sum(1 for c in soft_checks if not c["passed"]),
        "checks":           checks,
    }

    validation_path = os.path.join("data", "validation.json")
    with open(validation_path, "w", encoding="utf-8") as fh:
        json.dump(result, fh, indent=2)

    hard_fail = result["hard_failures"]
    soft_warn = result["soft_warnings"]
    print(
        f"[Validator] {passed_count}/{len(checks)} passed | "
        f"hard failures: {hard_fail} | soft warnings: {soft_warn} | "
        f"saved → {validation_path}"
    )

    return json.dumps({"validation_path": validation_path, "summary": result})

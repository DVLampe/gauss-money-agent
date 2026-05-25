"""
Tool: metrics_calculator
Computes monthly subscription KPIs from the generated CSV.

Metric definitions
------------------
active_users  : users with is_active=True at end of month M
paid_users    : users where payment_status='paid' in month M
churned_users : users with is_active=False in month M (their last record)
monthly_revenue : sum(amount_paid) for month M
churn_rate    : churned_users[M] / active_users[M-1]   (0.0 for month 1)
arpu          : monthly_revenue[M] / active_users[M]

Note: active_users[M-1] == total rows in month M, because every user who
entered month M either stayed active or churned exactly once.
"""

import json
import os

import pandas as pd


def calculate_metrics(data_path: str) -> str:
    """
    Calculate monthly KPIs and save metrics.csv.

    Returns
    -------
    str
        JSON: {"metrics_path": "data/metrics.csv", "metrics": [...]}
    """
    os.makedirs("data", exist_ok=True)

    df = pd.read_csv(data_path)
    months = sorted(df["month"].unique())

    metrics = []
    prev_active: int | None = None

    for month in months:
        m = df[df["month"] == month]

        active_users  = int(m["is_active"].sum())
        paid_users    = int((m["payment_status"] == "paid").sum())
        churned_users = int((~m["is_active"]).sum())
        revenue       = round(float(m["amount_paid"].sum()), 2)

        if month == 1:
            churn_rate = 0.0
        else:
            churn_rate = round(churned_users / prev_active, 4) if prev_active else 0.0

        arpu = round(revenue / active_users, 2) if active_users > 0 else 0.0

        metrics.append({
            "month":           int(month),
            "active_users":    active_users,
            "paid_users":      paid_users,
            "churned_users":   churned_users,
            "monthly_revenue": revenue,
            "churn_rate":      churn_rate,
            "arpu":            arpu,
        })

        prev_active = active_users

    metrics_df = pd.DataFrame(metrics)
    metrics_path = os.path.join("data", "metrics.csv")
    metrics_df.to_csv(metrics_path, index=False)

    print("[MetricsCalculator] Monthly metrics:")
    print(metrics_df.to_string(index=False))

    return json.dumps({"metrics_path": metrics_path, "metrics": metrics})

"""
Tool: data_generator
Generates synthetic subscription data for 1000 users over 12 months.

Design decisions:
- Each user gets a single "churn month" drawn up front via rolling churn probabilities.
- Churn probability decreases over time (early-phase drop-off → loyal survivors).
- Payment failures (5%) are independent of churn — a user can fail a payment but stay subscribed.
- Churned users receive one final record with is_active=False, amount_paid=0 (last billing attempt failed).
- After the churn month the user has no further records.
"""

import json
import os

import numpy as np
import pandas as pd


def generate_data(n_users: int = 1000, n_months: int = 12, seed: int = 42) -> str:
    """
    Generate synthetic subscription CSV.

    Returns
    -------
    str
        JSON: {"data_path": "data/users.csv", "summary": {...}}
    """
    np.random.seed(seed)
    os.makedirs("data", exist_ok=True)

    # ── Plans ──────────────────────────────────────────────────────────────────
    plans = np.random.choice(["basic", "premium"], size=n_users, p=[0.6, 0.4])
    prices = np.where(plans == "basic", 9.99, 19.99).astype(float)

    # ── Monthly churn rates (month → probability of churning at start of that month)
    # Realistic SaaS pattern: high early churn, stabilises as loyalists remain.
    churn_rates: dict[int, float] = {
        2: 0.08, 3: 0.07, 4: 0.06,
        5: 0.05, 6: 0.05, 7: 0.05, 8: 0.05,
        9: 0.04, 10: 0.04, 11: 0.04, 12: 0.04,
    }
    payment_failure_rate = 0.05  # 5 % of active users fail payment each month

    # ── Determine churn month per user ────────────────────────────────────────
    # n_months + 1 means "survived the entire observation period"
    churn_month = np.full(n_users, n_months + 1, dtype=int)
    for month in range(2, n_months + 1):
        rate = churn_rates[month]
        eligible = churn_month == n_months + 1      # hasn't churned yet
        will_churn = eligible & (np.random.rand(n_users) < rate)
        churn_month[will_churn] = month

    # ── Generate records ──────────────────────────────────────────────────────
    records = []
    for uid in range(n_users):
        last_month = min(churn_month[uid], n_months)

        for month in range(1, last_month + 1):
            is_churning = month == churn_month[uid]

            if is_churning:
                records.append({
                    "user_id": f"user_{uid:04d}",
                    "month": month,
                    "plan": plans[uid],
                    "monthly_price": float(prices[uid]),
                    "payment_status": "failed",
                    "amount_paid": 0.0,
                    "is_active": False,
                })
            else:
                paid = np.random.rand() > payment_failure_rate
                records.append({
                    "user_id": f"user_{uid:04d}",
                    "month": month,
                    "plan": plans[uid],
                    "monthly_price": float(prices[uid]),
                    "payment_status": "paid" if paid else "failed",
                    "amount_paid": float(prices[uid]) if paid else 0.0,
                    "is_active": True,
                })

    df = (
        pd.DataFrame(records)
        .sort_values(["user_id", "month"])
        .reset_index(drop=True)
    )

    data_path = os.path.join("data", "users.csv")
    df.to_csv(data_path, index=False)

    print(
        f"[DataGenerator] {len(df):,} records | {n_users:,} users | "
        f"{n_months} months | saved → {data_path}"
    )

    return json.dumps({
        "data_path": data_path,
        "summary": {
            "total_records": len(df),
            "total_users": n_users,
            "months": n_months,
            "basic_users": int((plans == "basic").sum()),
            "premium_users": int((plans == "premium").sum()),
        },
    })

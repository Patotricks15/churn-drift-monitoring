import argparse
from pathlib import Path

import pandas as pd

from monitor.config import (
    DATASET_NAME,
    FEATURE_COLUMNS,
    H_DAYS,
    MODEL_VERSION,
    MONITOR_START,
    PSI_BINS,
    PSI_THRESHOLD,
    REFERENCE_END,
    ROLLING_WINDOW_MONTHS,
)
from monitor.performance import monthly_metrics, rolling_pr_auc
from monitor.plots import plot_monitor
from monitor.psi import monthly_feature_psi
from monitor.split import split_by_snapshot
from monitor.trigger import retrain_decision


def main() -> None:
    parser = argparse.ArgumentParser(description="Monitor feature drift and PR-AUC for the churn score.")
    parser.add_argument("--scores", required=True, help="CSV with snapshot_date, churned, proba and features.")
    parser.add_argument("--out-dir", default="reports", help="Directory for tables and figures.")
    args = parser.parse_args()

    out_dir = Path(args.out_dir)
    (out_dir / "figures").mkdir(parents=True, exist_ok=True)

    scores = pd.read_csv(args.scores, parse_dates=["snapshot_date"])
    scores["month"] = scores["snapshot_date"].dt.to_period("M")
    reference, monitored = split_by_snapshot(
        scores, "snapshot_date", REFERENCE_END, MONITOR_START, H_DAYS
    )

    psi = monthly_feature_psi(reference, monitored, FEATURE_COLUMNS, "month", PSI_BINS)
    perf = monthly_metrics(monitored, "month", "churned", "proba")
    roll = rolling_pr_auc(monitored, "month", "churned", "proba", ROLLING_WINDOW_MONTHS)

    monthly = (
        perf.merge(roll, on="month")
        .merge(psi.max(axis=1).rename("max_psi").reset_index(), on="month")
        .merge(psi.idxmax(axis=1).rename("max_psi_feature").reset_index(), on="month")
    )
    baseline = monthly["pr_auc"].iloc[0]
    decisions = [
        retrain_decision(row.max_psi, row.pr_auc, baseline) for row in monthly.itertuples()
    ]
    monthly["retrain"] = [d.retrain for d in decisions]
    monthly["reasons"] = ["; ".join(d.reasons) for d in decisions]

    psi.to_csv(out_dir / "monitor_psi_by_month.csv")
    monthly.to_csv(out_dir / "monitor_monthly.csv", index=False)

    title = (
        f"{DATASET_NAME}: monitored snapshots from {MONITOR_START}, H={H_DAYS} days, "
        f"reference to {REFERENCE_END}\n{MODEL_VERSION}"
    )
    plot_monitor(monthly, "month", title, out_dir / "figures" / "monitor_psi_pr_auc.png", PSI_THRESHOLD)

    print(monthly[["month", "n", "churn_rate", "pr_auc", "rolling_pr_auc", "max_psi", "retrain"]].to_string(index=False))
    print(f"Baseline PR-AUC (first monitored month): {baseline:.3f}")


if __name__ == "__main__":
    main()

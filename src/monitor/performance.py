"""Monthly and rolling discrimination metrics for the frozen churn score."""

import numpy as np
import pandas as pd
from sklearn.metrics import average_precision_score, roc_auc_score


def _metrics(y: np.ndarray, score: np.ndarray) -> dict:
    # PR-AUC and ROC-AUC are undefined when a month has only one class.
    defined = 0 < y.sum() < len(y)
    return {
        "n": len(y),
        "churn_rate": float(y.mean()),
        "pr_auc": float(average_precision_score(y, score)) if defined else np.nan,
        "roc_auc": float(roc_auc_score(y, score)) if defined else np.nan,
    }


def monthly_metrics(df: pd.DataFrame, month_col: str, label_col: str, score_col: str) -> pd.DataFrame:
    rows = []
    for month, group in df.groupby(month_col):
        rows.append(
            {
                month_col: month,
                **_metrics(group[label_col].to_numpy(), group[score_col].to_numpy()),
            }
        )
    return pd.DataFrame(rows)


def rolling_pr_auc(
    df: pd.DataFrame, month_col: str, label_col: str, score_col: str, window: int
) -> pd.DataFrame:
    """PR-AUC on the pooled rows of the current month and the window-1 months before it."""
    months = sorted(df[month_col].unique())
    rows = []
    for i, month in enumerate(months):
        pooled = df[df[month_col].isin(months[max(0, i - window + 1) : i + 1])]
        y = pooled[label_col].to_numpy()
        pr = average_precision_score(y, pooled[score_col].to_numpy()) if 0 < y.sum() < len(y) else np.nan
        rows.append({month_col: month, "rolling_pr_auc": float(pr)})
    return pd.DataFrame(rows)

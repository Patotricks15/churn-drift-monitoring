"""Plots for the monthly monitor."""

from pathlib import Path

import matplotlib

matplotlib.use("Agg")
import matplotlib.pyplot as plt  # noqa: E402


def plot_monitor(
    monthly,
    month_col: str,
    title: str,
    path: str | Path,
    psi_threshold: float,
) -> None:
    """Bars show the largest feature PSI per month; the line shows PR-AUC on the right axis."""
    labels = [str(m) for m in monthly[month_col]]
    fig, ax = plt.subplots(figsize=(9, 4.8))
    ax.bar(labels, monthly["max_psi"], color="tab:gray", label="max feature PSI")
    ax.axhline(psi_threshold, ls="--", color="tab:red", label=f"PSI alert {psi_threshold}")
    ax.set_ylabel("PSI")
    ax.set_xlabel("Snapshot month")

    ax2 = ax.twinx()
    ax2.plot(labels, monthly["pr_auc"], marker="o", color="tab:blue", label="PR-AUC")
    ax2.plot(labels, monthly["rolling_pr_auc"], marker="s", ls=":", color="tab:green", label="rolling PR-AUC")
    ax2.set_ylabel("PR-AUC")
    ax2.set_ylim(0, 1)

    handles, names = [], []
    for axis in (ax, ax2):
        h, n = axis.get_legend_handles_labels()
        handles += h
        names += n
    ax.legend(handles, names, loc="upper left", fontsize=8)
    ax.set_title(title, fontsize=9)
    fig.tight_layout()
    fig.savefig(path, dpi=150)
    plt.close(fig)

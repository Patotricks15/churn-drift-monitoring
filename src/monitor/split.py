"""Snapshot cut between the reference period and the monitored period."""

import pandas as pd


def split_by_snapshot(
    df: pd.DataFrame,
    date_col: str,
    reference_end: str,
    monitor_start: str,
    h_days: int,
) -> tuple[pd.DataFrame, pd.DataFrame]:
    """Reference rows are on or before reference_end. Monitored rows are on or after monitor_start.

    Rows in between are dropped. monitor_start must be at least h_days after reference_end, so the
    reference labels are observed before any monitored snapshot.
    """
    ref_end = pd.Timestamp(reference_end)
    mon_start = pd.Timestamp(monitor_start)
    if (mon_start - ref_end).days < h_days:
        raise ValueError(
            f"monitor_start {mon_start.date()} is less than {h_days} days after "
            f"reference_end {ref_end.date()}."
        )
    reference = df[df[date_col] <= ref_end]
    monitored = df[df[date_col] >= mon_start]
    return reference, monitored

"""Population stability index (PSI) for numeric features."""

import numpy as np
import pandas as pd

from monitor.config import PSI_BINS

EPS = 1e-6


def _finite(values) -> np.ndarray:
    arr = np.asarray(values, dtype=float)
    return arr[np.isfinite(arr)]


def bin_edges(reference, n_bins: int = 10) -> np.ndarray:
    """Interior quantile edges of the reference sample. Bins are right-open and cover all reals."""
    ref = _finite(reference)
    if ref.size == 0:
        raise ValueError("reference sample has no finite values")
    return np.unique(np.quantile(ref, np.linspace(0, 1, n_bins + 1)[1:-1]))


def bin_shares(values, edges: np.ndarray) -> np.ndarray:
    """Share of finite values in each bin. Non-finite values (for example missing) are dropped."""
    vals = _finite(values)
    if vals.size == 0:
        raise ValueError("sample has no finite values")
    idx = np.searchsorted(edges, vals, side="right")
    counts = np.bincount(idx, minlength=len(edges) + 1)
    return counts / counts.sum()


def psi_from_shares(expected: np.ndarray, actual: np.ndarray) -> float:
    """PSI = sum (a_i - e_i) * ln(a_i / e_i). Empty bins are floored at EPS to keep the log finite."""
    e = np.clip(expected, EPS, None)
    a = np.clip(actual, EPS, None)
    return float(np.sum((a - e) * np.log(a / e)))


def feature_psi(reference, current, n_bins: int = PSI_BINS) -> float:
    edges = bin_edges(reference, n_bins)
    return psi_from_shares(bin_shares(reference, edges), bin_shares(current, edges))


def monthly_feature_psi(
    reference: pd.DataFrame,
    monitored: pd.DataFrame,
    features: list[str],
    month_col: str,
    n_bins: int = PSI_BINS,
) -> pd.DataFrame:
    """PSI of each feature for each month, with bins fixed by the reference sample."""
    edges = {f: bin_edges(reference[f], n_bins) for f in features}
    expected = {f: bin_shares(reference[f], edges[f]) for f in features}
    rows = {}
    for month, group in monitored.groupby(month_col):
        rows[month] = {
            f: psi_from_shares(expected[f], bin_shares(group[f], edges[f])) for f in features
        }
    out = pd.DataFrame.from_dict(rows, orient="index")
    out.index.name = month_col
    return out

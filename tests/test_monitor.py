import math

import numpy as np
import pandas as pd
import pytest

from monitor.performance import monthly_metrics, rolling_pr_auc
from monitor.psi import bin_edges, feature_psi, monthly_feature_psi, psi_from_shares
from monitor.split import split_by_snapshot
from monitor.trigger import retrain_decision


def test_psi_matches_hand_computed_value():
    expected = np.array([0.5, 0.5])
    actual = np.array([0.8, 0.2])
    want = 0.3 * math.log(0.8 / 0.5) + (-0.3) * math.log(0.2 / 0.5)
    assert psi_from_shares(expected, actual) == pytest.approx(want)


def test_psi_is_zero_for_identical_samples():
    x = np.arange(200, dtype=float)
    assert feature_psi(x, x.copy()) == pytest.approx(0.0, abs=1e-12)


def test_psi_flags_a_large_shift():
    reference = np.arange(1000, dtype=float)
    shifted = reference + 500
    assert feature_psi(reference, shifted) > 0.25


def test_monthly_feature_psi_uses_reference_bins_per_month():
    reference = pd.DataFrame({"x": np.arange(100, dtype=float), "month": ["ref"] * 100})
    monitored = pd.DataFrame(
        {
            "x": np.concatenate([np.arange(100, dtype=float), np.full(100, 99.0)]),
            "month": ["2018-01"] * 100 + ["2018-02"] * 100,
        }
    )
    table = monthly_feature_psi(reference, monitored, ["x"], "month", n_bins=10)
    assert list(table.index) == ["2018-01", "2018-02"]
    assert table.loc["2018-01", "x"] == pytest.approx(0.0, abs=1e-12)
    assert table.loc["2018-02", "x"] > 0.25
    assert bin_edges(reference["x"], 10).size == 9


def test_split_keeps_reference_and_monitored_rows_apart():
    df = pd.DataFrame({"snapshot_date": pd.to_datetime(["2017-10-01", "2017-11-01", "2018-01-01", "2018-02-01"])})
    reference, monitored = split_by_snapshot(df, "snapshot_date", "2017-10-01", "2018-01-01", h_days=90)
    assert list(reference["snapshot_date"].dt.strftime("%Y-%m-%d")) == ["2017-10-01"]
    assert list(monitored["snapshot_date"].dt.strftime("%Y-%m-%d")) == ["2018-01-01", "2018-02-01"]


def test_split_rejects_monitoring_before_the_label_gap_closes():
    df = pd.DataFrame({"snapshot_date": pd.to_datetime(["2017-10-01"])})
    with pytest.raises(ValueError):
        split_by_snapshot(df, "snapshot_date", "2017-10-01", "2017-12-01", h_days=90)


def test_monthly_metrics_group_by_month_and_handle_single_class():
    df = pd.DataFrame(
        {
            "month": ["2018-01"] * 4 + ["2018-02"] * 3,
            "churned": [0, 1, 0, 1, 0, 0, 0],
            "proba": [0.1, 0.9, 0.2, 0.8, 0.3, 0.2, 0.1],
        }
    )
    out = monthly_metrics(df, "month", "churned", "proba").set_index("month")
    assert out.loc["2018-01", "pr_auc"] == pytest.approx(1.0)
    assert out.loc["2018-01", "n"] == 4
    assert out.loc["2018-01", "churn_rate"] == pytest.approx(0.5)
    assert np.isnan(out.loc["2018-02", "pr_auc"])


def test_rolling_pr_auc_pools_rows_across_the_window():
    df = pd.DataFrame(
        {
            "month": ["2018-01", "2018-01", "2018-02", "2018-02", "2018-03", "2018-03"],
            "churned": [1, 0, 0, 1, 1, 0],
            "proba": [0.9, 0.1, 0.1, 0.9, 0.9, 0.1],
        }
    )
    out = rolling_pr_auc(df, "month", "churned", "proba", window=2).set_index("month")
    assert out.loc["2018-01", "rolling_pr_auc"] == pytest.approx(1.0)
    assert out.loc["2018-03", "rolling_pr_auc"] == pytest.approx(1.0)


def test_trigger_fires_on_psi_and_on_pr_auc_drop():
    assert retrain_decision(max_psi=0.3, pr_auc=0.5, baseline_pr_auc=0.5).retrain
    drop = retrain_decision(max_psi=0.1, pr_auc=0.35, baseline_pr_auc=0.5)
    assert drop.retrain and len(drop.reasons) == 1


def test_trigger_stays_quiet_when_stable_or_undefined():
    assert not retrain_decision(max_psi=0.1, pr_auc=0.45, baseline_pr_auc=0.5).retrain
    assert not retrain_decision(max_psi=np.nan, pr_auc=np.nan, baseline_pr_auc=0.5).retrain

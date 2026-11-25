# churn-drift-monitoring (CH06)

**Question:** When does the churn model stop being reliable?

This repository monitors a frozen churn model over time. It computes the population stability index (PSI) for each feature by month, the monthly PR-AUC and ROC-AUC, a rolling three-month PR-AUC, and a retrain decision. A feature alert fires when PSI > 0.25. A retrain is recommended when a feature alert fires, or when monthly PR-AUC is more than 20% below the baseline month.

**Status:** the code is tested with unit tests. **Results are not yet produced.** The notebook has not been executed, because the data is not on this machine. Monthly numbers and the first retrain date will be added after a Kaggle run.

## Data

- **Dataset:** Olist Brazilian E-Commerce, Kaggle slug [`olistbr/brazilian-ecommerce`](https://www.kaggle.com/datasets/olistbr/brazilian-ecommerce).
- **License (per Kaggle metadata):** CC BY-NC-SA 4.0. Non-commercial use only.
- **Read path:** `/kaggle/input/brazilian-ecommerce/` (see [data/README.md](data/README.md)).
- **Metadata read:** 2026-11-08 (plan Appendix A). No files were downloaded.

## Setup

- **Churn window:** `H = 90` days. A customer churns if no purchase occurs in `[snapshot, snapshot + 90)`. Single-order customers are excluded, as in CH02.
- **Frozen model:** LightGBM trained on monthly snapshots from 2017-04-01 to 2017-10-01. It is never refit during monitoring.
- **Reference period:** snapshots on or before 2017-10-01. These set the PSI bins (deciles of each feature).
- **Monitored period:** snapshots from 2018-01-01 to 2018-05-01 (last label window ends 2018-07-30). The cut is at least `H` days after the reference end, so reference labels are observed before monitoring starts.
- **Metric:** PR-AUC is primary, because churn is imbalanced. ROC-AUC is reported beside it.
- **Baseline:** the first monitored month (2018-01) sets the baseline PR-AUC for the 20% drop rule.

The monitored features are the CH02 RFM and behaviour features (`recency_days`, `frequency`, `monetary`, and others).

## Method

- **PSI:** `PSI = Σ (a_i − e_i) · ln(a_i / e_i)`, where `e_i` and `a_i` are the reference and monitored shares in decile bin `i`. Empty bins are floored at `1e-6`. Non-finite values are dropped.
- **Retrain rule:** retrain when any feature's PSI > 0.25, or when PR-AUC is more than 20% below the baseline. The 20% is a choice made here, not a value from the plan.
- **Outputs:** `monitor_psi_by_month.csv`, `monitor_monthly.csv` (metrics, max PSI, trigger and reasons), and `figures/monitor_psi_pr_auc.png`.

## Tests

`tests/test_monitor.py` covers:

- PSI: the formula on a hand-computed example, zero for identical samples, a large value for a shift, and per-month binning against the reference.
- Snapshot cut: reference and monitored rows do not overlap, and monitoring that starts less than `H` days after the reference end is rejected.
- Monthly metrics: grouping by month, PR-AUC of 1.0 for perfect ranking, and NaN for a single-class month.
- Rolling PR-AUC: pooling across the window.
- Trigger: fires on PSI or on a PR-AUC drop, and stays quiet when stable or undefined.

## Reproduce

```bash
make install
make test
```

The notebook in `notebooks/` builds the monthly scores with the CH02 package, then runs `scripts/run_monitor.py` inside a Kaggle Notebook with the Olist dataset attached.

## Limitation

The monitored window has only five monthly snapshots, and the model is trained on the same early period that sets the PSI bins. Results show drift and PR-AUC changes over a short horizon for one Brazilian marketplace. They do not show how the model would behave after a retrain.

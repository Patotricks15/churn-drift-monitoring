DATASET_NAME = "Olist Brazilian E-Commerce"
MODEL_VERSION = "CH06 v0.1 (frozen LightGBM, trained on snapshots up to 2017-10-01)"

# Churn window in days, as in CH01 and CH02.
H_DAYS = 90

# Reference (training-time) feature distribution ends here. Monitored snapshots start at least
# H_DAYS after it, so the reference labels are fully observed before monitoring begins.
REFERENCE_END = "2017-10-01"
MONITOR_START = "2018-01-01"

PSI_BINS = 10
PSI_THRESHOLD = 0.25
# Retrain when PR-AUC falls more than this fraction below the baseline month.
PR_AUC_DROP = 0.20
ROLLING_WINDOW_MONTHS = 3

FEATURE_COLUMNS = [
    "recency_days",
    "frequency",
    "monetary",
    "tenure_days",
    "avg_order_value",
    "mean_days_between_orders",
    "mean_freight",
    "distinct_products",
]

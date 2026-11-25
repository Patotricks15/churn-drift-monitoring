# Data

Raw data is not committed and was not downloaded to this machine.

- **Kaggle slug:** `olistbr/brazilian-ecommerce`
- **License (per Kaggle metadata):** CC BY-NC-SA 4.0. Non-commercial use only.
- **Read path:** `/kaggle/input/brazilian-ecommerce/`. The data is read in place from a Kaggle Notebook with the dataset attached. It is never copied into this repository.
- **Metadata read date:** 2026-11-08 (plan Appendix A, Kaggle dataset metadata only).
- **Files read on this machine:** none.

CH06 builds its monthly scores from the Olist order files through the CH02 package (`churn.data`, `churn.pipeline`). The monitor itself reads only the exported scores CSV (`ch06_monthly_scores.csv`), which the notebook writes.

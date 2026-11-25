"""Retrain trigger: feature drift or a PR-AUC drop relative to the baseline month."""

from dataclasses import dataclass

import numpy as np

from monitor.config import PR_AUC_DROP, PSI_THRESHOLD


@dataclass(frozen=True)
class RetrainDecision:
    retrain: bool
    reasons: tuple[str, ...]


def retrain_decision(
    max_psi: float,
    pr_auc: float,
    baseline_pr_auc: float,
    psi_threshold: float = PSI_THRESHOLD,
    pr_auc_drop: float = PR_AUC_DROP,
) -> RetrainDecision:
    """Retrain if any feature PSI exceeds the threshold, or PR-AUC is more than pr_auc_drop below baseline.

    A missing PR-AUC (single-class month) or PSI (NaN) does not trigger on its own.
    """
    reasons = []
    if max_psi > psi_threshold:
        reasons.append(f"max feature PSI {max_psi:.3f} > {psi_threshold}")
    if not np.isnan(pr_auc) and pr_auc < baseline_pr_auc * (1 - pr_auc_drop):
        reasons.append(
            f"PR-AUC {pr_auc:.3f} is more than {pr_auc_drop:.0%} below baseline {baseline_pr_auc:.3f}"
        )
    return RetrainDecision(retrain=bool(reasons), reasons=tuple(reasons))

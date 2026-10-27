"""
Unit tests for Phase 5D: scripts/leaderboard_sim.py
"""

import numpy as np
import pandas as pd
from scripts.leaderboard_sim import (
    FINDINGS,
    evaluate_auc_per_label,
    bootstrap_leaderboard_auc,
    simulate_and_diagnose,
)


def test_evaluate_and_simulation():
    np.random.seed(42)
    n = 58
    studies = [f"study_{i:03d}" for i in range(n)]

    gold_df = pd.DataFrame({"StudyInstanceUID": studies})
    oof_df = pd.DataFrame({"StudyInstanceUID": studies})
    soft_df = pd.DataFrame({"StudyInstanceUID": studies})

    for f in FINDINGS:
        gold_df[f] = (np.random.rand(n) > 0.6).astype(int)
        # Moderate correlation with gold
        oof_df[f] = np.clip(gold_df[f] * 0.7 + 0.15 + np.random.randn(n) * 0.1, 0.01, 0.99)
        # High correlation with soft
        soft_df[f] = (oof_df[f] > 0.45).astype(int)

    macro, per_c = evaluate_auc_per_label(oof_df, gold_df)
    assert 0.5 <= macro <= 1.0
    assert len(per_c) == len(FINDINGS)

    # Bootstrapping
    boot_stats = bootstrap_leaderboard_auc(oof_df, gold_df, n_bootstraps=50)
    assert "mean_lb_auc" in boot_stats
    assert "ci_95_low" in boot_stats
    assert "ci_95_high" in boot_stats
    assert boot_stats["ci_95_low"] <= boot_stats["mean_lb_auc"] <= boot_stats["ci_95_high"]

    # Full simulation report
    report = simulate_and_diagnose(oof_df, gold_df, soft_df, n_bootstraps=50)
    assert "gold_macro_auc" in report
    assert "soft_macro_auc" in report
    assert "memorization_gap" in report
    assert "overfitting_warning" in report
    assert isinstance(report["overfitting_warning"], (bool, np.bool_))


if __name__ == "__main__":
    test_evaluate_and_simulation()

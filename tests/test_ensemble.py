"""
Unit tests for Phase 5A: src/ensemble.py
"""

import numpy as np
import pandas as pd
from src.ensemble import (
    FINDINGS,
    optimize_label_weights,
    optimize_multilabel_weights,
    blend_predictions,
    evaluate_ensemble_oof,
)


def test_optimize_label_weights():
    np.random.seed(42)
    n = 60
    # True binary targets
    y_true = (np.random.rand(n) > 0.5).astype(int)

    # Model 1 is good (correlates with y_true)
    m1 = np.clip(y_true * 0.7 + 0.15 + np.random.randn(n) * 0.1, 0.01, 0.99)
    # Model 2 is random noise
    m2 = np.random.uniform(0.1, 0.9, size=n)

    weights = optimize_label_weights([m1, m2], y_true, method="SLSQP", loss="bce")
    assert len(weights) == 2
    assert np.isclose(np.sum(weights), 1.0, atol=1e-4)
    assert weights[0] > weights[1], f"Better model should get higher weight: {weights}"


def test_optimize_multilabel_weights_and_blend():
    np.random.seed(42)
    studies = [f"study_{i:03d}" for i in range(58)]
    gold_data = {"StudyInstanceUID": studies}
    for f in FINDINGS:
        gold_data[f] = (np.random.rand(58) > 0.6).astype(int)
    gold_df = pd.DataFrame(gold_data)

    # Synthetic OOF predictions for 2 models
    oof_m1 = {"StudyInstanceUID": studies}
    oof_m2 = {"StudyInstanceUID": studies}
    for f in FINDINGS:
        oof_m1[f] = np.clip(gold_df[f].values * 0.6 + 0.2 + np.random.randn(58) * 0.05, 0.01, 0.99)
        oof_m2[f] = np.random.uniform(0.1, 0.9, size=58)

    m1_df = pd.DataFrame(oof_m1)
    m2_df = pd.DataFrame(oof_m2)

    oof_dfs = {"m1": m1_df, "m2": m2_df}
    opt_weights = optimize_multilabel_weights(oof_dfs, gold_df, loss="bce")

    assert len(opt_weights) == len(FINDINGS)
    for f in FINDINGS:
        w = opt_weights[f]
        assert "m1" in w and "m2" in w
        assert np.isclose(w["m1"] + w["m2"], 1.0, atol=1e-4)
        assert w["m1"] >= w["m2"]

    # Blend
    blended = blend_predictions(oof_dfs, opt_weights, rank_blend=False)
    assert len(blended) == 58
    assert "StudyInstanceUID" in blended.columns
    for f in FINDINGS:
        assert f in blended.columns
        assert (blended[f] >= 0.0).all() and (blended[f] <= 1.0).all()

    # Rank blend
    blended_rank = blend_predictions(oof_dfs, opt_weights, rank_blend=True)
    for f in FINDINGS:
        assert (blended_rank[f] >= 0.0).all() and (blended_rank[f] <= 1.0).all()

    # Evaluation
    eval_res = evaluate_ensemble_oof(blended, gold_df)
    assert "macro_auc" in eval_res
    assert eval_res["macro_auc"] > 0.6
    assert eval_res["num_studies"] == 58


if __name__ == "__main__":
    test_optimize_label_weights()
    test_optimize_multilabel_weights_and_blend()

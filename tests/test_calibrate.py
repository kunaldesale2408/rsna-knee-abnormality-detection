"""
Unit tests for Phase 5B: src/calibrate.py
"""

import numpy as np
import pandas as pd
from scipy.stats import spearmanr
from sklearn.metrics import roc_auc_score

from src.calibrate import (
    FINDINGS,
    rank_calibrate_array,
    fit_temperature_for_label,
    fit_temperature_scaling,
    apply_temperature_scaling,
    calibrate_predictions,
)


def test_rank_calibration_preserves_auc():
    np.random.seed(42)
    n = 100
    y_true = (np.random.rand(n) > 0.5).astype(int)
    raw_preds = y_true * 0.5 + np.random.uniform(0.0, 0.5, size=n)

    raw_auc = roc_auc_score(y_true, raw_preds)
    cal_preds = rank_calibrate_array(raw_preds)
    cal_auc = roc_auc_score(y_true, cal_preds)

    assert np.isclose(raw_auc, cal_auc, atol=1e-6), "Rank calibration must preserve exact ROC-AUC"
    assert cal_preds.min() >= 0.0 and cal_preds.max() <= 1.0
    # Spearman rank correlation should be 1.0
    corr, _ = spearmanr(raw_preds, cal_preds)
    assert np.isclose(corr, 1.0, atol=1e-5)


def test_temperature_scaling():
    np.random.seed(42)
    n = 100
    y_true = (np.random.rand(n) > 0.5).astype(int)
    # Over-confident predictions
    raw_preds = np.where(y_true == 1, 0.99, 0.01)

    t_opt = fit_temperature_for_label(raw_preds, y_true)
    assert t_opt > 0.0

    studies = [f"s_{i}" for i in range(50)]
    df_pred = pd.DataFrame({"StudyInstanceUID": studies})
    df_gold = pd.DataFrame({"StudyInstanceUID": studies})

    for f in FINDINGS:
        df_gold[f] = (np.random.rand(50) > 0.5).astype(int)
        df_pred[f] = np.clip(df_gold[f] * 0.8 + 0.1, 0.01, 0.99)

    temps = fit_temperature_scaling(df_pred, df_gold)
    assert len(temps) == len(FINDINGS)

    scaled_df = apply_temperature_scaling(df_pred, temps)
    for f in FINDINGS:
        assert (scaled_df[f] >= 0.0).all() and (scaled_df[f] <= 1.0).all()


def test_calibrate_predictions_wrapper():
    studies = [f"s_{i}" for i in range(30)]
    df = pd.DataFrame({"StudyInstanceUID": studies})
    for f in FINDINGS:
        df[f] = np.random.uniform(0.1, 0.9, size=30)

    cal_rank = calibrate_predictions(df, method="rank")
    assert len(cal_rank) == 30
    for f in FINDINGS:
        assert (cal_rank[f] >= 0.0).all() and (cal_rank[f] <= 1.0).all()


if __name__ == "__main__":
    test_rank_calibration_preserves_auc()
    test_temperature_scaling()
    test_calibrate_predictions_wrapper()

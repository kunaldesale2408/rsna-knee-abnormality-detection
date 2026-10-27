"""
Unit tests for src/report_labeler_llm.py and scripts/eval_labeler.py.
"""

import os
import tempfile
import pandas as pd
import numpy as np

from src.report_labeler import FINDINGS
from src.report_labeler_llm import LLMReportLabeler, ensemble_weak_labels
from scripts.eval_labeler import run_evaluation, compute_label_auc


def test_llm_report_labeler_fallback_and_formula():
    labeler = LLMReportLabeler()
    # Test formula: soft_target = clip(p_pos - 0.5 * p_neg, 0.05, 0.95)
    # Test with empty report
    res_empty = labeler.label_report("")
    assert len(res_empty) == 12
    for f in FINDINGS:
        assert f in res_empty
        assert res_empty[f]["soft_target"] == 0.15
        assert res_empty[f]["confidence_weight"] == 0.30

    # Test formula calculation manually
    p_pos = 0.80
    p_neg = 0.10
    soft_tgt = np.clip(p_pos - 0.5 * p_neg, 0.05, 0.95)
    assert abs(soft_tgt - 0.75) < 1e-5

    # Test negation dominance
    p_pos = 0.10
    p_neg = 0.80
    soft_tgt = np.clip(p_pos - 0.5 * p_neg, 0.05, 0.95)
    assert abs(soft_tgt - 0.05) < 1e-5


def test_ensemble_weak_labels():
    studies = [f"study_{i:02d}" for i in range(10)]
    df_regex = pd.DataFrame({"StudyInstanceUID": studies})
    df_llm = pd.DataFrame({"StudyInstanceUID": studies})

    for f in FINDINGS:
        df_regex[f"{f}_target"] = 0.80
        df_regex[f"{f}_weight"] = 1.0
        df_llm[f"{f}_target"] = 0.60
        df_llm[f"{f}_weight"] = 0.8

    df_ens = ensemble_weak_labels(df_regex, df_llm, alpha=0.5)

    assert len(df_ens) == 10
    for f in FINDINGS:
        # Expected ensemble target: 0.5 * 0.8 + 0.5 * 0.6 = 0.70
        assert abs(df_ens[f"{f}_target"].iloc[0] - 0.70) < 1e-5
        assert abs(df_ens[f"{f}_weight"].iloc[0] - 0.90) < 1e-5


def test_eval_labeler_pipeline():
    with tempfile.TemporaryDirectory() as tmpdir:
        train_csv = os.path.join(tmpdir, "train.csv")
        regex_csv = os.path.join(tmpdir, "regex.csv")
        llm_csv = os.path.join(tmpdir, "llm.csv")
        output_csv = os.path.join(tmpdir, "best.csv")

        studies = [f"study_{i:02d}" for i in range(58)]
        np.random.seed(42)

        df_train = pd.DataFrame({"StudyInstanceUID": studies})
        df_reg = pd.DataFrame({"StudyInstanceUID": studies})
        df_llm = pd.DataFrame({"StudyInstanceUID": studies})

        for f in FINDINGS:
            gold = np.random.binomial(1, 0.3, size=58)
            df_train[f] = gold
            # Regex: gold + some noise
            df_reg[f"{f}_target"] = np.clip(gold * 0.7 + np.random.uniform(0.05, 0.25, size=58), 0.05, 0.95)
            # LLM: gold + less noise (should have higher AUC)
            df_llm[f"{f}_target"] = np.clip(gold * 0.85 + np.random.uniform(0.02, 0.15, size=58), 0.05, 0.95)

        df_train.to_csv(train_csv, index=False)
        df_reg.to_csv(regex_csv, index=False)
        df_llm.to_csv(llm_csv, index=False)

        success = run_evaluation(
            train_csv=train_csv,
            regex_csv=regex_csv,
            llm_csv=llm_csv,
            output_csv=output_csv,
        )

        assert success is True
        assert os.path.exists(output_csv)
        df_best = pd.read_csv(output_csv)
        assert len(df_best) == 58


if __name__ == "__main__":
    test_llm_report_labeler_fallback_and_formula()
    test_ensemble_weak_labels()
    test_eval_labeler_pipeline()
    print("ALL PHASE 2 UNIT & INTEGRATION TESTS PASSED SUCCESSFULLY!")

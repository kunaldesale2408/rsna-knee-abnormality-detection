"""
scripts/leaderboard_sim.py
Phase 5D: Leaderboard Simulation & Overfitting Diagnostic.
Computes expected public LB macro AUC from OOF predictions, runs bootstrap confidence intervals,
and checks for signs of scanner memorization (soft-label vs gold-label gap).
"""

import argparse
import sys
from typing import Dict, List, Optional, Tuple, Any

import numpy as np
import pandas as pd
from sklearn.metrics import roc_auc_score

FINDINGS = [
    'ACL',
    'MCL',
    'Medial Meniscus',
    'Lateral Meniscus',
    'Medial OA',
    'Lateral OA',
    'PF OA',
    'Effusion',
    'Synovitis',
    "Baker's",
    'Contusion',
    'Fracture',
]


def evaluate_auc_per_label(
    df_preds: pd.DataFrame,
    df_targets: pd.DataFrame,
    findings: List[str] = FINDINGS,
    id_col: str = "StudyInstanceUID",
    threshold: float = 0.5,
) -> Tuple[float, Dict[str, float]]:
    """
    Compute macro AUC and per-class AUC between predictions and ground truth.
    """
    common_ids = sorted(list(set(df_preds[id_col].astype(str)).intersection(set(df_targets[id_col].astype(str)))))
    if not common_ids:
        return 0.5, {f: 0.5 for f in findings}

    p = df_preds.set_index(id_col).loc[common_ids]
    t = df_targets.set_index(id_col).loc[common_ids]

    aucs = {}
    valid = []
    for f in findings:
        if f in p.columns and f in t.columns:
            y_true = (t[f].to_numpy(dtype=float) >= threshold).astype(int)
            y_pred = p[f].to_numpy(dtype=float)
            if len(np.unique(y_true)) > 1:
                val = float(roc_auc_score(y_true, y_pred))
                aucs[f] = val
                valid.append(val)
            else:
                aucs[f] = 0.5
        else:
            aucs[f] = 0.5

    macro = float(np.mean(valid)) if valid else 0.5
    return macro, aucs


def bootstrap_leaderboard_auc(
    df_preds: pd.DataFrame,
    df_gold: pd.DataFrame,
    n_bootstraps: int = 1000,
    sample_ratio: float = 0.35,  # Typical Kaggle public test proportion
    findings: List[str] = FINDINGS,
    id_col: str = "StudyInstanceUID",
    random_state: int = 42,
) -> Dict[str, float]:
    """
    Simulate public leaderboard variation via bootstrapping over gold annotations.
    """
    common_ids = sorted(list(set(df_preds[id_col].astype(str)).intersection(set(df_gold[id_col].astype(str)))))
    n_samples = max(10, int(len(common_ids) * sample_ratio))

    rng = np.random.RandomState(random_state)
    boot_macros = []

    p_indexed = df_preds.set_index(id_col).loc[common_ids]
    g_indexed = df_gold.set_index(id_col).loc[common_ids]

    for _ in range(n_bootstraps):
        sampled_ids = rng.choice(common_ids, size=n_samples, replace=True)
        sub_p = p_indexed.loc[sampled_ids]
        sub_g = g_indexed.loc[sampled_ids]

        sub_aucs = []
        for f in findings:
            if f in sub_p.columns and f in sub_g.columns:
                yt = (sub_g[f].to_numpy(dtype=float) >= 0.5).astype(int)
                yp = sub_p[f].to_numpy(dtype=float)
                if len(np.unique(yt)) > 1:
                    sub_aucs.append(roc_auc_score(yt, yp))
        if sub_aucs:
            boot_macros.append(float(np.mean(sub_aucs)))

    boot_macros = np.array(boot_macros) if boot_macros else np.array([0.5])
    return {
        "mean_lb_auc": float(np.mean(boot_macros)),
        "std_lb_auc": float(np.std(boot_macros)),
        "ci_95_low": float(np.percentile(boot_macros, 2.5)),
        "ci_95_high": float(np.percentile(boot_macros, 97.5)),
    }


def simulate_and_diagnose(
    oof_preds_df: pd.DataFrame,
    gold_df: pd.DataFrame,
    soft_df: Optional[pd.DataFrame] = None,
    n_bootstraps: int = 1000,
    findings: List[str] = FINDINGS,
    id_col: str = "StudyInstanceUID",
) -> Dict[str, Any]:
    """
    Run full diagnostic simulation comparing gold-only AUC and soft-label AUC.
    """
    gold_macro, gold_class = evaluate_auc_per_label(oof_preds_df, gold_df, findings, id_col)
    soft_macro, soft_class = (0.0, {})
    has_soft = soft_df is not None

    if has_soft:
        soft_macro, soft_class = evaluate_auc_per_label(oof_preds_df, soft_df, findings, id_col)

    boot_stats = bootstrap_leaderboard_auc(
        oof_preds_df, gold_df, n_bootstraps=n_bootstraps, findings=findings, id_col=id_col
    )

    memorization_gap = (soft_macro - gold_macro) if has_soft else 0.0
    overfitting_warning = memorization_gap > 0.05

    report = {
        "gold_macro_auc": gold_macro,
        "soft_macro_auc": soft_macro if has_soft else None,
        "memorization_gap": memorization_gap if has_soft else None,
        "overfitting_warning": overfitting_warning,
        "expected_lb_mean": boot_stats["mean_lb_auc"],
        "expected_lb_std": boot_stats["std_lb_auc"],
        "expected_lb_ci": (boot_stats["ci_95_low"], boot_stats["ci_95_high"]),
        "per_class_gold": gold_class,
        "per_class_soft": soft_class if has_soft else None,
    }

    return report


def parse_args():
    parser = argparse.ArgumentParser(description="Phase 5D: Leaderboard Simulation & Scanner Memorization Check")
    parser.add_argument("--oof-file", type=str, required=True, help="Path to OOF predictions CSV")
    parser.add_argument("--gold-file", type=str, required=True, help="Path to gold labels CSV (58 studies)")
    parser.add_argument("--soft-file", type=str, default=None, help="Optional full dataset soft labels CSV")
    parser.add_argument("--n-bootstraps", type=int, default=1000, help="Number of bootstrap resamples")
    return parser.parse_args()


def main():
    args = parse_args()
    oof_df = pd.read_csv(args.oof_file)
    gold_df = pd.read_csv(args.gold_file)
    soft_df = pd.read_csv(args.soft_file) if args.soft_file else None

    print(f"[+] Loaded OOF predictions: {args.oof_file} ({len(oof_df)} rows)")
    print(f"[+] Loaded Gold labels: {args.gold_file} ({len(gold_df)} rows)")
    if soft_df is not None:
        print(f"[+] Loaded Soft labels: {args.soft_file} ({len(soft_df)} rows)")

    res = simulate_and_diagnose(oof_df, gold_df, soft_df, n_bootstraps=args.n_bootstraps)

    print("\n" + "=" * 65)
    print("           RSNA KNEE LEADERBOARD SIMULATION REPORT           ")
    print("=" * 65)
    print(f"  Gold-58 Macro ROC-AUC:         {res['gold_macro_auc']:.4f}")
    if res['soft_macro_auc'] is not None:
        print(f"  Soft-Label Macro ROC-AUC:      {res['soft_macro_auc']:.4f}")
        print(f"  Gap (Soft - Gold):             {res['memorization_gap']:+.4f}")
        if res['overfitting_warning']:
            print("  [!] WARNING: Gap > 0.05! Evidence of scanner memorization / soft label overfitting.")
        else:
            print("  [OK] Gap <= 0.05. Model generalizes stably between soft and gold distributions.")

    print(f"\n  Expected Public LB Mean:       {res['expected_lb_mean']:.4f}")
    print(f"  Expected Public LB Std:        {res['expected_lb_std']:.4f}")
    print(f"  95% Confidence Interval:       [{res['expected_lb_ci'][0]:.4f}, {res['expected_lb_ci'][1]:.4f}]")
    print("=" * 65)

    print("\nPer-Finding Gold ROC-AUC Breakdown:")
    for f in FINDINGS:
        g_auc = res['per_class_gold'].get(f, 0.5)
        s_str = f" | Soft: {res['per_class_soft'].get(f, 0.5):.4f}" if res['per_class_soft'] else ""
        print(f"  {f:18s}: {g_auc:.4f}{s_str}")


if __name__ == "__main__":
    main()

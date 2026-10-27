"""
RSNA Knee Abnormality Detection — Labeler Evaluation Script (Phase 2).

Compares Regex Labeler vs LLM (mDeBERTa) Labeler vs Ensemble on the 58 gold studies.
Target: LLM labeler should beat regex labeler by >= +0.03 macro AUC.
If it doesn't, automatically falls back to ensemble: 0.5 * regex_soft + 0.5 * llm_soft.

Usage:
    python scripts/eval_labeler.py [--train_csv data/train.csv] [--regex_csv results/weak_labels_v0.csv] [--llm_csv results/weak_labels_llm.csv] [--output_csv results/weak_labels_best.csv]
"""

import os
import sys
import argparse
from typing import Dict, List, Optional
import pandas as pd
import numpy as np
from sklearn.metrics import roc_auc_score

# Ensure root directory is on sys.path
REPO_ROOT = os.path.abspath(os.path.join(os.path.dirname(__file__), ".."))
if REPO_ROOT not in sys.path:
    sys.path.insert(0, REPO_ROOT)

from src.report_labeler import FINDINGS
from src.report_labeler_llm import ensemble_weak_labels

if hasattr(sys.stdout, "reconfigure"):
    try:
        sys.stdout.reconfigure(encoding="utf-8")
    except Exception:
        pass


def parse_args():
    parser = argparse.ArgumentParser(description="Evaluate weak labelers against gold labels.")
    parser.add_argument(
        "--train_csv",
        type=str,
        default=os.path.join(REPO_ROOT, "data", "train.csv"),
        help="Path to train.csv with gold labels",
    )
    parser.add_argument(
        "--regex_csv",
        type=str,
        default=os.path.join(REPO_ROOT, "results", "weak_labels_v0.csv"),
        help="Path to regex weak labels CSV",
    )
    parser.add_argument(
        "--llm_csv",
        type=str,
        default=os.path.join(REPO_ROOT, "results", "weak_labels_llm.csv"),
        help="Path to LLM weak labels CSV",
    )
    parser.add_argument(
        "--output_csv",
        type=str,
        default=os.path.join(REPO_ROOT, "results", "weak_labels_best.csv"),
        help="Path to export best/selected weak labels",
    )
    parser.add_argument(
        "--study_id_col",
        type=str,
        default="StudyInstanceUID",
        help="Study instance UID column name",
    )
    return parser.parse_args()


def compute_label_auc(y_true: np.ndarray, y_score: np.ndarray) -> Optional[float]:
    """Compute binary ROC-AUC, returning None if only single class present."""
    unique_vals = np.unique(y_true[~np.isnan(y_true)])
    if len(unique_vals) < 2:
        return None
    try:
        return float(roc_auc_score(y_true, y_score))
    except Exception:
        return None


def run_evaluation(
    train_csv: str,
    regex_csv: str,
    llm_csv: str,
    output_csv: str,
    study_id_col: str = "StudyInstanceUID",
) -> bool:
    print("=" * 75)
    print(" RSNA KNEE — WEAK LABELER EVALUATION (REGEX vs LLM vs ENSEMBLE)")
    print("=" * 75)

    if not os.path.exists(train_csv):
        print(f"[-] ERROR: train_csv not found at: {train_csv}")
        return False
    if not os.path.exists(regex_csv):
        print(f"[-] ERROR: regex_csv not found at: {regex_csv}")
        return False
    if not os.path.exists(llm_csv):
        print(f"[-] ERROR: llm_csv not found at: {llm_csv}")
        print("    Run the LLM labeler first to produce results/weak_labels_llm.csv")
        return False

    df_train = pd.read_csv(train_csv)
    df_regex = pd.read_csv(regex_csv)
    df_llm = pd.read_csv(llm_csv)

    # Filter for gold studies (studies where finding columns are populated)
    gold_cols = [f for f in FINDINGS if f in df_train.columns]
    if not gold_cols:
        print("[-] ERROR: No finding columns matching the 12 findings found in train.csv")
        return False

    gold_mask = df_train[gold_cols].notna().any(axis=1)
    df_gold = df_train[gold_mask].copy()
    print(f"[+] Found {len(df_gold)} gold-label studies for evaluation (expected: ~58).")

    # Merge gold labels with regex and LLM predictions
    merged_regex = pd.merge(df_gold, df_regex, on=study_id_col, suffixes=("_gold", "_regex"))
    merged_llm = pd.merge(df_gold, df_llm, on=study_id_col, suffixes=("_gold", "_llm"))

    # Also compute ensemble
    df_ensemble = ensemble_weak_labels(df_regex, df_llm, alpha=0.5, study_id_col=study_id_col)
    merged_ens = pd.merge(df_gold, df_ensemble, on=study_id_col, suffixes=("_gold", "_ens"))

    regex_aucs: Dict[str, Optional[float]] = {}
    llm_aucs: Dict[str, Optional[float]] = {}
    ens_aucs: Dict[str, Optional[float]] = {}

    print("\n" + "-" * 75)
    print(f"{'Finding':20s} {'Regex AUC':>12s} {'LLM AUC':>12s} {'Ensemble AUC':>14s} {'Delta (LLM-Reg)':>15s}")
    print("-" * 75)

    for f in FINDINGS:
        gold_vals = merged_regex[f"{f}_gold"].values if f"{f}_gold" in merged_regex.columns else merged_regex[f].values
        
        # Regex predictions
        reg_col = f"{f}_target" if f"{f}_target" in merged_regex.columns else f"{f}_regex"
        r_auc = compute_label_auc(gold_vals, merged_regex[reg_col].values)
        regex_aucs[f] = r_auc

        # LLM predictions
        llm_col = f"{f}_target" if f"{f}_target" in merged_llm.columns else f"{f}_llm"
        l_auc = compute_label_auc(gold_vals, merged_llm[llm_col].values)
        llm_aucs[f] = l_auc

        # Ensemble predictions
        ens_col = f"{f}_target" if f"{f}_target" in merged_ens.columns else f"{f}_ens"
        e_auc = compute_label_auc(gold_vals, merged_ens[ens_col].values)
        ens_aucs[f] = e_auc

        r_str = f"{r_auc:.4f}" if r_auc is not None else "N/A"
        l_str = f"{l_auc:.4f}" if l_auc is not None else "N/A"
        e_str = f"{e_auc:.4f}" if e_auc is not None else "N/A"
        
        if r_auc is not None and l_auc is not None:
            delta = l_auc - r_auc
            delta_str = f"{delta:+.4f}"
        else:
            delta_str = "N/A"

        print(f"{f:20s} {r_str:>12s} {l_str:>12s} {e_str:>14s} {delta_str:>15s}")

    # Compute Macro AUCs
    valid_r = [v for v in regex_aucs.values() if v is not None]
    valid_l = [v for v in llm_aucs.values() if v is not None]
    valid_e = [v for v in ens_aucs.values() if v is not None]

    macro_regex = float(np.mean(valid_r)) if valid_r else 0.5
    macro_llm = float(np.mean(valid_l)) if valid_l else 0.5
    macro_ens = float(np.mean(valid_e)) if valid_e else 0.5
    overall_delta = macro_llm - macro_regex

    print("-" * 75)
    print(f"{'MACRO ROC-AUC':20s} {macro_regex:>12.4f} {macro_llm:>12.4f} {macro_ens:>14.4f} {overall_delta:>+15.4f}")
    print("=" * 75)

    # Decision Rule: LLM should beat regex by >= +0.03
    TARGET_MARGIN = 0.03
    os.makedirs(os.path.dirname(os.path.abspath(output_csv)), exist_ok=True)

    if overall_delta >= TARGET_MARGIN:
        print(f"\n[OK] SUCCESS: LLM Labeler beat Regex Labeler by {overall_delta:+.4f} (>= +{TARGET_MARGIN:.2f} target margin)!")
        print(f"    Exporting pure LLM weak labels to: {output_csv}")
        df_llm.to_csv(output_csv, index=False)
    else:
        print(f"\n[!] NOTICE: LLM improvement ({overall_delta:+.4f}) is below +{TARGET_MARGIN:.2f} threshold.")
        print(f"    Falling back to ENSEMBLE (0.5 * regex + 0.5 * llm) with Macro AUC: {macro_ens:.4f}")
        print(f"    Exporting calibrated ensemble weak labels to: {output_csv}")
        df_ensemble.to_csv(output_csv, index=False)

    print("[OK] Evaluation and export complete.")
    return True


def main():
    args = parse_args()
    success = run_evaluation(
        train_csv=args.train_csv,
        regex_csv=args.regex_csv,
        llm_csv=args.llm_csv,
        output_csv=args.output_csv,
        study_id_col=args.study_id_col,
    )
    if not success:
        sys.exit(1)


if __name__ == "__main__":
    main()

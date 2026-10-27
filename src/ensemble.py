"""
src/ensemble.py
Phase 5A: Multi-model per-label optimization and ensembling for RSNA Knee Abnormality Detection.
"""

import argparse
import os
import sys
from typing import Dict, List, Tuple, Optional, Any, Union

import numpy as np
import pandas as pd
from scipy.optimize import minimize
from scipy.stats import rankdata
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


def optimize_label_weights(
    preds_list: List[np.ndarray],
    gold_targets: np.ndarray,
    method: str = "SLSQP",
    loss: str = "bce",
) -> np.ndarray:
    """
    Find optimal non-negative simplex weights [w_1, ..., w_M] for a single finding label.
    
    Args:
        preds_list: List of 1D numpy arrays of shape (N,), each containing probabilities from one model.
        gold_targets: 1D numpy array of shape (N,) containing binary gold ground truth (0 or 1).
        method: Optimization algorithm ('SLSQP', 'Nelder-Mead', 'Powell').
        loss: Objective loss ('bce', 'auc', or 'brier').
        
    Returns:
        1D numpy array of length M containing optimal normalized weights summing to 1.
    """
    m = len(preds_list)
    if m == 1:
        return np.array([1.0], dtype=np.float64)

    # Validate target diversity
    unique_targets = np.unique(gold_targets)
    if len(unique_targets) < 2:
        return np.full(m, 1.0 / m, dtype=np.float64)

    p_matrix = np.column_stack(preds_list)  # (N, M)
    eps = 1e-7

    def objective(w: np.ndarray) -> float:
        w_norm = np.clip(w, 0.0, None)
        s = np.sum(w_norm)
        if s <= eps:
            w_norm = np.full(m, 1.0 / m)
        else:
            w_norm = w_norm / s

        pred_ens = np.clip(np.dot(p_matrix, w_norm), eps, 1.0 - eps)

        if loss == "bce":
            bce = -(gold_targets * np.log(pred_ens) + (1.0 - gold_targets) * np.log(1.0 - pred_ens))
            return float(np.mean(bce))
        elif loss == "brier":
            return float(np.mean((pred_ens - gold_targets) ** 2))
        elif loss == "auc":
            try:
                # RankNet smooth approximation or negative AUC
                auc = roc_auc_score(gold_targets, pred_ens)
                return -float(auc)
            except ValueError:
                return 0.0
        else:
            raise ValueError(f"Unknown loss: {loss}")

    init_w = np.full(m, 1.0 / m, dtype=np.float64)
    bounds = [(0.0, 1.0) for _ in range(m)]

    if method.upper() == "SLSQP":
        constraints = {"type": "eq", "fun": lambda w: np.sum(w) - 1.0}
        res = minimize(objective, init_w, method="SLSQP", bounds=bounds, constraints=constraints)
    else:
        res = minimize(objective, init_w, method=method, bounds=bounds)

    opt_w = np.clip(res.x, 0.0, None)
    s = np.sum(opt_w)
    if s > eps:
        opt_w = opt_w / s
    else:
        opt_w = np.full(m, 1.0 / m, dtype=np.float64)

    return opt_w


def optimize_multilabel_weights(
    oof_dfs: Dict[str, pd.DataFrame],
    gold_df: pd.DataFrame,
    findings: List[str] = FINDINGS,
    id_col: str = "StudyInstanceUID",
    loss: str = "bce",
) -> Dict[str, Dict[str, float]]:
    """
    Optimize weights for all 12 findings independently across multiple model OOF predictions.
    
    Args:
        oof_dfs: Dict mapping model_name -> DataFrame with id_col and finding columns.
        gold_df: DataFrame with id_col and true binary finding columns.
        findings: List of finding column names.
        id_col: Column name identifying the study.
        loss: Objective loss for optimization.
        
    Returns:
        Dict mapping finding -> {model_name: weight}.
    """
    model_names = list(oof_dfs.keys())
    assert len(model_names) > 0, "At least one OOF model dataframe is required."

    # Identify common gold studies
    common_ids = set(gold_df[id_col].astype(str))
    for name, df in oof_dfs.items():
        common_ids = common_ids.intersection(set(df[id_col].astype(str)))

    if len(common_ids) == 0:
        raise ValueError("No common StudyInstanceUIDs found between OOF predictions and gold ground truth.")

    id_list = sorted(list(common_ids))
    gold_aligned = gold_df.set_index(id_col).loc[id_list]
    oof_aligned = {name: df.set_index(id_col).loc[id_list] for name, df in oof_dfs.items()}

    optimal_weights: Dict[str, Dict[str, float]] = {}

    for f in findings:
        if f not in gold_aligned.columns:
            continue
        targets = gold_aligned[f].to_numpy(dtype=np.float64)
        preds_list = [oof_aligned[m_name][f].to_numpy(dtype=np.float64) for m_name in model_names]

        w = optimize_label_weights(preds_list, targets, method="SLSQP", loss=loss)
        optimal_weights[f] = {m_name: float(w[i]) for i, m_name in enumerate(model_names)}

    return optimal_weights


def blend_predictions(
    model_preds: Dict[str, pd.DataFrame],
    weights: Dict[str, Dict[str, float]],
    findings: List[str] = FINDINGS,
    id_col: str = "StudyInstanceUID",
    rank_blend: bool = False,
) -> pd.DataFrame:
    """
    Apply finding-specific weights to blend predictions across models.
    
    Args:
        model_preds: Dict mapping model_name -> DataFrame with id_col and finding columns.
        weights: Dict mapping finding -> {model_name: weight}.
        findings: List of finding columns to blend.
        id_col: Study identifier column.
        rank_blend: If True, normalize model predictions to percentile ranks before weighting.
        
    Returns:
        DataFrame with id_col and blended finding probabilities.
    """
    first_df = next(iter(model_preds.values()))
    blended = pd.DataFrame({id_col: first_df[id_col].values})

    n = len(first_df)
    for f in findings:
        f_weights = weights.get(f, {m: 1.0 / len(model_preds) for m in model_preds})
        accum = np.zeros(n, dtype=np.float64)

        for m_name, df in model_preds.items():
            w = f_weights.get(m_name, 0.0)
            if w <= 1e-9:
                continue
            vals = df[f].to_numpy(dtype=np.float64)
            if rank_blend:
                # Transform to percentile rank in [0, 1]
                vals = (rankdata(vals, method="average") - 0.5) / n
            accum += w * vals

        blended[f] = accum

    return blended


def evaluate_ensemble_oof(
    blended_df: pd.DataFrame,
    gold_df: pd.DataFrame,
    findings: List[str] = FINDINGS,
    id_col: str = "StudyInstanceUID",
) -> Dict[str, Any]:
    """
    Compute macro AUC and per-class AUC of an ensemble DataFrame vs Gold ground truth.
    """
    common_ids = sorted(list(set(blended_df[id_col].astype(str)).intersection(set(gold_df[id_col].astype(str)))))
    b_aligned = blended_df.set_index(id_col).loc[common_ids]
    g_aligned = gold_df.set_index(id_col).loc[common_ids]

    aucs = {}
    valid_aucs = []
    for f in findings:
        if f in g_aligned.columns and f in b_aligned.columns:
            y_true = (g_aligned[f].to_numpy(dtype=float) >= 0.5).astype(int)
            y_pred = b_aligned[f].to_numpy(dtype=float)
            if len(np.unique(y_true)) > 1:
                score = float(roc_auc_score(y_true, y_pred))
                aucs[f] = score
                valid_aucs.append(score)
            else:
                aucs[f] = 0.5

    macro_auc = float(np.mean(valid_aucs)) if valid_aucs else 0.5
    return {
        "macro_auc": macro_auc,
        "num_studies": len(common_ids),
        "per_class_auc": aucs,
    }


def parse_args():
    parser = argparse.ArgumentParser(description="Phase 5A: Multi-model Ensembling & Weight Optimization")
    parser.add_argument("--oof-files", nargs="+", required=True, help="List of model OOF CSV files (format: name=path or path)")
    parser.add_argument("--gold-file", type=str, required=True, help="Path to gold labels CSV (58 studies)")
    parser.add_argument("--test-files", nargs="*", default=None, help="Optional test prediction CSV files matching oof models")
    parser.add_argument("--output-csv", type=str, default="final_submission.csv", help="Output path for submission/blended CSV")
    parser.add_argument("--rank-blend", action="store_true", help="Perform percentile rank blending instead of raw probability blend")
    parser.add_argument("--loss", type=str, default="bce", choices=["bce", "brier", "auc"], help="Optimization loss metric")
    return parser.parse_args()


def main():
    args = parse_args()
    oof_dfs: Dict[str, pd.DataFrame] = {}
    for item in args.oof_files:
        if "=" in item:
            name, path = item.split("=", 1)
        else:
            name = os.path.splitext(os.path.basename(item))[0]
            path = item
        oof_dfs[name] = pd.read_csv(path)
        print(f"[+] Loaded OOF predictions: '{name}' from {path} ({len(oof_dfs[name])} rows)")

    gold_df = pd.read_csv(args.gold_file)
    print(f"[+] Loaded Gold labels from {args.gold_file} ({len(gold_df)} rows)")

    # Optimize weights
    weights = optimize_multilabel_weights(oof_dfs, gold_df, loss=args.loss)
    print("\n[+] OPTIMAL ENSEMBLE WEIGHTS:")
    for f in FINDINGS:
        w_str = ", ".join([f"{m}: {w:.3f}" for m, w in weights.get(f, {}).items()])
        print(f"  {f:18s} -> {w_str}")

    # Evaluate OOF
    blended_oof = blend_predictions(oof_dfs, weights, rank_blend=args.rank_blend)
    eval_res = evaluate_ensemble_oof(blended_oof, gold_df)
    print(f"\n[+] OOF Gold Macro AUC: {eval_res['macro_auc']:.4f} across {eval_res['num_studies']} studies")

    # Blend test files if provided
    if args.test_files:
        test_dfs: Dict[str, pd.DataFrame] = {}
        for item in args.test_files:
            if "=" in item:
                name, path = item.split("=", 1)
            else:
                name = os.path.splitext(os.path.basename(item))[0]
                path = item
            test_dfs[name] = pd.read_csv(path)
        final_test = blend_predictions(test_dfs, weights, rank_blend=args.rank_blend)
        final_test.to_csv(args.output_csv, index=False)
        print(f"[+] Successfully wrote final blended test submission to {args.output_csv}")
    else:
        blended_oof.to_csv(args.output_csv, index=False)
        print(f"[+] Wrote blended OOF predictions to {args.output_csv}")


if __name__ == "__main__":
    main()

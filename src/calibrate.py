"""
src/calibrate.py
Phase 5B: Post-processing calibration routines for RSNA Knee Abnormality Detection.
Includes:
- Rank-based calibration (uniform rank projection [0, 1], rank-invariant for ROC-AUC)
- Temperature scaling (optimizing T per label to minimize NLL on gold OOF)
"""

import argparse
import os
from typing import Dict, List, Optional, Tuple, Any, Union

import numpy as np
import pandas as pd
from scipy.optimize import minimize_scalar
from scipy.stats import rankdata
from sklearn.metrics import log_loss, roc_auc_score

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


def logit(p: np.ndarray, eps: float = 1e-7) -> np.ndarray:
    """Compute numerically stable logit: log(p / (1 - p))."""
    p_c = np.clip(p, eps, 1.0 - eps)
    return np.log(p_c / (1.0 - p_c))


def sigmoid(z: np.ndarray) -> np.ndarray:
    """Compute numerically stable sigmoid: 1 / (1 + exp(-z))."""
    return 1.0 / (1.0 + np.exp(-np.clip(z, -30.0, 30.0)))


def rank_calibrate_array(preds: np.ndarray) -> np.ndarray:
    """
    Transform 1D prediction array to uniform [0, 1] using average rank.
    Formula: (rank - 0.5) / N
    Preserves exact Spearman correlation and pairwise order (ROC-AUC invariant).
    """
    n = len(preds)
    if n <= 1:
        return np.copy(preds)
    ranks = rankdata(preds, method="average")
    return (ranks - 0.5) / n


def fit_temperature_for_label(
    preds: np.ndarray,
    targets: np.ndarray,
    bounds: Tuple[float, float] = (0.1, 10.0),
) -> float:
    """
    Find scalar temperature T > 0 that minimizes Binary Cross-Entropy / NLL on ground truth.
    
    Args:
        preds: 1D probability predictions in [0, 1].
        targets: 1D binary ground truth targets (0 or 1).
        bounds: Range of valid temperatures (T_min, T_max).
        
    Returns:
        Optimal temperature T.
    """
    unique = np.unique(targets)
    if len(unique) < 2:
        return 1.0

    z = logit(preds)
    eps = 1e-7

    def loss(t: float) -> float:
        if t <= 0.0:
            return 1e9
        cal_p = np.clip(sigmoid(z / t), eps, 1.0 - eps)
        nll = -(targets * np.log(cal_p) + (1.0 - targets) * np.log(1.0 - cal_p))
        return float(np.mean(nll))

    res = minimize_scalar(loss, bounds=bounds, method="bounded")
    return float(res.x)


def fit_temperature_scaling(
    oof_df: pd.DataFrame,
    gold_df: pd.DataFrame,
    findings: List[str] = FINDINGS,
    id_col: str = "StudyInstanceUID",
) -> Dict[str, float]:
    """
    Fit optimal temperature per label using OOF predictions aligned against gold studies.
    """
    common_ids = sorted(list(set(oof_df[id_col].astype(str)).intersection(set(gold_df[id_col].astype(str)))))
    if not common_ids:
        return {f: 1.0 for f in findings}

    oof_sub = oof_df.set_index(id_col).loc[common_ids]
    gold_sub = gold_df.set_index(id_col).loc[common_ids]

    temperatures: Dict[str, float] = {}
    for f in findings:
        if f in oof_sub.columns and f in gold_sub.columns:
            preds = oof_sub[f].to_numpy(dtype=float)
            targets = (gold_sub[f].to_numpy(dtype=float) >= 0.5).astype(int)
            temperatures[f] = fit_temperature_for_label(preds, targets)
        else:
            temperatures[f] = 1.0

    return temperatures


def apply_temperature_scaling(
    df: pd.DataFrame,
    temperatures: Dict[str, float],
    findings: List[str] = FINDINGS,
    id_col: str = "StudyInstanceUID",
) -> pd.DataFrame:
    """
    Apply temperature scaling per label: p_cal = sigmoid(logit(p) / T).
    """
    out_df = df.copy()
    for f in findings:
        if f in out_df.columns:
            t = temperatures.get(f, 1.0)
            if abs(t - 1.0) > 1e-4 and t > 0:
                z = logit(out_df[f].to_numpy(dtype=float))
                out_df[f] = sigmoid(z / t)
    return out_df


def calibrate_predictions(
    df: pd.DataFrame,
    method: str = "rank",
    temperatures: Optional[Dict[str, float]] = None,
    findings: List[str] = FINDINGS,
    id_col: str = "StudyInstanceUID",
) -> pd.DataFrame:
    """
    Main calibration function supporting 'rank' and 'temperature' methods.
    
    Args:
        df: DataFrame containing predictions.
        method: 'rank' (uniform rank normalization) or 'temperature' (logit scaling).
        temperatures: Dict of per-label temperatures (required if method='temperature').
        findings: List of finding columns to calibrate.
        id_col: Identifier column.
        
    Returns:
        Calibrated DataFrame.
    """
    out_df = df.copy()
    if method == "rank":
        for f in findings:
            if f in out_df.columns:
                out_df[f] = rank_calibrate_array(out_df[f].to_numpy(dtype=float))
    elif method == "temperature":
        if temperatures is None:
            raise ValueError("temperatures dictionary must be provided for temperature calibration")
        out_df = apply_temperature_scaling(out_df, temperatures, findings=findings, id_col=id_col)
    else:
        raise ValueError(f"Unknown calibration method: {method}")

    return out_df


def parse_args():
    parser = argparse.ArgumentParser(description="Phase 5B: Prediction Calibration")
    parser.add_argument("--input-csv", type=str, required=True, help="Input predictions CSV to calibrate")
    parser.add_argument("--gold-csv", type=str, default=None, help="Optional gold labels CSV (for fitting temperatures)")
    parser.add_argument("--method", type=str, default="rank", choices=["rank", "temperature"], help="Calibration method")
    parser.add_argument("--output-csv", type=str, required=True, help="Path to write calibrated CSV")
    return parser.parse_args()


def main():
    args = parse_args()
    df = pd.read_csv(args.input_csv)
    print(f"[+] Loaded predictions from {args.input_csv} ({len(df)} rows)")

    if args.method == "rank":
        cal_df = calibrate_predictions(df, method="rank")
        print("[+] Applied rank-based calibration [0, 1] per finding.")
    elif args.method == "temperature":
        if not args.gold_csv:
            raise ValueError("--gold-csv is required to fit temperature scaling.")
        gold_df = pd.read_csv(args.gold_csv)
        temps = fit_temperature_scaling(df, gold_df)
        print("\n[+] FITTED TEMPERATURES:")
        for f, t in temps.items():
            print(f"  {f:18s} -> T = {t:.3f}")
        cal_df = calibrate_predictions(df, method="temperature", temperatures=temps)

    cal_df.to_csv(args.output_csv, index=False)
    print(f"[+] Calibrated predictions saved to {args.output_csv}")


if __name__ == "__main__":
    main()

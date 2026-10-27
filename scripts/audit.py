"""
RSNA Knee Abnormality Detection — Data Audit Script.

Usage:
    python scripts/audit.py [--train_csv data/train.csv] [--train_series_csv data/train_series.csv] [--output_csv results/weak_labels_v0.csv]
    python scripts/audit.py data/train.csv data/train_series.csv

Performs:
1. Reads train.csv and train_series.csv
2. Prints:
   - Total studies
   - Studies with gold labels (expected: 58)
   - Series count per plane (Sagittal / Coronal / Axial)
   - Slice count statistics
3. Validates:
   - All studies in train.csv have at least one Sagittal + Coronal + Axial series
4. Saves weak labels using report_labeler.py to results/weak_labels_v0.csv
"""

import os
import sys
import argparse
from typing import Optional, List, Dict, Any, Set
import pandas as pd
import numpy as np

# Ensure repository root is on sys.path
REPO_ROOT = os.path.abspath(os.path.join(os.path.dirname(__file__), ".."))
if REPO_ROOT not in sys.path:
    sys.path.insert(0, REPO_ROOT)

from src.report_labeler import ReportLabeler, FINDINGS


if hasattr(sys.stdout, "reconfigure"):
    try:
        sys.stdout.reconfigure(encoding="utf-8")
    except Exception:
        pass


def parse_args():
    parser = argparse.ArgumentParser(description="Audit RSNA Knee dataset and generate weak labels.")
    parser.add_argument(
        "pos_args",
        nargs="*",
        help="Optional positional arguments: [train_csv] [train_series_csv] [output_csv]",
    )
    parser.add_argument(
        "--train_csv",
        type=str,
        default=os.path.join(REPO_ROOT, "data", "train.csv"),
        help="Path to train.csv",
    )
    parser.add_argument(
        "--train_series_csv",
        type=str,
        default=os.path.join(REPO_ROOT, "data", "train_series.csv"),
        help="Path to train_series.csv",
    )
    parser.add_argument(
        "--output_csv",
        type=str,
        default=os.path.join(REPO_ROOT, "results", "weak_labels_v0.csv"),
        help="Destination path for weak labels CSV",
    )
    parser.add_argument(
        "--report_col",
        type=str,
        default=None,
        help="Report column name in train.csv (auto-detected if None)",
    )
    parser.add_argument(
        "--study_id_col",
        type=str,
        default=None,
        help="Study ID column name (auto-detected if None)",
    )
    return parser.parse_args()


def detect_column(df: pd.DataFrame, candidates: List[str]) -> Optional[str]:
    """Find the first matching column name from a candidate list (case-insensitive)."""
    df_cols_lower = {c.lower(): c for c in df.columns}
    for cand in candidates:
        if cand.lower() in df_cols_lower:
            return df_cols_lower[cand.lower()]
    return None


def run_audit(
    train_csv: str,
    train_series_csv: str,
    output_csv: str,
    report_col: Optional[str] = None,
    study_id_col: Optional[str] = None,
) -> bool:
    print("=" * 70)
    print(" RSNA KNEE ABNORMALITY DETECTION — DATA AUDIT")
    print("=" * 70)
    print(f"Train CSV        : {train_csv}")
    print(f"Train Series CSV : {train_series_csv}")
    print(f"Output CSV       : {output_csv}")
    print("-" * 70)

    # Check files exist
    if not os.path.exists(train_csv):
        print(f"[-] ERROR: train_csv not found at: {train_csv}")
        print("    Please place the competition train.csv in the data/ directory.")
        return False

    if not os.path.exists(train_series_csv):
        print(f"[-] ERROR: train_series_csv not found at: {train_series_csv}")
        print("    Please place the competition train_series.csv in the data/ directory.")
        return False

    # 1. Load CSVs
    print("[+] Loading CSVs...")
    df_train = pd.read_csv(train_csv)
    df_series = pd.read_csv(train_series_csv)

    print(f"    Loaded train.csv with {len(df_train):,} rows and {len(df_train.columns)} columns.")
    print(f"    Loaded train_series.csv with {len(df_series):,} rows and {len(df_series.columns)} columns.")

    # Detect Study UID column
    study_col = study_id_col or detect_column(df_train, ["StudyInstanceUID", "study_uid", "study_id"])
    if not study_col:
        study_col = df_train.columns[0]
        print(f"    [!] Defaulting to first column as study identifier: {study_col}")
    else:
        print(f"    Using study ID column: '{study_col}'")

    series_study_col = detect_column(df_series, ["StudyInstanceUID", "study_uid", "study_id"]) or study_col

    # 2. Total studies
    total_studies = df_train[study_col].nunique()
    print(f"\n[1] Total Studies: {total_studies:,}")

    # 3. Studies with gold labels
    # Gold labels are indicated either by a dedicated column or non-null/valid finding labels
    gold_indicators = [c for c in df_train.columns if c in FINDINGS]
    has_gold_col = detect_column(df_train, ["has_gold", "is_gold", "gold_label", "gold"])

    if has_gold_col:
        gold_count = int(df_train[has_gold_col].astype(bool).sum())
    elif gold_indicators:
        # Studies where any gold finding column is non-null and numeric (0 or 1)
        valid_mask = df_train[gold_indicators].notna().any(axis=1)
        # Check if values are binary 0/1
        gold_count = int(valid_mask.sum())
    else:
        gold_count = 0

    print(f"[2] Studies with Gold Labels: {gold_count} (expected: ~58)")

    # 4. Series count per plane
    plane_col = detect_column(df_series, ["Plane", "plane", "series_plane", "SeriesDescription"])
    if plane_col:
        print(f"\n[3] Series Count per Plane (column: '{plane_col}'):")
        plane_counts = df_series[plane_col].astype(str).str.strip().str.capitalize().value_counts()
        for pl, cnt in plane_counts.items():
            print(f"    - {pl:15s}: {cnt:,} series")
    else:
        print("\n[3] [!] Plane column not identified in train_series.csv")

    # 5. Slice count stats
    slice_col = detect_column(df_series, ["num_slices", "SliceCount", "slices", "count", "num_images"])
    if slice_col:
        slice_vals = pd.to_numeric(df_series[slice_col], errors="coerce").dropna()
        print(f"\n[4] Slice Count Statistics (column: '{slice_col}'):")
        print(f"    - Min    : {int(slice_vals.min())}")
        print(f"    - Max    : {int(slice_vals.max())}")
        print(f"    - Mean   : {slice_vals.mean():.2f}")
        print(f"    - Median : {slice_vals.median():.1f}")
        print(f"    - Std    : {slice_vals.std():.2f}")
    else:
        print("\n[4] [!] Slice count column not identified in train_series.csv")

    # 6. Validate: all studies in train.csv have at least one Sagittal + Coronal + Axial series
    print("\n[5] Multi-Plane Validation:")
    print("    Validating all studies have at least one Sagittal, Coronal, and Axial series...")

    missing_plane_report: Dict[str, List[str]] = {
        "Sagittal": [],
        "Coronal": [],
        "Axial": [],
    }
    studies_missing_any: Set[str] = set()

    if plane_col and series_study_col:
        # Group series by study and available planes
        df_series_planes = df_series[[series_study_col, plane_col]].copy()
        df_series_planes["norm_plane"] = df_series_planes[plane_col].astype(str).str.strip().str.capitalize()

        grouped_planes = df_series_planes.groupby(series_study_col)["norm_plane"].apply(set).to_dict()

        for s_id in df_train[study_col].unique():
            available = grouped_planes.get(s_id, set())
            for required_plane in ["Sagittal", "Coronal", "Axial"]:
                # Match required plane
                if not any(required_plane.lower() in p.lower() for p in available):
                    missing_plane_report[required_plane].append(str(s_id))
                    studies_missing_any.add(str(s_id))

        if not studies_missing_any:
            print("    [OK] SUCCESS: All studies contain at least one Sagittal + Coronal + Axial series!")
        else:
            print(f"    [!] WARNING: {len(studies_missing_any)} studies are missing one or more required planes:")
            for pl, miss in missing_plane_report.items():
                print(f"        - Missing {pl:8s}: {len(miss)} studies")
    else:
        print("    [!] Could not validate planes due to missing columns.")

    # 7. Extract weak labels using report_labeler.py
    print("\n[6] Generating Weak Labels using src/report_labeler.py...")
    rep_col = report_col or detect_column(df_train, ["report", "report_text", "text", "radiology_report", "impression"])
    if not rep_col:
        print("    [-] ERROR: Report text column not found in train.csv.")
        print(f"    Available columns: {list(df_train.columns)}")
        return False

    labeler = ReportLabeler()
    df_weak = labeler.label_dataframe(
        df_train,
        report_col=rep_col,
        study_id_col=study_col,
    )

    # Ensure output directory exists
    os.makedirs(os.path.dirname(os.path.abspath(output_csv)), exist_ok=True)
    df_weak.to_csv(output_csv, index=False)
    print(f"    [OK] Weak labels saved successfully to: {output_csv}")
    print(f"    Generated labels for {len(df_weak):,} studies across 12 findings.")

    # Print summary statistics of generated labels
    print("\n[7] Weak Labels Summary (Positive rate & Mean target):")
    print(f"    {'Finding':20s} {'Pos Rate':>10s} {'Mean Target':>12s} {'Mean Weight':>12s}")
    print("    " + "-" * 56)
    for f in FINDINGS:
        pos_rate = (df_weak[f"{f}_state"] == "explicit_positive").mean() * 100
        mean_tgt = df_weak[f"{f}_target"].mean()
        mean_wt = df_weak[f"{f}_weight"].mean()
        print(f"    {f:20s} {pos_rate:>9.1f}% {mean_tgt:>12.3f} {mean_wt:>12.3f}")

    print("\n" + "=" * 70)
    print(" AUDIT COMPLETE")
    print("=" * 70)
    return True


def main():
    args = parse_args()

    train_csv = args.train_csv
    train_series_csv = args.train_series_csv
    output_csv = args.output_csv

    # Support positional arguments if provided
    if args.pos_args:
        if len(args.pos_args) >= 1:
            train_csv = args.pos_args[0]
        if len(args.pos_args) >= 2:
            train_series_csv = args.pos_args[1]
        if len(args.pos_args) >= 3:
            output_csv = args.pos_args[2]

    success = run_audit(
        train_csv=train_csv,
        train_series_csv=train_series_csv,
        output_csv=output_csv,
        report_col=args.report_col,
        study_id_col=args.study_id_col,
    )
    if not success:
        sys.exit(1)


if __name__ == "__main__":
    main()

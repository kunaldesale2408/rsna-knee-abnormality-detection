"""
Integration test for scripts/audit.py.
"""

import os
import subprocess
import tempfile
import pandas as pd
import numpy as np
from src.report_labeler import FINDINGS


def test_audit_pipeline():
    with tempfile.TemporaryDirectory() as tmpdir:
        train_csv_path = os.path.join(tmpdir, "train.csv")
        train_series_path = os.path.join(tmpdir, "train_series.csv")
        output_csv_path = os.path.join(tmpdir, "weak_labels.csv")

        # Create 60 sample studies, 58 of which have gold labels
        studies = [f"study_{i:04d}" for i in range(60)]
        reports = [
            "Normal knee exam. No effusion or ligament tears.",
            "Complete tear of the anterior cruciate ligament. Trace effusion.",
            "Moderate medial meniscus tear and patellofemoral osteoarthritis.",
            "Gross joint effusion. Baker's cyst noted.",
            "Fracture line in tibial plateau with bone contusion.",
        ] * 12

        train_records = []
        for i, s in enumerate(studies):
            rec = {
                "StudyInstanceUID": s,
                "report": reports[i],
            }
            # Add gold labels for first 58
            for f in FINDINGS:
                rec[f] = 1 if (i < 58 and i % 3 == 0) else (0 if i < 58 else np.nan)
            train_records.append(rec)

        df_train = pd.DataFrame(train_records)
        df_train.to_csv(train_csv_path, index=False)

        # Create series: each study has Sagittal, Coronal, Axial
        series_records = []
        for s in studies:
            for plane in ["Sagittal", "Coronal", "Axial"]:
                series_records.append({
                    "StudyInstanceUID": s,
                    "SeriesInstanceUID": f"{s}_{plane}",
                    "Plane": plane,
                    "Fluid_Sensitive": 1,
                    "Fat_Suppression": 1,
                    "num_slices": 28,
                })

        df_series = pd.DataFrame(series_records)
        df_series.to_csv(train_series_path, index=False)

        # Execute scripts/audit.py via CLI
        cmd = [
            "python",
            "scripts/audit.py",
            "--train_csv",
            train_csv_path,
            "--train_series_csv",
            train_series_path,
            "--output_csv",
            output_csv_path,
        ]
        result = subprocess.run(cmd, capture_output=True, text=True)
        print("Audit stdout:\n", result.stdout)
        if result.stderr:
            print("Audit stderr:\n", result.stderr)

        assert result.returncode == 0, f"audit.py exited with code {result.returncode}"
        assert os.path.exists(output_csv_path), "Weak labels CSV was not created"

        # Verify weak labels CSV content
        df_weak = pd.read_csv(output_csv_path)
        assert len(df_weak) == 60
        assert "StudyInstanceUID" in df_weak.columns
        for f in FINDINGS:
            assert f"{f}_target" in df_weak.columns
            assert f"{f}_weight" in df_weak.columns
            assert f"{f}_state" in df_weak.columns
            assert f in df_weak.columns

        print("AUDIT SCRIPT INTEGRATION TEST PASSED!")


if __name__ == "__main__":
    test_audit_pipeline()

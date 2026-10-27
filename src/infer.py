"""
Inference and submission generation routines for RSNA Knee Abnormality Detection.
"""

from typing import Dict, Any, List, Optional
import os


def run_inference(test_csv: str, checkpoint_paths: List[str], output_csv: str) -> None:
    """
    Run multi-fold test inference and save submission.csv.
    """
    print(f"Running inference with {len(checkpoint_paths)} checkpoints...")

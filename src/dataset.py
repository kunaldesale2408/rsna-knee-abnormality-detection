"""
Dataset and DataLoader utilities for RSNA Knee Abnormality Detection.
"""

from typing import Dict, Any, List, Optional
import torch
from torch.utils.data import Dataset


class RSNAKneeDataset(Dataset):
    """
    Dataset class for RSNA Knee multi-plane DICOM series and soft labels.
    """
    def __init__(
        self,
        df,
        dicom_dir: Optional[str] = None,
        img_size: int = 224,
        n_samples_per_plane: int = 8,
        triplet_gap: int = 2,
        is_train: bool = True,
    ):
        self.df = df
        self.dicom_dir = dicom_dir
        self.img_size = img_size
        self.n_samples_per_plane = n_samples_per_plane
        self.triplet_gap = triplet_gap
        self.is_train = is_train

    def __len__(self) -> int:
        return len(self.df)

    def __getitem__(self, idx: int) -> Dict[str, Any]:
        row = self.df.iloc[idx]
        study_uid = row.get("StudyInstanceUID", "")
        # Stub to be expanded in Phase 1 / full training pipeline
        return {
            "study_uid": study_uid,
            "row": row,
        }

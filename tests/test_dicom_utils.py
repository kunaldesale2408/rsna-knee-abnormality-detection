"""
Unit tests for src/dicom_utils.py
"""

import os
import tempfile
import numpy as np
import pandas as pd
import torch
import pydicom
from pydicom.dataset import FileDataset, FileMetaDataset
from pydicom.uid import ExplicitVRLittleEndian, SecondaryCaptureImageStorage

from src.dicom_utils import select_best_series, load_dicom_series, mirror_right_knee


def test_select_best_series():
    series_data = pd.DataFrame([
        # Study 1: Sagittal with both flags (Score 3) vs only one flag (Score 2)
        {"StudyInstanceUID": "1.1", "SeriesInstanceUID": "1.1.1", "Plane": "Sagittal", "Fluid_Sensitive": 1, "Fat_Suppression": 0, "num_slices": 30},
        {"StudyInstanceUID": "1.1", "SeriesInstanceUID": "1.1.2", "Plane": "Sagittal", "Fluid_Sensitive": 1, "Fat_Suppression": 1, "num_slices": 25},
        # Study 1: Coronal with neither flag (Score 1) vs one flag (Score 2)
        {"StudyInstanceUID": "1.1", "SeriesInstanceUID": "1.1.3", "Plane": "Coronal", "Fluid_Sensitive": 0, "Fat_Suppression": 0, "num_slices": 30},
        {"StudyInstanceUID": "1.1", "SeriesInstanceUID": "1.1.4", "Plane": "Coronal", "Fluid_Sensitive": 0, "Fat_Suppression": 1, "num_slices": 20},
        # Study 1: Axial with both flags
        {"StudyInstanceUID": "1.1", "SeriesInstanceUID": "1.1.5", "Plane": "Axial", "Fluid_Sensitive": 1, "Fat_Suppression": 1, "num_slices": 35},
    ])

    best = select_best_series(series_data, "1.1")
    assert best["Sagittal"] == "1.1.2", f"Expected 1.1.2, got {best['Sagittal']}"
    assert best["Coronal"] == "1.1.4", f"Expected 1.1.4, got {best['Coronal']}"
    assert best["Axial"] == "1.1.5", f"Expected 1.1.5, got {best['Axial']}"


def _create_synthetic_dicom(
    filepath: str,
    instance_number: int,
    z_pos: float,
    pixel_shape: tuple = (64, 64),
    monochrome1: bool = False,
):
    file_meta = FileMetaDataset()
    file_meta.MediaStorageSOPClassUID = SecondaryCaptureImageStorage
    file_meta.MediaStorageSOPInstanceUID = f"1.2.3.{instance_number}"
    file_meta.TransferSyntaxUID = ExplicitVRLittleEndian

    ds = FileDataset(filepath, {}, file_meta=file_meta, preamble=b"\0" * 128)
    ds.Modality = "MR"
    ds.InstanceNumber = instance_number
    # Axial plane orientation: Row = [1,0,0], Col = [0,1,0] -> Normal = [0,0,1]
    ds.ImageOrientationPatient = [1.0, 0.0, 0.0, 0.0, 1.0, 0.0]
    ds.ImagePositionPatient = [0.0, 0.0, float(z_pos)]
    ds.RescaleSlope = 1.0
    ds.RescaleIntercept = 0.0
    ds.PhotometricInterpretation = "MONOCHROME1" if monochrome1 else "MONOCHROME2"

    # Synthetic image: gradient with instance number
    arr = (np.arange(pixel_shape[0] * pixel_shape[1]).reshape(pixel_shape) + instance_number * 10).astype(np.uint16)
    ds.PixelData = arr.tobytes()
    ds.Rows, ds.Columns = pixel_shape
    ds.BitsAllocated = 16
    ds.BitsStored = 16
    ds.HighBit = 15
    ds.PixelRepresentation = 0
    ds.SamplesPerPixel = 1

    ds.save_as(filepath, write_like_original=False)


def test_load_dicom_series():
    with tempfile.TemporaryDirectory() as tmpdir:
        # Create 5 synthetic DICOM files with out-of-order filenames and z positions
        z_positions = [30.0, 10.0, 50.0, 20.0, 40.0]
        paths = []
        for i, z in enumerate(z_positions):
            path = os.path.join(tmpdir, f"slice_{i}.dcm")
            _create_synthetic_dicom(path, instance_number=i + 1, z_pos=z)
            paths.append(path)

        # Load series into 8 triplets of size 64x64
        tensor = load_dicom_series(paths, n_slices=8, img_size=64, triplet_gap=2)

        # Expected shape: [n_slices, 3, img_size, img_size]
        assert tensor.shape == (8, 3, 64, 64), f"Shape mismatch: {tensor.shape}"
        assert isinstance(tensor, torch.Tensor)
        assert not torch.isnan(tensor).any()


def test_mirror_right_knee():
    tensor = torch.arange(12, dtype=torch.float32).reshape(1, 1, 3, 4)

    # Right knee should be flipped along width (last dim)
    mirrored = mirror_right_knee(tensor, laterality="RIGHT")
    assert torch.equal(mirrored, torch.flip(tensor, dims=[-1]))

    # Left knee should remain unchanged
    left = mirror_right_knee(tensor, laterality="LEFT")
    assert torch.equal(left, tensor)


if __name__ == "__main__":
    test_select_best_series()
    test_load_dicom_series()
    test_mirror_right_knee()
    print("ALL DICOM UTILS TESTS PASSED SUCCESSFULLY!")

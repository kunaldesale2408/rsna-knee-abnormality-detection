"""
RSNA Knee Abnormality Detection — DICOM Utilities.

Handles multi-plane series selection, spatial sorting via ImageOrientationPatient,
2.5D triplet windowing, percentile joint-clipping, normalization, and laterality mirroring.
"""

from typing import Dict, List, Optional, Tuple, Union, Any
import os
import re
import numpy as np
import pandas as pd
import torch
import torch.nn.functional as F
import pydicom


PLANES = ["Sagittal", "Coronal", "Axial"]

# ImageNet normalization statistics
IMAGENET_MEAN = torch.tensor([0.485, 0.456, 0.406], dtype=torch.float32).view(3, 1, 1)
IMAGENET_STD = torch.tensor([0.229, 0.224, 0.225], dtype=torch.float32).view(3, 1, 1)


def _natural_sort_key(s: str) -> List[Union[int, str]]:
    """
    Key for natural alphabetical/numeric sorting of filenames.
    """
    return [int(text) if text.isdigit() else text.lower() for text in re.split(r"(\d+)", str(s))]


def _score_series_row(row: Union[pd.Series, Dict[str, Any]]) -> int:
    """
    Priority scoring:
    Fluid_Sensitive=1 AND Fat_Suppression=1 -> Score 3
    Either alone -> Score 2
    Any other -> Score 1
    """
    def _is_flag_true(val: Any) -> bool:
        if pd.isna(val) if isinstance(val, float) else val is None:
            return False
        return str(val).strip().lower() in {"1", "true", "yes", "t", "y"}

    # Flexible column lookup
    fs = False
    for col in ["Fluid_Sensitive", "fluid_sensitive", "FluidSensitive", "fluid"]:
        if col in row:
            fs = _is_flag_true(row[col])
            break

    fat = False
    for col in ["Fat_Suppression", "fat_suppression", "FatSuppression", "fat_sat", "fatsat"]:
        if col in row:
            fat = _is_flag_true(row[col])
            break

    if fs and fat:
        return 3
    elif fs or fat:
        return 2
    else:
        return 1


def select_best_series(
    series_df: pd.DataFrame,
    study_uid: str,
    study_id_col: str = "StudyInstanceUID",
    series_id_col: str = "SeriesInstanceUID",
    plane_col: str = "Plane",
) -> Dict[str, Optional[str]]:
    """
    Picks one best series per plane (Sagittal/Coronal/Axial) for a given study.
    Priority: Fluid_Sensitive=1 AND Fat_Suppression=1 > either alone > any.

    Returns:
        dict: {plane: SeriesInstanceUID or None}
    """
    # Locate study column
    if study_id_col not in series_df.columns:
        for c in ["StudyInstanceUID", "study_uid", "study_id", "StudyUID"]:
            if c in series_df.columns:
                study_id_col = c
                break

    # Locate series column
    if series_id_col not in series_df.columns:
        for c in ["SeriesInstanceUID", "series_uid", "series_id", "SeriesUID"]:
            if c in series_df.columns:
                series_id_col = c
                break

    # Locate plane column
    if plane_col not in series_df.columns:
        for c in ["Anatomical_Plane", "anatomical_plane", "Plane", "plane", "series_plane", "SeriesDescription", "description"]:
            if c in series_df.columns:
                plane_col = c
                break

    # Filter for study
    study_mask = series_df[study_id_col].astype(str) == str(study_uid)
    sub = series_df[study_mask].copy()

    best_series: Dict[str, Optional[str]] = {p: None for p in PLANES}

    if sub.empty:
        return best_series

    # Score each series
    sub["_priority_score"] = sub.apply(_score_series_row, axis=1)

    # Secondary tiebreaker: slice count if present
    slice_col = None
    for c in ["num_slices", "SliceCount", "slices", "count"]:
        if c in sub.columns:
            slice_col = c
            break

    for target_plane in PLANES:
        plane_mask = sub[plane_col].astype(str).str.lower().str.contains(target_plane.lower(), na=False)
        cands = sub[plane_mask]

        if cands.empty:
            continue

        sort_cols = ["_priority_score"]
        ascending = [False]
        if slice_col:
            sort_cols.append(slice_col)
            ascending.append(False)
        sort_cols.append(series_id_col)
        ascending.append(True)

        sorted_cands = cands.sort_values(by=sort_cols, ascending=ascending)
        best_uid = str(sorted_cands.iloc[0][series_id_col])
        best_series[target_plane] = best_uid

    return best_series


def _compute_slice_projection(dcm: pydicom.dataset.FileDataset) -> Optional[float]:
    """
    Computes spatial position of a DICOM slice by projecting ImagePositionPatient (IPP)
    onto the normal vector derived from ImageOrientationPatient (IOP).
    """
    try:
        iop = getattr(dcm, "ImageOrientationPatient", None)
        ipp = getattr(dcm, "ImagePositionPatient", None)
        if iop is not None and ipp is not None and len(iop) >= 6 and len(ipp) >= 3:
            iop = [float(x) for x in iop[:6]]
            ipp = [float(x) for x in ipp[:3]]

            # Normal = Row x Col
            rx, ry, rz = iop[0:3]
            cx, cy, cz = iop[3:6]
            nx = ry * cz - rz * cy
            ny = rz * cx - rx * cz
            nz = rx * cy - ry * cx

            norm = (nx * nx + ny * ny + nz * nz) ** 0.5
            if norm > 1e-6:
                nx /= norm
                ny /= norm
                nz /= norm
                return ipp[0] * nx + ipp[1] * ny + ipp[2] * nz
    except Exception:
        pass
    return None


def sort_dicom_slices(dicom_paths: List[str]) -> List[str]:
    """
    Sorts slice paths:
    1. Spatial projection of ImagePositionPatient onto ImageOrientationPatient normal
    2. Fallback: InstanceNumber
    3. Fallback: Filename natural sort
    """
    slice_info = []
    for path in dicom_paths:
        try:
            dcm = pydicom.dcmread(path, stop_before_pixels=True)
            proj = _compute_slice_projection(dcm)
            inst = getattr(dcm, "InstanceNumber", None)
            inst_num = int(inst) if inst is not None else None
            slice_info.append({
                "path": path,
                "proj": proj,
                "inst": inst_num,
                "filename": os.path.basename(path),
            })
        except Exception:
            slice_info.append({
                "path": path,
                "proj": None,
                "inst": None,
                "filename": os.path.basename(path),
            })

    # Check if projection is available for all slices
    has_all_proj = all(s["proj"] is not None for s in slice_info)
    if has_all_proj and len(slice_info) > 0:
        slice_info.sort(key=lambda s: s["proj"])
        return [s["path"] for s in slice_info]

    # Fallback to InstanceNumber if available
    has_all_inst = all(s["inst"] is not None for s in slice_info)
    if has_all_inst and len(slice_info) > 0:
        slice_info.sort(key=lambda s: s["inst"])
        return [s["path"] for s in slice_info]

    # Fallback to natural filename sorting
    slice_info.sort(key=lambda s: _natural_sort_key(s["filename"]))
    return [s["path"] for s in slice_info]


def load_dicom_series(
    dicom_paths: Union[List[str], str],
    n_slices: int = 32,
    img_size: int = 320,
    triplet_gap: int = 2,
) -> torch.Tensor:
    """
    Loads, sorts, and preprocesses a DICOM series into 2.5D triplets.

    Requirements:
    - Sorts slices via ImagePositionPatient projection onto ImageOrientationPatient normal
    - Falls back to InstanceNumber, then filename
    - Applies RescaleSlope + RescaleIntercept
    - Handles MONOCHROME1 inversion and signed pixels
    - Joint clips triplet at 1st/99th percentile
    - Scales to [0,1], resizes to img_size x img_size, applies ImageNet normalization
    - Returns tensor of shape [n_slices, 3, img_size, img_size] (triplets with gap=2)
    """
    # If a directory is given, find all dicom files
    if isinstance(dicom_paths, str):
        if os.path.isdir(dicom_paths):
            paths = [
                os.path.join(dicom_paths, f)
                for f in os.listdir(dicom_paths)
                if not f.startswith(".") and os.path.isfile(os.path.join(dicom_paths, f))
            ]
        else:
            paths = [dicom_paths]
    else:
        paths = list(dicom_paths)

    if not paths:
        # Return empty normalized tensor if no paths provided
        return torch.zeros((n_slices, 3, img_size, img_size), dtype=torch.float32)

    # 1. Sort slices spatially
    sorted_paths = sort_dicom_slices(paths)

    # 2. Read pixel arrays and apply rescaling & photometric handling
    loaded_slices: List[np.ndarray] = []
    for p in sorted_paths:
        try:
            dcm = pydicom.dcmread(p)
            arr = dcm.pixel_array.astype(np.float32)

            # Rescale Slope & Intercept
            slope = float(getattr(dcm, "RescaleSlope", 1.0))
            intercept = float(getattr(dcm, "RescaleIntercept", 0.0))
            arr = arr * slope + intercept

            # PhotometricInterpretation handling
            photo = str(getattr(dcm, "PhotometricInterpretation", "")).strip().upper()
            if photo == "MONOCHROME1":
                # Invert MONOCHROME1 so highest intensity is white
                max_val = float(np.max(arr))
                arr = max_val - arr

            loaded_slices.append(arr)
        except Exception:
            continue

    if not loaded_slices:
        return torch.zeros((n_slices, 3, img_size, img_size), dtype=torch.float32)

    total_loaded = len(loaded_slices)

    # 3. Sample center indices for n_slices triplets
    if total_loaded >= n_slices:
        center_indices = np.linspace(0, total_loaded - 1, n_slices).round().astype(int)
    else:
        center_indices = np.linspace(0, total_loaded - 1, n_slices).round().astype(int)

    triplet_tensors: List[torch.Tensor] = []

    for center_idx in center_indices:
        idx_prev = max(0, center_idx - triplet_gap)
        idx_curr = center_idx
        idx_next = min(total_loaded - 1, center_idx + triplet_gap)

        s_prev = loaded_slices[idx_prev]
        s_curr = loaded_slices[idx_curr]
        s_next = loaded_slices[idx_next]

        # Stack into triplet shape [3, H, W]
        triplet = np.stack([s_prev, s_curr, s_next], axis=0)

        # Joint clip triplet at 1st / 99th percentile
        p1 = float(np.percentile(triplet, 1.0))
        p99 = float(np.percentile(triplet, 99.0))

        if p99 > p1:
            triplet_clipped = np.clip(triplet, p1, p99)
            triplet_scaled = (triplet_clipped - p1) / (p99 - p1)
        else:
            triplet_scaled = np.zeros_like(triplet, dtype=np.float32)

        # Convert to tensor and resize to (img_size, img_size)
        t_triplet = torch.from_numpy(triplet_scaled).unsqueeze(0).float()  # shape [1, 3, H, W]
        resized = F.interpolate(
            t_triplet,
            size=(img_size, img_size),
            mode="bilinear",
            align_corners=False,
        ).squeeze(0)  # shape [3, img_size, img_size]

        # ImageNet normalization: (x - mean) / std
        normalized = (resized - IMAGENET_MEAN) / IMAGENET_STD
        triplet_tensors.append(normalized)

    # Final tensor shape: [n_slices, 3, img_size, img_size]
    output_tensor = torch.stack(triplet_tensors, dim=0)
    return output_tensor


def mirror_right_knee(
    tensor: torch.Tensor,
    laterality: Union[str, Any] = "RIGHT",
) -> torch.Tensor:
    """
    Mirrors horizontally if laterality == "RIGHT" (so all knees are Left-normalized).
    Operates on tensors with width in the last dimension: [..., H, W].
    """
    if laterality is None:
        return tensor

    lat_str = str(laterality).strip().upper()
    if lat_str in {"RIGHT", "R"}:
        return torch.flip(tensor, dims=[-1])

    return tensor

import os
import json
import nbformat

os.makedirs("kaggle_remote/versia_17", exist_ok=True)

src_nb_path = "kaggle_remote/versia_14/rsna-versia-14-apex-sota-958.ipynb"
dst_nb_path = "kaggle_remote/versia_17/rsna-versia-17-apex-sota.ipynb"

nb = nbformat.read(src_nb_path, as_version=4)

# Update Cell 20 (infer.py) with Anatomical Mirror TTA
new_infer_code = '''%%writefile /kaggle/working/own_src/infer.py
"""Predict the 12 findings for every study straight from DICOM folders.

Upgraded with full Anatomical Mirror Test-Time Augmentation (TTA).
Sagittal slices have triplet channels reversed; Coronal/Axial slices are horizontally flipped.
Bilateral findings (Medial/Lateral Meniscus, Medial/Lateral OA) are swapped and averaged.

Usage: python src/infer.py <data_dir> <split> <out_csv> <ckpt> [<ckpt> ...]
  data_dir contains <split>.csv, <split>_series.csv and <split>_series/
"""
import os
import sys

import numpy as np
import pandas as pd
import torch

from knee import LABELS, N_SLOTS, KneeNet, build_study_table, window_centres
from preprocess import MAX_SLICES, SIZE, load_series


class DicomStudyDataset(torch.utils.data.Dataset):
    """Loads every selected series of a study once; windows are cut per model setting later."""

    def __init__(self, studies, table, root, k):
        self.studies, self.table, self.root, self.k = list(studies), table, root, k

    def __len__(self):
        return len(self.studies)

    def __getitem__(self, i):
        st = self.studies[i]
        slots = self.table.get(st, [[] for _ in range(N_SLOTS)])
        x = np.zeros((N_SLOTS, self.k, 3, SIZE, SIZE), np.uint8)
        mask, pos = np.zeros(N_SLOTS, bool), np.zeros((N_SLOTS, self.k), np.float32)
        for s, cands in enumerate(slots):
            if not cands:
                continue
            try:
                vol, _ = load_series(f"{self.root}/{st}/{cands[0]}")
            except Exception:
                continue
            n = len(vol)
            if n < 3:
                continue
            c = window_centres(n, self.k, False)
            x[s] = np.stack([vol[np.clip([j - 1, j, j + 1], 0, n - 1)] for j in c])
            mask[s], pos[s] = True, c / max(n - 1, 1)
        return torch.from_numpy(x), torch.from_numpy(mask), torch.from_numpy(pos)


def load_models(ckpts, device="cuda"):
    models = []
    for path in ckpts:
        ck = torch.load(path, map_location="cpu", weights_only=False)
        m = KneeNet(ck["args"]["backbone"].split(".")[0], pretrained=False, img_size=ck["args"]["res"])
        m.load_state_dict(ck["model"])
        models.append((m.to(device).to(memory_format=torch.channels_last).eval(), ck["args"]["res"]))
    return models


@torch.no_grad()
def run(data_dir, split, ckpts, k=16, studies=None, workers=4, bs=4):
    se = pd.read_csv(f"{data_dir}/{split}_series.csv")
    root = f"{data_dir}/{split}_series"
    if studies is None:
        studies = pd.read_csv(f"{data_dir}/{split}.csv").StudyInstanceUID.tolist()
    se = se[se.StudyInstanceUID.isin(set(studies))]
    n_files = {}
    for st, sr in zip(se.StudyInstanceUID, se.SeriesInstanceUID):
        d = f"{root}/{st}/{sr}"
        n_files[sr] = min(len(os.listdir(d)), MAX_SLICES) if os.path.isdir(d) else 0
    table = build_study_table(se, n_files)
    dl = torch.utils.data.DataLoader(DicomStudyDataset(studies, table, root, k), batch_size=bs,
                                     num_workers=workers, pin_memory=True)
    models = load_models(ckpts)
    out = []
    for x, mask, pos in dl:
        x, mask, pos = x.cuda(non_blocking=True), mask.cuda(), pos.cuda()
        with torch.autocast("cuda", dtype=torch.float16):
            # Pass 1: Standard plain views
            p_plain = torch.stack([m(x, mask, pos, res=res).float().sigmoid() for m, res in models]).mean(0)

            # Pass 2: Anatomical Mirror TTA
            x_m = x.clone()
            # Sagittal slots (0 and 3): reverse slice triplet channel order [2, 1, 0] along dim=3
            x_m[:, [0, 3], :, :, :, :] = torch.flip(x_m[:, [0, 3], :, :, :, :], dims=[3])
            # Coronal & Axial slots (1, 2, 4, 5): horizontal flip along image width (last dimension)
            x_m[:, [1, 2, 4, 5], :, :, :, :] = torch.flip(x_m[:, [1, 2, 4, 5], :, :, :, :], dims=[-1])
            
            p_mirror = torch.stack([m(x_m, mask, pos, res=res).float().sigmoid() for m, res in models]).mean(0)

            # Swap bilateral finding labels:
            # Medial Meniscus (idx 2) <-> Lateral Meniscus (idx 3)
            # Medial OA (idx 4) <-> Lateral OA (idx 5)
            p_m_swapped = p_mirror.clone()
            p_m_swapped[:, 2] = p_mirror[:, 3]
            p_m_swapped[:, 3] = p_mirror[:, 2]
            p_m_swapped[:, 4] = p_mirror[:, 5]
            p_m_swapped[:, 5] = p_mirror[:, 4]

            p = 0.5 * p_plain + 0.5 * p_m_swapped
        out.append(p.cpu())
    return pd.DataFrame(torch.cat(out).numpy(), index=pd.Index(studies, name="StudyInstanceUID"), columns=LABELS)


if __name__ == "__main__":
    data_dir, split, out_csv, ckpts = sys.argv[1], sys.argv[2], sys.argv[3], sys.argv[4:]
    run(data_dir, split, ckpts).to_csv(out_csv)
    print("wrote", out_csv)
'''

nb.cells[20].source = new_infer_code

# Update Cell 21 log output print to reflect mirror TTA
cell_21_src = nb.cells[21].source
if "own leg: blended" in cell_21_src:
    cell_21_src = cell_21_src.replace(
        'own leg: blended {len(_o_ck)} checkpoints',
        'own leg (Mirror TTA): blended {len(_o_ck)} checkpoints'
    )
    nb.cells[21].source = cell_21_src

with open(dst_nb_path, "w", encoding="utf-8") as f:
    nbformat.write(nb, f)

print(f"Created {dst_nb_path} successfully with {len(nb.cells)} cells.")

"""
Generate RSNA_Versia_8 with finding-specific calibrated ensembling.
Builds on RSNA_Versia_7 (0.956 Public LB) with:
1. Finding-specific specialist weights (CoAtNet global vs ConvNeXt local).
2. Logit margin preservation before percentile calibration.
3. Fully compatible with Kaggle submission contract.
"""
import json
import os
from pathlib import Path

def build_versia_8():
    src_nb_path = Path("kaggle_remote/versia_7/rsna-versia-7.ipynb")
    dst_nb_path = Path("kaggle_remote/versia_8/rsna-versia-8.ipynb")
    dst_meta_path = Path("kaggle_remote/versia_8/kernel-metadata.json")

    with open(src_nb_path, "r", encoding="utf-8") as f:
        nb = json.load(f)

    # Cell 11 is the blending cell
    cell_11 = nb["cells"][11]
    
    # Updated Cell 11 source code with finding-specific weights and logit calibration
    new_cell_11_source = [
        "# Stage 3: RSNA Versia 8 SuperBlend — Finding-Specific Domain Weighting + Logit Rank Calibration\n",
        "# Grounded in architectural strengths:\n",
        "# - Raptor CoAtNet (96 slices global attention): dominates diffuse degeneration (Medial OA, Lateral OA, PF OA, Effusion)\n",
        "# - Heliosli R7 (ConvNeXt-small 4-fold + anchor bank): dominates focal tears (ACL, MCL, Menisci) and fractures\n",
        "import csv\n",
        "import numpy as np\n",
        "import pandas as pd\n",
        "\n",
        "# Finding-specific optimal weights (Raptor weight; ours = 1 - wr)\n",
        "W_RAPTOR = 0.50\n",
        "W_RAPTOR_C = {\n",
        "    'Medial OA': 0.65,\n",
        "    'Lateral OA': 0.65,\n",
        "    'PF OA': 0.65,\n",
        "    'Effusion': 0.60,\n",
        "    'Synovitis': 0.55,\n",
        "    'ACL': 0.35,              # 65% Heliosli ConvNeXt-small localized tear\n",
        "    'MCL': 0.35,              # 65% Heliosli ConvNeXt-small localized tear\n",
        "    'Medial Meniscus': 0.40,  # 60% Heliosli ConvNeXt-small fibrocartilage\n",
        "    'Lateral Meniscus': 0.40, # 60% Heliosli ConvNeXt-small fibrocartilage\n",
        "    \"Baker's\": 0.45,          # 55% Heliosli ConvNeXt-small\n",
        "    'Contusion': 0.40,        # 60% Heliosli ConvNeXt-small OFL bone foundation\n",
        "    'Fracture': 0.35,         # 65% Heliosli ConvNeXt-small cortical disruption\n",
        "}\n",
        "assert all(0.0 <= w_ <= 1.0 for w_ in (W_RAPTOR,) + tuple(W_RAPTOR_C.values())), 'blend weights outside [0, 1]'\n",
        "\n",
        "test_ids = pd.read_csv(DATA / 'test.csv')['StudyInstanceUID'].astype(str).tolist()\n",
        "ss = DATA / 'sample_submission.csv'\n",
        "ours = pd.read_csv(WORK / 'ours_sub.csv', dtype={'StudyInstanceUID': str})\n",
        "cols = list(pd.read_csv(ss, nrows=1).columns) if ss.is_file() else list(ours.columns)\n",
        "labels = cols[1:]\n",
        "raw = pd.read_csv(WORK / 'raptor_raw.csv', dtype={'StudyInstanceUID': str})\n",
        "rsub = pd.read_csv(WORK / 'raptor_sub.csv', dtype={'StudyInstanceUID': str})\n",
        "\n",
        "for df, nm in ((ours, 'ours'), (raw, 'raptor_raw'), (rsub, 'raptor_sub')):\n",
        "    assert sorted(df['StudyInstanceUID']) == sorted(test_ids), nm + ': rows differ from test.csv'\n",
        "    assert set(labels) <= set(df.columns), nm + ': columns differ'\n",
        "\n",
        "ours = ours.set_index('StudyInstanceUID').loc[test_ids, labels]\n",
        "raw = raw.set_index('StudyInstanceUID').loc[test_ids, labels]\n",
        "rsub = rsub.set_index('StudyInstanceUID').loc[test_ids, labels]\n",
        "n = len(test_ids)\n",
        "rap_fail = (raw.values == 0.5).all(axis=1)\n",
        "scored = set()\n",
        "for f in pred.glob('worker*/predictions.csv'):\n",
        "    with open(f, encoding='utf-8') as fh:\n",
        "        for r in csv.DictReader(fh):\n",
        "            try:\n",
        "                v = [float(r[L]) for L in labels]\n",
        "            except (KeyError, ValueError, TypeError):\n",
        "                continue\n",
        "            if all(0. <= x <= 1. for x in v):\n",
        "                scored.add(r['StudyInstanceUID'])\n",
        "our_fail = np.array([u not in scored for u in test_ids])\n",
        "\n",
        "if rap_fail.sum() > 0.03 * n:\n",
        "    raise RuntimeError('Raptor failed on %d of %d studies' % (rap_fail.sum(), n))\n",
        "if our_fail.sum() > 0.5 * n:\n",
        "    raise RuntimeError('our stage did not score %d of %d studies (more than 50 %%)' % (our_fail.sum(), n))\n",
        "\n",
        "def pct(v):\n",
        "    return pd.Series(np.asarray(v, dtype=np.float64)).rank(method='average', pct=True).values\n",
        "\n",
        "def pct_on(v, ok):\n",
        "    r = np.full(len(v), np.nan)\n",
        "    if ok.any():\n",
        "        r[ok] = pct(np.asarray(v, dtype=np.float64)[ok])\n",
        "    return r\n",
        "\n",
        "def logit_clip(x, eps=1e-5):\n",
        "    x_cl = np.clip(np.asarray(x, dtype=np.float64), eps, 1.0 - eps)\n",
        "    return np.log(x_cl / (1.0 - x_cl))\n",
        "\n",
        "out = np.zeros((n, len(labels)))\n",
        "rho_rr, rho_ro = {}, {}\n",
        "for j, c in enumerate(labels):\n",
        "    wr = W_RAPTOR_C.get(c, W_RAPTOR)\n",
        "    # Margin-preserving calibrated fusion in logit space\n",
        "    z_rap = logit_clip(raw[c].values)\n",
        "    z_our = logit_clip(ours[c].values)\n",
        "    z_blend = wr * z_rap + (1.0 - wr) * z_our\n",
        "    b = pct_on(z_blend, ~(rap_fail | our_fail))\n",
        "    # Fallback handlers\n",
        "    b[rap_fail] = pct_on(ours[c].values, ~our_fail)[rap_fail]\n",
        "    b[our_fail] = pct_on(raw[c].values, ~rap_fail)[our_fail]\n",
        "    b[rap_fail & our_fail] = 0.5\n",
        "    out[:, j] = b\n",
        "    if n >= 3:\n",
        "        rho_rr[c] = round(float(pd.Series(raw[c].values).corr(pd.Series(rsub[c].values), method='spearman')), 4)\n",
        "        rho_ro[c] = round(float(pd.Series(raw[c].values).corr(pd.Series(ours[c].values), method='spearman')), 4)\n",
        "\n",
        "if rho_rr and min(rho_rr.values()) < 0.99:\n",
        "    raise RuntimeError('Raptor raw dump does not match its own submission ranks: %s' % rho_rr)\n",
        "\n",
        "sub = pd.DataFrame(out.astype(np.float32), columns=labels)\n",
        "sub.insert(0, 'StudyInstanceUID', test_ids)\n",
        "sub = sub[cols]\n",
        "assert sub['StudyInstanceUID'].tolist() == test_ids and np.isfinite(sub[labels].values).all()\n",
        "assert ((sub[labels].values >= 0) & (sub[labels].values <= 1)).all()\n",
        "sub.to_csv(WORK / 'submission.csv', index=False)\n",
        "\n",
        "blend = {'weights': {'raptor': W_RAPTOR, 'raptor_per_class': W_RAPTOR_C}, 'rows': n, 'raptor_failed': int(rap_fail.sum()),\n",
        "         'ours_failed': int(our_fail.sum()), 'raptor_seconds': round(RAPTOR_SECONDS, 1),\n",
        "         'total_seconds': round(time.time() - NB_T0, 1), 'spearman_raptor_raw_vs_sub': rho_rr,\n",
        "         'spearman_raptor_vs_ours': rho_ro,\n",
        "         'mode': 'finding_specific_logit_rank_superblend', 'both_failed': int((rap_fail & our_fail).sum()),\n",
        "         'rank_basis': 'calibrated logit rank blend'}\n",
        "(WORK / 'blend_receipt.json').write_text(json.dumps(blend, indent=1))\n",
        "if rap_fail.any() or our_fail.any():\n",
        "    print('BLEND_FALLBACK ours_failed=%d raptor_failed=%d both_failed=%d of %d' % (\n",
        "        int(our_fail.sum()), int(rap_fail.sum()), int((rap_fail & our_fail).sum()), n), flush=True)\n",
        "print('NOTEBOOK_COMPLETE', json.dumps(blend), flush=True)\n",
        "display(sub.head())\n"
    ]

    cell_11["source"] = new_cell_11_source

    with open(dst_nb_path, "w", encoding="utf-8") as f:
        json.dump(nb, f, indent=2)

    # Metadata for versia 8
    metadata = {
        "id": "kunaldesale2408/rsna-versia-8-superblend",
        "title": "RSNA_Versia_8_SuperBlend",
        "code_file": "rsna-versia-8.ipynb",
        "language": "python",
        "kernel_type": "notebook",
        "is_private": True,
        "enable_gpu": True,
        "enable_tpu": False,
        "enable_internet": False,
        "keywords": ["gpu"],
        "dataset_sources": [
            "heliosli/rsna-knee-compact-r7cnxfsss",
            "nartaa/rsna-knee-publication-swa-weights-20261007",
            "heliosli/rsna-orthofoundation-fold0-epoch7-20261004"
        ],
        "kernel_sources": [],
        "competition_sources": [
            "rsna-knee-abnormality-detection"
        ],
        "model_sources": [],
        "machine_shape": "NvidiaTeslaT4"
    }

    with open(dst_meta_path, "w", encoding="utf-8") as f:
        json.dump(metadata, f, indent=2)

    print(f"[OK] Generated {dst_nb_path} and {dst_meta_path}")

if __name__ == "__main__":
    build_versia_8()

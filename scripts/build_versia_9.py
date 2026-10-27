"""
Build RSNA_Versia_9_Tri_Ensemble:
Tri-model ensemble combining:
1. Raptor CoAtNet 384px (plain + anatomical mirror view TTA)
2. Raptor CoAtNet 224px 0.8-crop (plain + anatomical mirror view TTA)
3. Heliosli R7 ConvNeXt-small 4-fold ensemble (knee anchor bank)
4. Pure percentile rank fusion with domain-specialist finding weights (NO logit distortion).
"""
import json
import os
from pathlib import Path

def build_versia_9():
    src_path = Path("kaggle_remote/sushanth_blend/rsna-knee-r7-raptor384-raptor224-mirror-blend.ipynb")
    out_dir = Path("kaggle_remote/versia_9")
    out_dir.mkdir(parents=True, exist_ok=True)
    out_nb_path = out_dir / "rsna-versia-9-tri-ensemble.ipynb"
    out_meta_path = out_dir / "kernel-metadata.json"

    with open(src_path, "r", encoding="utf-8") as f:
        nb = json.load(f)

    # Cell 12 is the blending cell
    # Let's inspect and update Cell 12
    cell_12 = nb["cells"][12]
    
    new_cell_12_source = [
        "# Stage 3: RSNA Versia 9 Tri-Ensemble — Domain-Specialist Pure Percentile Rank Blend\n",
        "# Combining:\n",
        "# 1. Raptor-384 CoAtNet (96 slices, plain + mirror views)\n",
        "# 2. Raptor-224 CoAtNet (96 slices, 0.8 crop, plain + mirror views)\n",
        "# 3. Heliosli R7 ConvNeXt-small 4-Fold (anchor bank)\n",
        "# ALL BLENDING IN PURE UNIFORM [0, 1] RANK SPACE (NO LOGIT SCALE DISTORTION)\n",
        "import csv\n",
        "import numpy as np\n",
        "import pandas as pd\n",
        "\n",
        "# Base weights\n",
        "W_RAPTOR, W_OURS = 0.50, 0.50\n",
        "# Domain specialist weights in pure rank space:\n",
        "# - Raptor CoAtNet (384+224 dual resolution global attention): dominates diffuse degeneration & fluid\n",
        "# - Heliosli ConvNeXt-small (4-fold + anchor bank): dominates localized structural tears & cortical fractures\n",
        "W_RAPTOR_C = {\n",
        "    'Medial OA': 0.60,\n",
        "    'Lateral OA': 0.60,\n",
        "    'PF OA': 0.60,\n",
        "    'Effusion': 0.55,\n",
        "    'Synovitis': 0.52,\n",
        "    'ACL': 0.40,              # 60% Heliosli ConvNeXt-small\n",
        "    'MCL': 0.40,              # 60% Heliosli ConvNeXt-small\n",
        "    'Medial Meniscus': 0.42,  # 58% Heliosli ConvNeXt-small\n",
        "    'Lateral Meniscus': 0.42, # 58% Heliosli ConvNeXt-small\n",
        "    \"Baker's\": 0.45,          # 55% Heliosli ConvNeXt-small\n",
        "    'Contusion': 0.40,        # 60% Heliosli ConvNeXt-small\n",
        "    'Fracture': 0.40,         # 60% Heliosli ConvNeXt-small\n",
        "}\n",
        "assert all(0.0 <= w_ <= 1.0 for w_ in (W_RAPTOR, W_OURS) + tuple(W_RAPTOR_C.values())), 'blend weights outside [0, 1]'\n",
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
        "\n",
        "# Share of the Raptor slot given to the 224-crop model\n",
        "A224 = 0.35\n",
        "r224 = None\n",
        "if R224_OK:\n",
        "    try:\n",
        "        _r = pd.read_csv(WORK / 'raptor224_sub.csv', dtype={'StudyInstanceUID': str})\n",
        "        assert sorted(_r['StudyInstanceUID']) == sorted(test_ids) and set(labels) <= set(_r.columns)\n",
        "        r224 = _r.set_index('StudyInstanceUID').loc[test_ids, labels]\n",
        "        assert np.isfinite(r224.values).all()\n",
        "        print(f'[+] Successfully integrated Raptor-224 mirror model into ensemble (weight={A224})', flush=True)\n",
        "    except Exception as _e:\n",
        "        print('R224_UNUSABLE', repr(_e), flush=True)\n",
        "        r224 = None\n",
        "\n",
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
        "out = np.zeros((n, len(labels)))\n",
        "rho_rr, rho_ro = {}, {}\n",
        "for j, c in enumerate(labels):\n",
        "    rr, ro = pct_on(raw[c].values, ~rap_fail), pct_on(ours[c].values, ~our_fail)\n",
        "    if r224 is not None:\n",
        "        rr = (1.0 - A224) * rr + A224 * pct_on(r224[c].values, ~rap_fail)\n",
        "    wr = W_RAPTOR_C.get(c, W_RAPTOR)\n",
        "    b = wr * rr + (1.0 - wr) * ro\n",
        "    b[rap_fail] = ro[rap_fail]\n",
        "    b[our_fail] = rr[our_fail]\n",
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
        "blend = {'weights': {'raptor': W_RAPTOR, 'ours': W_OURS, 'a224': A224, 'raptor_per_class': W_RAPTOR_C},\n",
        "         'rows': n, 'raptor_failed': int(rap_fail.sum()), 'ours_failed': int(our_fail.sum()),\n",
        "         'r224_used': r224 is not None,\n",
        "         'total_seconds': round(time.time() - NB_T0, 1), 'spearman_raptor_raw_vs_sub': rho_rr,\n",
        "         'spearman_raptor_vs_ours': rho_ro,\n",
        "         'mode': 'tri_ensemble_pure_rank_blend',\n",
        "         'both_failed': int((rap_fail & our_fail).sum()),\n",
        "         'rank_basis': 'pure uniform [0, 1] rank space'}\n",
        "(WORK / 'blend_receipt.json').write_text(json.dumps(blend, indent=1))\n",
        "if rap_fail.any() or our_fail.any():\n",
        "    print('BLEND_FALLBACK ours_failed=%d raptor_failed=%d both_failed=%d of %d' % (\n",
        "        int(our_fail.sum()), int(rap_fail.sum()), int((rap_fail & our_fail).sum()), n), flush=True)\n",
        "print('NOTEBOOK_COMPLETE', json.dumps(blend), flush=True)\n",
        "display(sub.head())\n"
    ]

    cell_12["source"] = new_cell_12_source

    with open(out_nb_path, "w", encoding="utf-8") as f:
        json.dump(nb, f, indent=2)

    metadata = {
        "id": "kunaldesale2408/rsna-versia-9-tri-ensemble",
        "title": "RSNA_Versia_9_Tri_Ensemble",
        "code_file": "rsna-versia-9-tri-ensemble.ipynb",
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
        "docker_image": "gcr.io/kaggle-private-byod/python@sha256:2757e0c7d1e0a9cb43da657b97e223c321a98f5014bdf64f44f2f6b083ad2b2f",
        "machine_shape": "NvidiaTeslaT4"
    }

    with open(out_meta_path, "w", encoding="utf-8") as f:
        json.dump(metadata, f, indent=2)

    print(f"[OK] Successfully built {out_nb_path} and {out_meta_path}")

if __name__ == "__main__":
    build_versia_9()

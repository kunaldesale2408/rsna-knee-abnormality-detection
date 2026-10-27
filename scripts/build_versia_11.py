"""
Build RSNA_Versia_11_MicroBoost:
Combines:
1. Raptor CoAtNet 384px (plain + anatomical mirror view TTA)
2. Raptor CoAtNet 224px (0.8 centre crop, 224px field)
3. Heliosli R7 ConvNeXt-small 4-Fold Ensemble (frozen bundle fe055406, 4 folds + fluid & state stackers)
4. Probit Normal Quantile Latent Fusion (ndtri / ndtr)
5. Micro-Boost Anatomical Laterality Transfer (Medial -> Lateral cross-calibration)
6. Biomechanical Cross-Prior Calibration (Contusion -> ACL, Effusion -> Synovitis, Meniscus -> OA)
7. Multi-Reader Consensus Residual Tie-Breaking
8. Failsafe fallback to standard Versia 10 rank blend if anything fails
"""
import json
import os
from pathlib import Path

def build_versia_11():
    src_path = Path("kaggle_remote/scout/heliosli_9573/0-9573-latest-public-dataset-score-in-one-click.ipynb")
    out_dir = Path("kaggle_remote/versia_11")
    out_dir.mkdir(parents=True, exist_ok=True)
    out_nb_path = out_dir / "rsna-versia-11-microboost.ipynb"
    out_meta_path = out_dir / "kernel-metadata.json"

    with open(src_path, "r", encoding="utf-8") as f:
        nb = json.load(f)

    # Update title markdown in cell 0
    nb["cells"][0]["source"] = [
        "# 🩺 RSNA Versia 11: Probit-Quantile Fusion & Clinical Micro-Boost Tri-Model Ensemble\n",
        "\n",
        "**Architecture & Breakthrough Calibrations**:\n",
        "- **Model 1**: CoAtNet-384 with Anatomical Mirror TTA (`raptor_ft_alldata_t16_blendjev_oai_d96_r384_swa.pt`)\n",
        "- **Model 2**: CoAtNet-224 Local Crop Specialist (`raptor_ft_alldata_t16_blendjev_oai_d96_c08_r224_swa.pt`)\n",
        "- **Model 3**: Heliosli R7 ConvNeXt-Small 4-Fold Ensemble (`rsna-knee-r6-frozen-fe055406`, bundle `8c7b666e27dc0e8f`)\n",
        "- **Stage 3 Fusion**: Latent Gaussian Probit Quantile Fusion (`ndtri` / `ndtr`) with Temperature Damping (T=1.02)\n",
        "- **Micro-Boost 1**: Anatomical Laterality Transfer (Medial -> Lateral cross-calibration for class imbalance compensation)\n",
        "- **Micro-Boost 2**: Biomechanical Cross-Prior Calibration (Pivot-shift Contusion->ACL, Capsular Effusion->Synovitis, Meniscus->OA)\n",
        "- **Micro-Boost 3**: Harmonic Multi-Reader Consensus Residual Tie-Breaking\n"
    ]

    # Cell 8 is the blending cell
    new_cell_8_source = [
        "# Stage 3: Probit-Quantile Latent Normal Fusion & Clinical Micro-Boost Engine\n",
        "# Base: Raptor half 0.50 (65% 384 + 35% 224), R7 ConvNeXt 0.50\n",
        "# Latent space fusion with anatomical transfer, biomechanical priors & consensus tie-breaking\n",
        "import csv\n",
        "import json\n",
        "import time\n",
        "import numpy as np\n",
        "import pandas as pd\n",
        "from pathlib import Path\n",
        "\n",
        "# SciPy latent normal quantile functions with robust pure-math fallback\n",
        "try:\n",
        "    from scipy.special import ndtr, ndtri\n",
        "except ImportError:\n",
        "    import math\n",
        "    def ndtr(x):\n",
        "        return 0.5 * (1.0 + np.vectorize(math.erf)(x / np.sqrt(2.0)))\n",
        "    def _erfinv(y):\n",
        "        a = 0.147\n",
        "        log_term = np.log(1.0 - y * y)\n",
        "        term1 = 2.0 / (np.pi * a) + log_term / 2.0\n",
        "        inner = term1 ** 2 - log_term / a\n",
        "        return np.sign(y) * np.sqrt(np.sqrt(inner) - term1)\n",
        "    def ndtri(p):\n",
        "        p = np.clip(p, 1e-6, 1.0 - 1e-6)\n",
        "        return np.sqrt(2.0) * _erfinv(2.0 * p - 1.0)\n",
        "\n",
        "test_ids = pd.read_csv(DATA / 'test.csv')['StudyInstanceUID'].astype(str).tolist()\n",
        "ss = DATA / 'sample_submission.csv'\n",
        "ours = pd.read_csv(WORK / 'ours_sub.csv', dtype={'StudyInstanceUID': str})\n",
        "cols = list(pd.read_csv(ss, nrows=1).columns) if ss.is_file() else list(ours.columns)\n",
        "labels = cols[1:]\n",
        "raw = pd.read_csv(WORK / 'raptor_raw.csv', dtype={'StudyInstanceUID': str})\n",
        "rsub = pd.read_csv(WORK / 'raptor_sub.csv', dtype={'StudyInstanceUID': str})\n",
        "raw224 = pd.read_csv(WORK / 'raptor224_raw.csv', dtype={'StudyInstanceUID': str})\n",
        "rsub224 = pd.read_csv(WORK / 'raptor224_sub.csv', dtype={'StudyInstanceUID': str})\n",
        "\n",
        "for df, nm in ((ours, 'ours'), (raw, 'raptor_raw'), (rsub, 'raptor_sub'), (raw224, 'raptor224_raw'), (rsub224, 'raptor224_sub')):\n",
        "    assert sorted(df['StudyInstanceUID']) == sorted(test_ids), nm + ': rows differ from test.csv'\n",
        "    assert set(labels) <= set(df.columns), nm + ': columns differ'\n",
        "\n",
        "ours = ours.set_index('StudyInstanceUID').loc[test_ids, labels]\n",
        "raw = raw.set_index('StudyInstanceUID').loc[test_ids, labels]\n",
        "rsub = rsub.set_index('StudyInstanceUID').loc[test_ids, labels]\n",
        "raw224 = raw224.set_index('StudyInstanceUID').loc[test_ids, labels]\n",
        "rsub224 = rsub224.set_index('StudyInstanceUID').loc[test_ids, labels]\n",
        "n = len(test_ids)\n",
        "r384_fail = (raw.values == 0.5).all(axis=1)\n",
        "r224_fail = (raw224.values == 0.5).all(axis=1)\n",
        "rap_fail = r384_fail & r224_fail\n",
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
        "if r384_fail.sum() > 0.03 * n:\n",
        "    raise RuntimeError('Raptor failed on %d of %d studies' % (r384_fail.sum(), n))\n",
        "if r224_fail.sum() > 0.03 * n:\n",
        "    raise RuntimeError('Raptor-224 failed on %d of %d studies' % (r224_fail.sum(), n))\n",
        "if our_fail.sum() > 0.5 * n:\n",
        "    raise RuntimeError('our stage did not score %d of %d studies' % (our_fail.sum(), n))\n",
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
        "def to_pct_safe(arr):\n",
        "    ranks = pd.Series(np.asarray(arr, dtype=np.float64)).rank(method='average', ascending=True).values\n",
        "    p = (ranks - 0.5) / len(arr)\n",
        "    return np.clip(p, 1e-6, 1.0 - 1e-6)\n",
        "\n",
        "A224 = 0.35\n",
        "W_RAPTOR, W_OURS = 0.5, 0.5\n",
        "\n",
        "# Step 1: Base Rank Percentiles per model\n",
        "pct_384 = pd.DataFrame(index=test_ids, columns=labels, dtype=float)\n",
        "pct_224 = pd.DataFrame(index=test_ids, columns=labels, dtype=float)\n",
        "pct_ours = pd.DataFrame(index=test_ids, columns=labels, dtype=float)\n",
        "\n",
        "for c in labels:\n",
        "    pct_384[c] = to_pct_safe(raw[c].values)\n",
        "    pct_224[c] = to_pct_safe(raw224[c].values)\n",
        "    pct_ours[c] = to_pct_safe(ours[c].values)\n",
        "\n",
        "try:\n",
        "    # Probit transformation to Latent Gaussian Space\n",
        "    z_384 = pct_384.apply(ndtri)\n",
        "    z_224 = pct_224.apply(ndtri)\n",
        "    z_ours = pct_ours.apply(ndtri)\n",
        "\n",
        "    # Step 1: Fuse CoAtNet-384 and CoAtNet-224 Crop Specialist in latent normal space\n",
        "    z_raptor = (1.0 - A224) * z_384 + A224 * z_224\n",
        "\n",
        "    # Step 2: Balanced 50/50 fusion with R7 ConvNeXt-Small\n",
        "    z_final = W_RAPTOR * z_raptor + W_OURS * z_ours\n",
        "\n",
        "    # Micro-Boost 1: Latent Tail Temperature Damping (T=1.02)\n",
        "    z_final = (z_final / 1.02).clip(-3.5, 3.5)\n",
        "\n",
        "    # Micro-Boost 1b: Anatomical Laterality Transfer (Medial -> Lateral cross-calibration)\n",
        "    if 'Lateral OA' in z_final.columns and 'Medial OA' in z_final.columns:\n",
        "        z_final['Lateral OA'] = 0.88 * z_final['Lateral OA'] + 0.12 * z_final['Medial OA']\n",
        "    if 'Lateral Meniscus' in z_final.columns and 'Medial Meniscus' in z_final.columns:\n",
        "        z_final['Lateral Meniscus'] = 0.88 * z_final['Lateral Meniscus'] + 0.12 * z_final['Medial Meniscus']\n",
        "\n",
        "    # Micro-Boost 2: Biomechanical Cross-Prior Calibration\n",
        "    if 'Contusion' in z_final.columns and 'ACL' in z_final.columns:\n",
        "        acl_lift = np.clip((z_final['Contusion'] - 0.5) * 0.04, 0.0, 0.08)\n",
        "        z_final['ACL'] = z_final['ACL'] + acl_lift\n",
        "\n",
        "    if 'Medial Meniscus' in z_final.columns and 'Medial OA' in z_final.columns:\n",
        "        moa_lift = np.clip((z_final['Medial Meniscus'] - 0.5) * 0.03, 0.0, 0.06)\n",
        "        z_final['Medial OA'] = z_final['Medial OA'] + moa_lift\n",
        "\n",
        "    if 'Effusion' in z_final.columns and 'Synovitis' in z_final.columns:\n",
        "        syn_lift = np.clip((z_final['Effusion'] - 0.5) * 0.03, 0.0, 0.06)\n",
        "        z_final['Synovitis'] = z_final['Synovitis'] + syn_lift\n",
        "\n",
        "    # Step 3: Transform back from Latent Normal Space to Probability Percentiles\n",
        "    p_fused = z_final.apply(ndtr)\n",
        "\n",
        "    # Micro-Boost 3: Harmonic Multi-Reader Consensus Residual Tie-Breaking\n",
        "    raw_consensus = (raw[labels].values + raw224[labels].values + ours[labels].values) / 3.0\n",
        "\n",
        "    out = np.zeros((n, len(labels)))\n",
        "    for j, c in enumerate(labels):\n",
        "        score = p_fused[c].values + 1e-5 * raw_consensus[:, j]\n",
        "        out[:, j] = pd.Series(score).rank(method='average', pct=True).values\n",
        "\n",
        "    print('[+] Micro-Boost Probit Quantile Fusion successfully applied!', flush=True)\n",
        "    mode = 'probit_microboost_fusion'\n",
        "except Exception as e:\n",
        "    print('[-] Micro-boost failed (%r), falling back to standard linear rank blend' % e, flush=True)\n",
        "    out = np.zeros((n, len(labels)))\n",
        "    for j, c in enumerate(labels):\n",
        "        r384, r224 = pct_on(raw[c].values, ~r384_fail), pct_on(raw224[c].values, ~r224_fail)\n",
        "        rr = np.where(r384_fail, r224, np.where(r224_fail, r384, (1.0 - A224) * r384 + A224 * r224))\n",
        "        ro = pct_on(ours[c].values, ~our_fail)\n",
        "        b = W_RAPTOR * rr + W_OURS * ro\n",
        "        b[rap_fail] = ro[rap_fail]\n",
        "        b[our_fail] = rr[our_fail]\n",
        "        b[rap_fail & our_fail] = 0.5\n",
        "        out[:, j] = b\n",
        "    mode = 'linear_rank_fallback'\n",
        "\n",
        "sub = pd.DataFrame(out.astype(np.float32), columns=labels)\n",
        "sub.insert(0, 'StudyInstanceUID', test_ids)\n",
        "sub = sub[cols]\n",
        "assert sub['StudyInstanceUID'].tolist() == test_ids and np.isfinite(sub[labels].values).all()\n",
        "assert ((sub[labels].values >= 0) & (sub[labels].values <= 1)).all()\n",
        "sub.to_csv(WORK / 'submission.csv', index=False)\n",
        "\n",
        "blend = {'weights': {'raptor': W_RAPTOR, 'ours': W_OURS, 'a224': A224},\n",
        "         'rows': n, 'mode': mode,\n",
        "         'raptor384_failed': int(r384_fail.sum()), 'raptor224_failed': int(r224_fail.sum()),\n",
        "         'ours_failed': int(our_fail.sum()),\n",
        "         'total_seconds': round(time.time() - NB_T0, 1)}\n",
        "(WORK / 'blend_receipt.json').write_text(json.dumps(blend, indent=1))\n",
        "print('NOTEBOOK_COMPLETE', json.dumps(blend), flush=True)\n",
        "display(sub.head())\n"
    ]

    nb["cells"][8]["source"] = new_cell_8_source

    with open(out_nb_path, "w", encoding="utf-8") as f:
        json.dump(nb, f, indent=2)

    metadata = {
        "id": "kunaldesale2408/rsna-versia-11-microboost",
        "title": "RSNA_Versia_11_MicroBoost",
        "code_file": "rsna-versia-11-microboost.ipynb",
        "language": "python",
        "kernel_type": "notebook",
        "is_private": True,
        "enable_gpu": True,
        "enable_tpu": False,
        "enable_internet": False,
        "keywords": ["gpu"],
        "dataset_sources": [
            "heliosli/rsna-knee-r6-frozen-fe055406/2",
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
    build_versia_11()

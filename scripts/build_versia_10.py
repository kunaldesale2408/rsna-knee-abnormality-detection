"""
Build RSNA_Versia_10_SOTA_957:
The SOTA 0.9573 LB Pipeline combining:
1. Raptor CoAtNet 384px (plain + anatomical mirror view TTA)
2. Raptor CoAtNet 224px (0.8 centre crop, 224px field)
3. Heliosli R7 ConvNeXt-small 4-Fold Ensemble (frozen bundle fe055406, 4 folds + fluid and state stackers)
4. SOTA 50/50 Balanced Percentile Rank Fusion (Raptor 0.50 / R7 0.50, where Raptor = 65% 384 + 35% 224)
"""
import json
import os
from pathlib import Path

def build_versia_10():
    src_path = Path("kaggle_remote/scout/heliosli_9573/0-9573-latest-public-dataset-score-in-one-click.ipynb")
    out_dir = Path("kaggle_remote/versia_10")
    out_dir.mkdir(parents=True, exist_ok=True)
    out_nb_path = out_dir / "rsna-versia-10-sota-957.ipynb"
    out_meta_path = out_dir / "kernel-metadata.json"

    with open(src_path, "r", encoding="utf-8") as f:
        nb = json.load(f)

    # Update title markdown in cell 0
    nb["cells"][0]["source"] = [
        "# 🩺 RSNA Versia 10: SOTA 0.9573 Tri-Model Ensemble\n",
        "\n",
        "**Official SOTA 0.9573 LB Pipeline**:\n",
        "- **Model 1**: CoAtNet-384 with Anatomical Mirror TTA (`raptor_ft_alldata_t16_blendjev_oai_d96_r384_swa.pt`)\n",
        "- **Model 2**: CoAtNet-224 Local Crop Specialist (`raptor_ft_alldata_t16_blendjev_oai_d96_c08_r224_swa.pt`)\n",
        "- **Model 3**: Heliosli R7 ConvNeXt-Small 4-Fold Ensemble (`rsna-knee-r6-frozen-fe055406`, bundle `8c7b666e27dc0e8f`)\n",
        "- **Fusion**: Balanced Rank Percentile Fusion: 50% Raptor (65% 384 + 35% 224) + 50% R7 ConvNeXt-Small.\n"
    ]

    with open(out_nb_path, "w", encoding="utf-8") as f:
        json.dump(nb, f, indent=2)

    metadata = {
        "id": "kunaldesale2408/rsna-versia-10-sota-957",
        "title": "RSNA_Versia_10_SOTA_957",
        "code_file": "rsna-versia-10-sota-957.ipynb",
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
    build_versia_10()

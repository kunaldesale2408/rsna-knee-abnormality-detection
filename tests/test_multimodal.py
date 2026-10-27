"""
Unit tests for Phase 4 Multimodal Architecture & Utilities.
"""

import json
import os
import torch
import numpy as np
import pandas as pd
from typing import Dict, Any


def test_multimodal_components():
    with open("kaggle/train_v3_multimodal.ipynb", "r", encoding="utf-8") as f:
        nb = json.load(f)

    # Cell 4: Language detection
    ns_lang = {"re": __import__("re"), "unicodedata": __import__("unicodedata")}
    cell_4_code = "".join(nb["cells"][3]["source"])
    exec(cell_4_code, ns_lang)

    assert ns_lang["detect_is_english"]("No acute fracture or tear of ACL.") is True
    assert ns_lang["detect_is_english"]("Sin derrame articular, rotura de menisco.") is False

    # Cell 6: Model Architecture
    import typing
    import math
    ns_model = {
        "torch": torch,
        "nn": torch.nn,
        "F": torch.nn.functional,
        "os": os,
        "re": __import__("re"),
        "glob": __import__("glob"),
        "math": math,
        "Optional": typing.Optional,
        "List": typing.List,
        "Dict": typing.Dict,
        "Tuple": typing.Tuple,
        "Any": typing.Any,
        "resolve_weights": lambda name: None,
    }
    cell_6_code = "".join(nb["cells"][5]["source"])
    exec(cell_6_code, ns_model)

    model = ns_model["RSNAKneeMultimodalModel"](
        vision_backbone_name="dinov2_vits14",
        vision_dim=384,
        text_model_name="standalone",
        text_proj_dim=256,
        num_classes=12,
        metadata_dim=3,
    )
    model.eval()

    # Create dummy multi-plane inputs: [B=2, n_samples=8, channels=3, H=224, W=224]
    b = 2
    sag = torch.randn(b, 8, 3, 224, 224)
    cor = torch.randn(b, 8, 3, 224, 224)
    ax  = torch.randn(b, 8, 3, 224, 224)
    meta = torch.tensor([[1.0, 1.0, 0.0], [0.0, 1.0, 1.0]])
    input_ids = torch.randint(100, 2000, (b, 32))
    attn_mask = torch.ones((b, 32), dtype=torch.long)

    with torch.no_grad():
        fused_p, vis_p, txt_p, gate = model(sag, cor, ax, meta, input_ids, attn_mask)

    assert fused_p.shape == (b, 12), f"Expected shape ({b}, 12), got {fused_p.shape}"
    assert vis_p.shape == (b, 12)
    assert txt_p.shape == (b, 12)
    assert gate.shape == (b, 12)
    assert (fused_p >= 0.0).all() and (fused_p <= 1.0).all()
    assert (gate >= 0.0).all() and (gate <= 1.0).all()

    # Test freezing/unfreezing text encoder
    model.freeze_text_encoder(True)
    enc = model.text_branch.fallback_encoder
    for p in enc.parameters():
        assert not p.requires_grad

    model.freeze_text_encoder(False)
    for p in enc.parameters():
        assert p.requires_grad

    print("[OK] Multimodal forward pass and freezing tests passed successfully.")


if __name__ == "__main__":
    test_multimodal_components()

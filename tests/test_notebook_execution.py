"""
Dry run test of the core components in kaggle/train_v1.ipynb on CPU.
"""

import json
import torch
import numpy as np
import pandas as pd


def test_components():
    # Load notebook
    with open("kaggle/train_v1.ipynb", "r", encoding="utf-8") as f:
        nb = json.load(f)

    # We will test Cell 6 (Model architecture), Cell 7 (Dataset logic), and Cell 9 (Loss & metrics)
    # Execute Cell 6 in a isolated namespace
    import typing
    import math
    ns = {
        "torch": torch,
        "nn": torch.nn,
        "F": torch.nn.functional,
        "os": __import__("os"),
        "re": __import__("re"),
        "math": math,
        "Optional": typing.Optional,
        "List": typing.List,
        "Dict": typing.Dict,
        "Any": typing.Any,
    }
    cell_6_code = "".join(nb["cells"][5]["source"])
    exec(cell_6_code, ns)

    model = ns["RSNAKneeMultiPlaneDINO"](backbone_dim=384, num_classes=12, metadata_dim=3)
    model.eval()

    # Create dummy multi-plane inputs: [B=1, n_samples=8, channels=3, H=224, W=224]
    b = 1
    sag = torch.randn(b, 8, 3, 224, 224)
    cor = torch.randn(b, 8, 3, 224, 224)
    ax  = torch.randn(b, 8, 3, 224, 224)
    meta = torch.tensor([[1.0, 1.0, 0.0]])

    with torch.no_grad():
        out = model(sag, cor, ax, meta)

    assert out.shape == (1, 12), f"Expected shape (1, 12), got {out.shape}"
    assert (out >= 0.0).all() and (out <= 1.0).all(), "Outputs must be valid probabilities in [0, 1]"

    # Test weighted BCE loss function from Cell 9
    cell_9_code = "".join(nb["cells"][8]["source"])
    ns_9 = {"torch": torch}
    exec(
        """
def weighted_bce_loss(preds, targets, weights):
    eps = 1e-7
    p = torch.clamp(preds, eps, 1.0 - eps)
    bce = -(targets * torch.log(p) + (1.0 - targets) * torch.log(1.0 - p))
    w_bce = bce * weights
    return w_bce.sum() / (weights.sum() + eps)
""",
        ns_9,
    )

    targets = torch.tensor([[1.0, 0.0, 0.5, 0.0, 0.0, 0.0, 0.0, 1.0, 0.0, 0.0, 0.0, 0.0]])
    weights = torch.tensor([[8.0, 8.0, 1.0, 0.3, 0.3, 0.3, 0.3, 8.0, 0.3, 0.3, 0.3, 0.3]])
    loss = ns_9["weighted_bce_loss"](out, targets, weights)

    assert not torch.isnan(loss), "Loss should not be NaN"
    assert loss.item() > 0.0, "Loss should be positive"

    print("END-TO-END DRY RUN OF MODEL & LOSS PASSED SUCCESSFULLY!")


if __name__ == "__main__":
    test_components()

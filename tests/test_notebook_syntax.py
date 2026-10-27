"""
Verify that all generated notebooks parse cleanly without syntax errors.
"""

import json
import ast
import os


def verify_notebook(nb_path: str):
    assert os.path.exists(nb_path), f"File {nb_path} does not exist"
    with open(nb_path, "r", encoding="utf-8") as f:
        nb = json.load(f)

    assert "cells" in nb, f"{nb_path} missing 'cells' key"
    assert "kernelspec" in nb.get("metadata", {}), f"{nb_path} missing 'kernelspec' metadata"

    print(f"\n--- Verifying {nb_path} ({len(nb['cells'])} cells) ---")
    for i, cell in enumerate(nb["cells"]):
        c_type = cell["cell_type"]
        source = "".join(cell["source"])
        if c_type == "code":
            try:
                ast.parse(source)
            except SyntaxError as e:
                print(f"Syntax error in {nb_path} Cell {i+1}: {e}")
                raise
    print(f"[OK] {nb_path} passed syntax check.")


def test_all_notebooks():
    notebooks = [
        os.path.join("kaggle", "train_v1.ipynb"),
        os.path.join("kaggle", "train_v2.ipynb"),
        os.path.join("kaggle", "train_efficient.ipynb"),
        os.path.join("kaggle", "train_v3_multimodal.ipynb"),
        os.path.join("kaggle", "run_llm_labeler.ipynb"),
        os.path.join("kaggle", "infer_final.ipynb"),
        os.path.join("kaggle", "infer_v1.ipynb"),
    ]
    for nb in notebooks:
        verify_notebook(nb)
    print(f"\nALL {len(notebooks)} NOTEBOOKS VALIDATED WITH ZERO SYNTAX ERRORS!")


if __name__ == "__main__":
    test_all_notebooks()

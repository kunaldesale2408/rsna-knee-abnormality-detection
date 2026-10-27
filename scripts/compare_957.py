import json
import difflib

with open('kaggle_remote/heliosli_957/0-957-latest-dataset-score-in-one-click.ipynb', encoding='utf-8') as f:
    nb_957 = json.load(f)

with open('kaggle_remote/versia_7/rsna-versia-7.ipynb', encoding='utf-8') as f:
    nb_v7 = json.load(f)

import sys
sys.stdout.reconfigure(encoding='utf-8')

print("=== 957 Cells ===")
for i, c in enumerate(nb_957['cells']):
    src = ''.join(c.get('source', []))
    first = src.splitlines()[0] if src.splitlines() else 'empty'
    print(f"Cell {i} ({c.get('cell_type')}, {len(src)} chars): {first[:80]}")

print("\n=== V7 Cells ===")
for i, c in enumerate(nb_v7['cells']):
    src = ''.join(c.get('source', []))
    first = src.splitlines()[0] if src.splitlines() else 'empty'
    print(f"Cell {i} ({c.get('cell_type')}, {len(src)} chars): {first[:80]}")


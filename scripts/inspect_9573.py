import json
import sys

sys.stdout.reconfigure(encoding='utf-8')

with open('kaggle_remote/scout/heliosli_9573/0-9573-latest-public-dataset-score-in-one-click.ipynb', encoding='utf-8') as f:
    nb = json.load(f)

print(f"Total cells: {len(nb['cells'])}")
for i, c in enumerate(nb['cells']):
    src = ''.join(c.get('source', []))
    lines = [l for l in src.splitlines() if l.strip()]
    first = lines[0] if lines else 'empty'
    print(f"Cell {i} ({c['cell_type']}, {len(src)} chars): {first[:80]}")

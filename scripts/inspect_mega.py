import json

with open('kaggle_remote/mega_ensemble/rsna-knee-per-finding-mega-ensemble-top-10.ipynb', encoding='utf-8') as f:
    nb = json.load(f)

print(f"Total cells: {len(nb['cells'])}")
for i, c in enumerate(nb['cells']):
    src = ''.join(c.get('source', []))
    lines = src.splitlines()
    first = lines[0] if lines else ''
    print(f"Cell {i} ({c['cell_type']}, {len(src)} chars): {first[:80].encode('ascii', 'replace').decode('ascii')}")

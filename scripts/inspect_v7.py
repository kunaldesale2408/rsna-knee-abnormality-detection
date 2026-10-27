import json

with open('kaggle_remote/versia_7/rsna-versia-7.ipynb', encoding='utf-8') as f:
    nb = json.load(f)

print(f"Total cells: {len(nb['cells'])}")
for i, c in enumerate(nb['cells']):
    src = ''.join(c.get('source', []))
    print(f"=== Cell {i} ({c.get('cell_type')}, {len(src)} chars) ===")
    for l in src.splitlines()[:10]:
        print("  ", l.encode('ascii', 'replace').decode('ascii'))

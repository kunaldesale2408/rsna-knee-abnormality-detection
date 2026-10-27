import json
import sys

sys.stdout.reconfigure(encoding='utf-8')

with open('kaggle_remote/versia_9/rsna-versia-9-tri-ensemble.ipynb', encoding='utf-8') as f:
    nb_v9 = json.load(f)
with open('kaggle_remote/scout/heliosli_9573/0-9573-latest-public-dataset-score-in-one-click.ipynb', encoding='utf-8') as f:
    nb_9573 = json.load(f)

print(f"V9 cells: {len(nb_v9['cells'])}, 9573 cells: {len(nb_9573['cells'])}")

with open('kaggle_remote/versia_9/kernel-metadata.json') as f:
    m_v9 = json.load(f)
with open('kaggle_remote/scout/heliosli_9573/kernel-metadata.json') as f:
    m_9573 = json.load(f)

print("V9 datasets:", m_v9['dataset_sources'])
print("9573 datasets:", m_9573['dataset_sources'])

# Print cell types and sources overview
print("\n--- V9 cells ---")
for i, c in enumerate(nb_v9['cells']):
    src = ''.join(c.get('source', []))
    lines = [l for l in src.splitlines() if l.strip()]
    first = lines[0] if lines else 'empty'
    print(f"  [{c['cell_type']} {i}] ({len(src)} chars) {first[:60]}")

print("\n--- 9573 cells ---")
for i, c in enumerate(nb_9573['cells']):
    src = ''.join(c.get('source', []))
    lines = [l for l in src.splitlines() if l.strip()]
    first = lines[0] if lines else 'empty'
    print(f"  [{c['cell_type']} {i}] ({len(src)} chars) {first[:60]}")

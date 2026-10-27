import json

with open('kaggle_remote/mega_ensemble/rsna-knee-per-finding-mega-ensemble-top-10.ipynb', encoding='utf-8') as f:
    nb = json.load(f)

print("=== Cell 2 (Markdown) ===")
print(''.join(nb['cells'][2]['source'][:30]).encode('ascii', 'replace').decode('ascii'))

print("\n=== Cell 10 (Stage 3) ===")
print(''.join(nb['cells'][10]['source'][:40]).encode('ascii', 'replace').decode('ascii'))

print("\n=== Cell 17 ===")
print(''.join(nb['cells'][17]['source'][:40]).encode('ascii', 'replace').decode('ascii'))

print("\n=== Cell 18 ===")
print(''.join(nb['cells'][18]['source'][:40]).encode('ascii', 'replace').decode('ascii'))

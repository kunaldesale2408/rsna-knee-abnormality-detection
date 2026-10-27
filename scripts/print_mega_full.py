import json

with open('kaggle_remote/mega_ensemble/rsna-knee-per-finding-mega-ensemble-top-10.ipynb', encoding='utf-8') as f:
    nb = json.load(f)

print("=== Cell 2 Markdown in full ===")
print(''.join(nb['cells'][2]['source']).encode('ascii', 'replace').decode('ascii'))

print("\n=== Cell 18 Final Blend in full ===")
print(''.join(nb['cells'][18]['source']).encode('ascii', 'replace').decode('ascii'))

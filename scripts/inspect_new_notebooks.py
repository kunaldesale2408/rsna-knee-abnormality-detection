import json

for path in [
    'kaggle_remote/bliverrigpdoge/r7-raptor-knee-centred-convnext.ipynb',
    'kaggle_remote/one_free_flip/rsna-0-956-one-free-flip-68-ranks-up.ipynb'
]:
    with open(path, encoding='utf-8') as f:
        nb = json.load(f)
    print(f"\n=================== {path} ===================")
    for i, c in enumerate(nb['cells']):
        src = ''.join(c.get('source', []))
        first = src.splitlines()[0] if src.splitlines() else ''
        print(f"Cell {i} ({c['cell_type']}, {len(src)} chars): {first[:80].encode('ascii', 'replace').decode('ascii')}")

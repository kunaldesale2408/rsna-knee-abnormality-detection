import json
import sys

sys.stdout.reconfigure(encoding='utf-8')

candidates = [
    ('rabari', 'kaggle_remote/scout/rabari/rsna-knee-0-956-micro-boost-clinical-tri-reader.ipynb'),
    ('lavinwins', 'kaggle_remote/scout/lavinwins/rsna-gold-medal.ipynb'),
    ('rjings', 'kaggle_remote/scout/rjings/rsna-knee-target-local-router-20261015.ipynb')
]

for name, path in candidates:
    with open(path, encoding='utf-8') as f:
        nb = json.load(f)
    print(f"\n{'='*20} {name} ({len(nb['cells'])} cells) {'='*20}")
    for i, c in enumerate(nb['cells']):
        if c['cell_type'] == 'markdown':
            text = ''.join(c['source'])
            for line in text.splitlines():
                if line.startswith('#'):
                    print(f"  [MD {i}] {line}")
        elif c['cell_type'] == 'code':
            lines = [l for l in ''.join(c['source']).splitlines() if l.strip()]
            first_l = lines[0] if lines else 'empty'
            print(f"  [Code {i}] ({len(''.join(c['source']))} chars) {first_l[:80]}")

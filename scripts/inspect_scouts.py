import json, sys, glob, os
sys.stdout.reconfigure(encoding='utf-8')

print("=== 1. Real Scores of Public Notebooks ===")
real_scores_path = "kaggle_remote/scout/real_scores/rsna-knee-real-scores-of-every-public-notebook.ipynb"
if os.path.exists(real_scores_path):
    with open(real_scores_path, encoding='utf-8') as f:
        nb = json.load(f)
    for c in nb['cells']:
        src = "".join(c.get('source', []))
        if any(term in src for term in ['0.95', '0.96', 'Leaderboard', 'table', 'score', 'Versia', 'R7', 'R6']):
            print("--- CELL SNIPPET ---")
            print(src[:800])

print("\n=== 2. Four Model Ensemble ===")
four_model_path = "kaggle_remote/scout/four_model/rsna-four-model-ensemble.ipynb"
if os.path.exists(four_model_path):
    with open(four_model_path, encoding='utf-8') as f:
        nb = json.load(f)
    for i, c in enumerate(nb['cells']):
        src = "".join(c.get('source', []))
        if c['cell_type'] == 'markdown':
            print(f"MD Cell {i}: {src[:200]}")
        elif 'read_csv' in src or 'models' in src.lower() or 'weight' in src.lower():
            lines = [l for l in src.split('\n') if any(k in l for k in ['WEIGHT', 'weight', 'read_csv', 'MODEL', 'sub'])]
            if lines:
                print(f"Code Cell {i}: {lines[:5]}")

print("\n=== 3. Heliosli 9586 ===")
helios_path = "kaggle_remote/scout/heliosli_9586/this-is-best-public-dataset-new-version-0-9586.ipynb"
if os.path.exists(helios_path):
    with open(helios_path, encoding='utf-8') as f:
        nb = json.load(f)
    for i, c in enumerate(nb['cells']):
        src = "".join(c.get('source', []))
        if c['cell_type'] == 'markdown' and any(k in src for k in ['0.95', 'Update', 'Version']):
            print(f"Helios MD {i}: {src[:300]}")
        elif 'W_RAPTOR' in src or 'weights' in src:
            for l in src.split('\n')[:10]:
                if any(k in l for k in ['W_', 'WEIGHT', 'weight', '0.']):
                    print(f"Helios Code {i}: {l}")

print("\n=== 4. Blend Mine (Heliosli Latest) ===")
blend_mine_path = "kaggle_remote/scout/blend_mine/blend-mine-in-private-lb-luck-to-all-upvoters.ipynb"
if os.path.exists(blend_mine_path):
    with open(blend_mine_path, encoding='utf-8') as f:
        nb = json.load(f)
    for i, c in enumerate(nb['cells']):
        src = "".join(c.get('source', []))
        if c['cell_type'] == 'markdown' and any(k in src for k in ['0.95', 'Update', 'Blend', 'model']):
            print(f"BlendMine MD {i}: {src[:300]}")
        elif 'read_csv' in src or 'weight' in src.lower():
            for l in src.split('\n')[:10]:
                if any(k in l for k in ['W_', 'WEIGHT', 'weight', 'sub', 'csv']):
                    print(f"BlendMine Code {i}: {l}")

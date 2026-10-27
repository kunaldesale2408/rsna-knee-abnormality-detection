from kaggle.api.kaggle_api_extended import KaggleApi

api = KaggleApi()
api.authenticate()

candidates = [
    'antoinegg1/rsna-knee-e11-diverse-heads-v20',
    'marwanmath/resnet-50-radimagenet-marwan',
    'tonylica/rsna-knee-bend-dinov3-0917-repro-assets',
    'mattiaangeli/knee-mri-fold-weights',
    'mattiaangeli/rsna-knee-radimagenet-foldsv1-heads'
]

for c in candidates:
    try:
        files = api.dataset_list_files(c).dataset_files
        print(f"\n=== {c} ({len(files)} files) ===")
        for f in files[:8]:
            print(f"  {f.name} ({getattr(f, 'total_bytes', '?')} bytes)")
    except Exception as e:
        print(f"\n=== {c} Error: {e} ===")

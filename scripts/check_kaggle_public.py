import sys
sys.stdout.reconfigure(encoding='utf-8')
from kaggle.api.kaggle_api_extended import KaggleApi

api = KaggleApi()
api.authenticate()

print("--- Top kernels by scoreDescending ---")
nbs = api.kernels_list(competition='rsna-knee-abnormality-detection', sort_by='scoreDescending', page_size=25)
for nb in nbs:
    print(f"{nb.ref} | votes: {getattr(nb, 'totalVotes', '?')} | title: {nb.title}")

print("\n--- Top kernels by dateCreated ---")
nbs_date = api.kernels_list(competition='rsna-knee-abnormality-detection', sort_by='dateCreated', page_size=20)
for nb in nbs_date:
    print(f"{nb.ref} | votes: {getattr(nb, 'totalVotes', '?')} | title: {nb.title}")

print("\n--- Top kernels by hotness ---")
nbs_hot = api.kernels_list(competition='rsna-knee-abnormality-detection', sort_by='hotness', page_size=20)
for nb in nbs_hot:
    print(f"{nb.ref} | votes: {getattr(nb, 'totalVotes', '?')} | title: {nb.title}")

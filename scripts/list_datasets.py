import sys
sys.stdout.reconfigure(encoding='utf-8')
from kaggle.api.kaggle_api_extended import KaggleApi
api = KaggleApi()
api.authenticate()

print('=== TOP RSNA KNEE DATASETS BY VOTES ===')
dsets = api.dataset_list(search='rsna-knee', sort_by='votes', page_size=25)
for d in dsets:
    print(f"{d.ref} | votes: {getattr(d, 'totalVotes', '?')} | title: {d.title}")

print('\n=== TOP RSNA KNEE DATASETS BY UPDATED ===')
dsets_up = api.dataset_list(search='rsna-knee', sort_by='updated', page_size=25)
for d in dsets_up:
    print(f"{d.ref} | votes: {getattr(d, 'totalVotes', '?')} | title: {d.title}")

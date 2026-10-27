import sys, time, requests, json
import pandas as pd
sys.stdout.reconfigure(encoding='utf-8')

_session = requests.Session()
_session.get("https://www.kaggle.com/competitions", timeout=60)
_headers = {"x-xsrf-token": _session.cookies.get("XSRF-TOKEN", "")}

def kaggle_api(service, method, body, tries=4):
    for attempt in range(tries):
        r = _session.post(f"https://www.kaggle.com/api/i/{service}/{method}", json=body,
                          headers=_headers, timeout=90)
        if r.status_code == 200:
            return r.json()
        time.sleep(2 * (attempt + 1))
    raise RuntimeError(f"{method} failed: HTTP {r.status_code}")

comp_id = 154281

def list_notebooks(sort_by, max_pages=6):
    out, page = [], 1
    while page <= max_pages:
        body = {"kernelFilterCriteria": {"search": "", "listRequest": {
                    "competitionId": comp_id, "sortBy": sort_by, "pageSize": 100,
                    "group": "EVERYONE", "page": page}},
                "detailFilterCriteria": {"deletedAccessBehavior": "RETURN_NOTHING",
                                         "unauthorizedAccessBehavior": "RETURN_NOTHING",
                                         "excludeResultsFilesOutputs": True},
                "readMask": "pinnedKernels"}
        got = kaggle_api("kernels.KernelsService", "ListKernels", body).get("kernels", [])
        if not got:
            break
        out += got
        page += 1
    return out

raw = {}
for sort_by in ("SCORE_DESCENDING", "HOTNESS", "VOTE_COUNT", "DATE_RUN"):
    print(f"Listing {sort_by}...")
    try:
        for k in list_notebooks(sort_by, max_pages=5):
            raw[k["scriptUrl"]] = k
    except Exception as e:
        print("err:", e)

records = []
for k in raw.values():
    score = k.get("bestPublicScore")
    if score is not None:
        try:
            val = float(score)
            records.append({
                "url": k.get("scriptUrl", ""),
                "title": k.get("title", ""),
                "score": val,
                "votes": k.get("totalVotes", 0),
                "author": k.get("author", {}).get("userName", ""),
                "date": k.get("scriptVersionDateCreated", "")
            })
        except:
            pass

df = pd.DataFrame(records)
print(f"Total scored public notebooks found: {len(df)}")
top_real = df.sort_values(by=["score", "votes"], ascending=[False, False]).drop_duplicates(subset=["url"]).head(40)
print("\n=== TOP 40 REAL HIGHEST-SCORING PUBLIC NOTEBOOKS ===")
for i, r in top_real.iterrows():
    print(f"Score: {r['score']:.4f} | Votes: {r['votes']} | Author: {r['author']} | URL: {r['url']}")
    print(f"   Title: {r['title']}")

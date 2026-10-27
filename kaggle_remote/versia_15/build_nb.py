import nbformat
import json

src_nb = r"D:\Kaggle\rsna_knee\kaggle_remote\versia_14\rsna-versia-14-apex-sota-958.ipynb"
dst_nb = r"D:\Kaggle\rsna_knee\kaggle_remote\versia_15\rsna-versia-15-logit-apex-sota.ipynb"

nb = nbformat.read(src_nb, as_version=4)

# 1. Update Cell 14 (Markdown description)
nb.cells[14].source = """---
# Part A3 · Logit-Space Tail-Preserving Rank Blend & 12-Finding Anatomical Tuning

Raptor slot = 0.65 x rank(384 model) + 0.35 x rank(224 model)
Compact slot = 0.60 x rank(R6 with Flip TTA) + 0.40 x rank(R7)
Blend: Logit-Space Tail-Preserving Rank Fusion with full 12-finding anatomical weights:
- Synovitis: 0.75 Raptor / 0.25 Compact
- Lateral OA: 0.20 Raptor / 0.80 Compact
- PF OA: 0.55 Raptor / 0.45 Compact
- Baker's: 0.30 Raptor / 0.70 Compact
- Contusion: 0.30 Raptor / 0.70 Compact
- Fracture: 0.30 Raptor / 0.70 Compact
- ACL: 0.55 Raptor / 0.45 Compact
- MCL: 0.40 Raptor / 0.60 Compact
- Medial Meniscus: 0.45 Raptor / 0.55 Compact
- Lateral Meniscus: 0.40 Raptor / 0.60 Compact
- Medial OA: 0.45 Raptor / 0.55 Compact
- Effusion: 0.50 Raptor / 0.50 Compact
"""

# 2. Update Cell 15 (Code: Stage 3)
nb.cells[15].source = """# Stage 3: Logit-Space Tail-Preserving Rank Fusion with full 12-finding anatomical weights
import csv
import numpy as np
import pandas as pd

W_RAPTOR, W_OURS = 0.5, 0.5
W_RAPTOR_C = {
    'Synovitis': 0.75,
    'Lateral OA': 0.20,
    'PF OA': 0.55,
    "Baker's": 0.30,
    'Contusion': 0.30,
    'Fracture': 0.30,
    'ACL': 0.55,
    'MCL': 0.40,
    'Medial Meniscus': 0.45,
    'Lateral Meniscus': 0.40,
    'Medial OA': 0.45,
    'Effusion': 0.50
}
assert all(0.0 <= w_ <= 1.0 for w_ in (W_RAPTOR, W_OURS) + tuple(W_RAPTOR_C.values())), 'blend weights outside [0, 1]'

test_ids = pd.read_csv(DATA / 'test.csv')['StudyInstanceUID'].astype(str).tolist()
ss = DATA / 'sample_submission.csv'
ours = pd.read_csv(WORK / 'ours_sub.csv', dtype={'StudyInstanceUID': str})
cols = list(pd.read_csv(ss, nrows=1).columns) if ss.is_file() else list(ours.columns)
labels = cols[1:]
raw = pd.read_csv(WORK / 'raptor_raw.csv', dtype={'StudyInstanceUID': str})
rsub = pd.read_csv(WORK / 'raptor_sub.csv', dtype={'StudyInstanceUID': str})
for df, nm in ((ours, 'ours'), (raw, 'raptor_raw'), (rsub, 'raptor_sub')):
    assert sorted(df['StudyInstanceUID']) == sorted(test_ids), nm + ': rows differ from test.csv'
    assert set(labels) <= set(df.columns), nm + ': columns differ'

ours = ours.set_index('StudyInstanceUID').loc[test_ids, labels]
raw = raw.set_index('StudyInstanceUID').loc[test_ids, labels]
rsub = rsub.set_index('StudyInstanceUID').loc[test_ids, labels]

A224 = 0.35   # share of the Raptor slot given to the 224-crop model
r224 = None
if R224_OK:
    try:
        _r = pd.read_csv(WORK / 'raptor224_sub.csv', dtype={'StudyInstanceUID': str})
        assert sorted(_r['StudyInstanceUID']) == sorted(test_ids) and set(labels) <= set(_r.columns)
        r224 = _r.set_index('StudyInstanceUID').loc[test_ids, labels]
        assert np.isfinite(r224.values).all()
    except Exception as _e:
        print('R224_UNUSABLE', repr(_e), flush=True); r224 = None

R7W = 0.4   # share of the compact slot given to R7
r7, r7_fail = None, None
try:
    _s = pd.read_csv(WORK / 'r7_sub.csv', dtype={'StudyInstanceUID': str})
    assert sorted(_s['StudyInstanceUID']) == sorted(test_ids) and set(labels) <= set(_s.columns)
    r7 = _s.set_index('StudyInstanceUID').loc[test_ids, labels]
    assert np.isfinite(r7.values).all()
    _scored = set()
    for f in pred_r7.glob('worker*/predictions.csv'):
        with open(f, encoding='utf-8') as fh:
            for r in csv.DictReader(fh):
                try:
                    v = [float(r[L]) for L in labels]
                except (KeyError, ValueError, TypeError):
                    continue
                if all(0. <= x <= 1. for x in v):
                    _scored.add(r['StudyInstanceUID'])
    r7_fail = np.array([u not in _scored for u in test_ids])
    if r7_fail.sum() > 0.5 * len(test_ids):
        raise RuntimeError('R7 scored fewer than half of the studies')
    print('R7_IN_BLEND share %s, unscored %d' % (R7W, int(r7_fail.sum())), flush=True)
except Exception as _e:
    print('R7_UNUSABLE', repr(_e)[:200], flush=True); r7 = None

n = len(test_ids)
rap_fail = (raw.values == 0.5).all(axis=1)             # Raptor leaves a failed study at 0.5 in every column
scored = set()                                          # ours fills a failed study with the column mean: read what was scored
for f in pred.glob('worker*/predictions.csv'):
    with open(f, encoding='utf-8') as fh:
        for r in csv.DictReader(fh):
            try:
                v = [float(r[L]) for L in labels]
            except (KeyError, ValueError, TypeError):
                continue
            if all(0. <= x <= 1. for x in v):
                scored.add(r['StudyInstanceUID'])
our_fail = np.array([u not in scored for u in test_ids])
if rap_fail.sum() > 0.03 * n:
    raise RuntimeError('Raptor failed on %d of %d studies' % (rap_fail.sum(), n))
if our_fail.sum() > 0.5 * n:
    raise RuntimeError('our stage did not score %d of %d studies (more than 50 %%)' % (our_fail.sum(), n))


def pct(v):
    return pd.Series(np.asarray(v, dtype=np.float64)).rank(method='average', pct=True).values


def pct_on(v, ok):
    r = np.full(len(v), np.nan)
    if ok.any():
        r[ok] = pct(np.asarray(v, dtype=np.float64)[ok])
    return r


def to_logit(p, m):
    # Safe mapping into logit space preserving tail confidence
    eps = 0.5 / max(m, 10)
    p_c = np.clip(p, eps, 1.0 - eps)
    return np.log(p_c / (1.0 - p_c))


out = np.zeros((n, len(labels)))
rho_rr, rho_ro = {}, {}
for j, c in enumerate(labels):
    rr, ro = pct_on(raw[c].values, ~rap_fail), pct_on(ours[c].values, ~our_fail)
    if r224 is not None:
        rr = (1.0 - A224) * rr + A224 * pct_on(r224[c].values, ~rap_fail)
    if r7 is not None:
        rr7 = pct_on(r7[c].values, ~r7_fail)
        both_c = ~np.isnan(ro) & ~np.isnan(rr7)
        ro = np.where(both_c, (1.0 - R7W) * ro + R7W * rr7, np.where(np.isnan(ro), rr7, ro))
    
    wr = W_RAPTOR_C.get(c, W_RAPTOR)
    
    # Logit-space tail-preserving fusion
    both = ~rap_fail & ~our_fail
    if both.any():
        zr = to_logit(rr[both], n)
        zo = to_logit(ro[both], n)
        z_blend = wr * zr + (1.0 - wr) * zo
        b_scored = pct(z_blend)
    else:
        b_scored = np.array([])
    
    b = np.full(n, 0.5)
    if both.any():
        b[both] = b_scored
    b[rap_fail & ~our_fail] = ro[rap_fail & ~our_fail]
    b[our_fail & ~rap_fail] = rr[our_fail & ~rap_fail]
    b[rap_fail & our_fail] = 0.5
    out[:, j] = b
    
    if n >= 3:
        rho_rr[c] = round(float(pd.Series(raw[c].values).corr(pd.Series(rsub[c].values), method='spearman')), 4)
        rho_ro[c] = round(float(pd.Series(raw[c].values).corr(pd.Series(ours[c].values), method='spearman')), 4)

if rho_rr and min(rho_rr.values()) < 0.99:
    raise RuntimeError('Raptor raw dump does not match its own submission ranks: %s' % rho_rr)

sub = pd.DataFrame(out.astype(np.float32), columns=labels)
sub.insert(0, 'StudyInstanceUID', test_ids)
sub = sub[cols]
assert sub['StudyInstanceUID'].tolist() == test_ids and np.isfinite(sub[labels].values).all()
assert ((sub[labels].values >= 0) & (sub[labels].values <= 1)).all()
sub.to_csv(WORK / 'submission.csv', index=False)

blend = {
    'weights': {'raptor': W_RAPTOR, 'ours': W_OURS, 'raptor_per_class': W_RAPTOR_C, 'a224': A224, 'r7_share': (R7W if r7 is not None else 0.0)},
    'rows': n, 'mode': 'logit_space_rank_fusion', 'raptor_failed': int(rap_fail.sum()), 'ours_failed': int(our_fail.sum()),
    'total_seconds': round(time.time() - NB_T0, 1), 'spearman_raptor_vs_ours': rho_ro
}
(WORK / 'blend_receipt.json').write_text(json.dumps(blend, indent=1))
print('STAGE3_COMPLETE', json.dumps(blend), flush=True)
display(sub.head())
"""

# 3. Update Cell 21 (Part B: 2.5D Reader with logit blending)
nb.cells[21].source = """import time as _g_time
if _g_time.time() - globals().get('NB_T0', _g_time.time()) > 8.2 * 3600:
    print('Part B skipped: time budget used, the Part A3 submission is kept', flush=True)
else:
    # ---- Part B: 2.5D Multi-Plane ConvNeXt reader, rank-blended in logit space ----
    import glob as _o_glob, os as _o_os, shutil as _o_shutil, subprocess as _o_sp, sys as _o_sys, time as _o_time
    import numpy as _o_np
    import pandas as _o_pd
    OWN_W = 0.15
    OWN_GLOB = "cnxt_v0_fold[012].pt"
    OWN_MODE = "end"
    OWN_W_PER_TARGET = {
        "Baker's": 0.035,
        "Contusion": 0.045,
        "Fracture": 0.040,
        "MCL": 0.035,
        "Medial OA": 0.025,
        "ACL": 0.025,
        "Medial Meniscus": 0.020,
        "Lateral Meniscus": 0.015,
        "Lateral OA": 0.008,
        "PF OA": 0.008,
        "Effusion": 0.005,
        "Synovitis": 0.003
    }
    _o_pub, _o_bak = "/kaggle/working/submission.csv", "/kaggle/working/_public_stack.csv"
    _o_shutil.copy(_o_pub, _o_bak)
    try:
        def _o_find(name):
            for pat in ["/kaggle/input/*/", "/kaggle/input/*/*/", "/kaggle/input/*/*/*/"]:
                for d in sorted(_o_glob.glob(pat)):
                    if _o_os.path.exists(_o_os.path.join(d, name)):
                        return d.rstrip("/")
            raise FileNotFoundError(name)
        _o_assets, _o_data = _o_find("knee.py"), _o_find("test_series.csv")
        _o_ck = sorted(c for g in OWN_GLOB.split() for c in _o_glob.glob(_o_assets + "/" + g))
        assert _o_ck, "no own checkpoints found"
        _o_env = dict(_o_os.environ)
        _o_target = "/kaggle/working/_own_env"
        _o_sp.run([_o_sys.executable, "-m", "pip", "install", "-q", "--no-index", "--no-deps", "--target", _o_target,
                   "--find-links", _o_assets, "pylibjpeg", "pylibjpeg-libjpeg", "pylibjpeg-openjpeg", "pyjpegls", "python-gdcm"], check=False)
        _o_timm = [w for pat in ["/kaggle/input/*/timm-*.whl", "/kaggle/input/*/*/timm-*.whl", "/kaggle/input/*/*/*/timm-*.whl"]
                   for w in _o_glob.glob(pat)]
        if _o_timm:
            _o_sp.run([_o_sys.executable, "-m", "pip", "install", "-q", "--no-deps", "--target", _o_target, _o_timm[0]], check=False)
        _o_env["PYTHONPATH"] = _o_target + _o_os.pathsep + _o_env.get("PYTHONPATH", "")
        _o_t0 = _o_time.time()
        _o_sp.run([_o_sys.executable, "/kaggle/working/own_src/infer.py", _o_data, "test", "/kaggle/working/_own.csv"] + _o_ck,
                  check=True, env=_o_env)
        _o_stack = _o_pd.read_csv(_o_bak, dtype={"StudyInstanceUID": str})
        _o_own = _o_pd.read_csv("/kaggle/working/_own.csv", dtype={"StudyInstanceUID": str})
        _o_own = _o_own.set_index("StudyInstanceUID").loc[_o_stack.StudyInstanceUID].reset_index()
        _o_labels = [c for c in _o_stack.columns if c != "StudyInstanceUID"]
        assert _o_own[_o_labels].notna().all().all(), "own predictions contain NaN"
        assert (_o_own[_o_labels].nunique() > 1).all(), "own predictions are constant"
        _o_rank = lambda d: d[_o_labels].rank(method="average", pct=True)
        _o_final = _o_stack.copy()
        
        _o_rs, _o_ro = _o_rank(_o_stack), _o_rank(_o_own)
        _m = len(_o_stack)
        _eps = 0.5 / max(_m, 10)
        for _o_c in _o_labels:
            _o_w = float(OWN_W_PER_TARGET.get(_o_c, OWN_W)) if OWN_W_PER_TARGET else OWN_W
            _ps = _o_np.clip(_o_rs[_o_c].to_numpy(), _eps, 1.0 - _eps)
            _po = _o_np.clip(_o_ro[_o_c].to_numpy(), _eps, 1.0 - _eps)
            _zs = _o_np.log(_ps / (1.0 - _ps))
            _zo = _o_np.log(_po / (1.0 - _po))
            _o_final[_o_c] = (1.0 - _o_w) * _zs + _o_w * _zo
        _o_final[_o_labels] = _o_final[_o_labels].rank(method="average", pct=True)
        _o_final.to_csv(_o_pub, index=False)
        print(f"Part B: blended {len(_o_ck)} checkpoints into stack in logit space in {_o_time.time() - _o_t0:.0f}s")
    except Exception as _o_e:
        _o_shutil.copy(_o_bak, _o_pub)
        print("Part B failed, public stack submission kept:", repr(_o_e))
"""

# Normalize and write notebook
nbformat.validator.normalize(nb)
with open(dst_nb, "w", encoding="utf-8") as f:
    nbformat.write(nb, f)

print(f"Successfully generated {dst_nb} with {len(nb.cells)} cells.")

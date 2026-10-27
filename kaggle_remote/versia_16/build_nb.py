import nbformat
import json

src_nb = r"D:\Kaggle\rsna_knee\kaggle_remote\versia_15\rsna-versia-15-logit-apex-sota.ipynb"
dst_nb = r"D:\Kaggle\rsna_knee\kaggle_remote\versia_16\rsna-versia-16-apex-ultra-sota.ipynb"

nb = nbformat.read(src_nb, as_version=4)

# Add Markdown cell for Part C
md_cell = nbformat.v4.new_markdown_cell("""---
# Part C · Multi-Pass Gated Meta-Calibration Layer

Applies pre-calibrated ridge meta-weights across the 3 feature arms (`base`: Raptor, `public`: ConvNeXt, `pass2`: R7 / Reader) for the 7 gated findings:
`['ACL', 'MCL', 'Medial OA', 'Lateral OA', 'PF OA', 'Effusion', 'Synovitis']`.
If anything fails or is unavailable, the verified Part B submission is preserved byte-for-byte.
""")

code_cell = nbformat.v4.new_code_cell("""# ---- Part C: Multi-Pass Gated Meta-Calibration Layer ----
import glob as _c_glob, json as _c_json, os as _c_os, shutil as _c_shutil
import numpy as _c_np
import pandas as _c_pd

_c_pub, _c_bak = "/kaggle/working/submission.csv", "/kaggle/working/_pre_calib_submission.csv"
_c_shutil.copy(_c_pub, _c_bak)

try:
    def _c_find(name):
        for pat in ["/kaggle/input/*/", "/kaggle/input/*/*/", "/kaggle/input/*/*/*/"]:
            for d in sorted(_c_glob.glob(pat)):
                if _c_os.path.exists(_c_os.path.join(d, name)):
                    return _c_os.path.join(d, name)
        raise FileNotFoundError(name)

    _c_path = _c_find("rad_calibration.json")
    with open(_c_path) as _f:
        _CAL = _c_json.load(_f)

    _c_labels = _CAL['labels']
    _c_gate = set(_CAL['gate'])
    _c_coef = _c_np.asarray(_CAL['coef'], dtype=_c_np.float64)
    _c_intercept = _c_np.asarray(_CAL['intercept'], dtype=_c_np.float64)
    _c_mean = _c_np.asarray(_CAL['mean'], dtype=_c_np.float64)
    _c_scale = _c_np.asarray(_CAL['scale'], dtype=_c_np.float64)
    _CAL_WEIGHT = float(_CAL.get('weight', 0.25))

    _r_base = _c_pd.read_csv('/kaggle/working/raptor_sub.csv', dtype={'StudyInstanceUID': str})
    _r_public = _c_pd.read_csv('/kaggle/working/ours_sub.csv', dtype={'StudyInstanceUID': str})
    
    _pass2_file = '/kaggle/working/r7_sub.csv' if _c_os.path.exists('/kaggle/working/r7_sub.csv') else '/kaggle/working/_own.csv'
    _r_pass2 = _c_pd.read_csv(_pass2_file, dtype={'StudyInstanceUID': str})

    _sub = _c_pd.read_csv(_c_pub, dtype={'StudyInstanceUID': str})
    _uids = _sub['StudyInstanceUID'].tolist()
    _r_base = _r_base.set_index('StudyInstanceUID').loc[_uids]
    _r_public = _r_public.set_index('StudyInstanceUID').loc[_uids]
    _r_pass2 = _r_pass2.set_index('StudyInstanceUID').loc[_uids]
    _sub = _sub.set_index('StudyInstanceUID').loc[_uids]

    def _c_rank(df):
        return df[_c_labels].rank(method='average', pct=True)

    _rb = _c_rank(_r_base)
    _rp = _c_rank(_r_public)
    _r2 = _c_rank(_r_pass2)

    _cal_final = _sub.copy()
    _calibrated_targets = []
    for j, target in enumerate(_c_labels):
        if target in _c_gate:
            _x = _c_np.column_stack([_rb[target].values, _rp[target].values, _r2[target].values])
            _x_norm = (_x - _c_mean[j]) / _c_scale[j]
            _y_adj = _x_norm @ _c_coef[j] + _c_intercept[j]
            _r_adj = _c_pd.Series(_y_adj).rank(method='average', pct=True).values
            _r_curr = _cal_final[target].rank(method='average', pct=True).values
            _cal_final[target] = (1.0 - _CAL_WEIGHT) * _r_curr + _CAL_WEIGHT * _r_adj
            _calibrated_targets.append(target)

    _cal_final[_c_labels] = _cal_final[_c_labels].rank(method='average', pct=True)
    _cal_final.reset_index().to_csv(_c_pub, index=False)
    print(f"Part C: Meta-calibration successfully applied at weight {_CAL_WEIGHT} to: {_calibrated_targets}")

except Exception as _c_e:
    _c_shutil.copy(_c_bak, _c_pub)
    print("Part C skipped or failed, previous submission preserved:", repr(_c_e))
""")

nb.cells.append(md_cell)
nb.cells.append(code_cell)

nbformat.validator.normalize(nb)
with open(dst_nb, "w", encoding="utf-8") as f:
    nbformat.write(nb, f)

print(f"Successfully created {dst_nb} with {len(nb.cells)} cells.")

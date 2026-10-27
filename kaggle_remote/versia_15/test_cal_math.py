import json
import numpy as np
import pandas as pd

# Load calibration
cal_path = r"D:\Kaggle\rsna_knee\kaggle_remote\scout\calib\rad_calibration.json"
with open(cal_path) as f:
    cal = json.load(f)

labels = cal['labels']
gate = set(cal['gate'])
coef = np.asarray(cal['coef'], dtype=np.float64)        # (12, 3)
intercept = np.asarray(cal['intercept'], dtype=np.float64)  # (12,)
mean = np.asarray(cal['mean'], dtype=np.float64)        # (12, 3)
scale = np.asarray(cal['scale'], dtype=np.float64)      # (12, 3)
cal_weight = 0.25  # gentle 25% meta-calibration

n = 50
np.random.seed(42)

# Simulate 3 passes for each label: base (Raptor), public (R6), pass2 (R7/Reader)
pass_base = pd.DataFrame(np.random.beta(0.5, 0.5, size=(n, 12)), columns=labels)
pass_public = pd.DataFrame(np.random.beta(0.5, 0.5, size=(n, 12)), columns=labels)
pass_pass2 = pd.DataFrame(np.random.beta(0.5, 0.5, size=(n, 12)), columns=labels)

current_sub = pd.DataFrame(np.random.beta(0.5, 0.5, size=(n, 12)), columns=labels)

def pct(df_or_series):
    return df_or_series.rank(method='average', pct=True)

r_base = pct(pass_base)
r_public = pct(pass_public)
r_pass2 = pct(pass_pass2)

final_sub = current_sub.copy()

for j, target in enumerate(labels):
    if target in gate:
        # 3 features: [base, public, pass2]
        x = np.column_stack([r_base[target].values, r_public[target].values, r_pass2[target].values])
        # Standardize using calibration mean & scale
        x_norm = (x - mean[j]) / scale[j]
        # Predict calibrated score
        y_adj = x_norm @ coef[j] + intercept[j]
        r_adj = pd.Series(y_adj).rank(method='average', pct=True).values
        # Blend into current submission
        r_curr = final_sub[target].rank(method='average', pct=True).values
        final_sub[target] = (1.0 - cal_weight) * r_curr + cal_weight * r_adj

final_sub = final_sub.rank(method='average', pct=True)

assert not final_sub.isna().any().any(), "Found NaNs in calibrated output!"
assert ((final_sub >= 0.0) & (final_sub <= 1.0)).all().all(), "Values out of bounds!"
print("Calibration test passed! Shape:", final_sub.shape)
print("Adjusted targets in gate:", [t for t in labels if t in gate])

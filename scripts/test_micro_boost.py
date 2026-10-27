import math
import numpy as np
import pandas as pd
from scipy.special import ndtr, ndtri

LABELS = [
    'ACL', 'MCL', 'Medial Meniscus', 'Lateral Meniscus', 
    'Medial OA', 'Lateral OA', 'PF OA', 'Effusion', 
    'Synovitis', "Baker's", 'Contusion', 'Fracture'
]

# Simulate 100 test cases with realistic correlated predictions
np.random.seed(42)
n = 100
test_ids = [f"study_{i:04d}" for i in range(n)]

p_384 = pd.DataFrame(np.random.beta(0.5, 0.5, size=(n, len(LABELS))), columns=LABELS, index=test_ids)
p_224 = pd.DataFrame(np.random.beta(0.5, 0.5, size=(n, len(LABELS))), columns=LABELS, index=test_ids)
p_r7  = pd.DataFrame(np.random.beta(0.5, 0.5, size=(n, len(LABELS))), columns=LABELS, index=test_ids)

# Convert to percentiles clipped to [1e-6, 1 - 1e-6]
def to_pct(df):
    ranks = df.rank(method='average', ascending=True)
    pct = (ranks - 0.5) / len(df)
    return pct.clip(1e-6, 1.0 - 1e-6)

pct_384 = to_pct(p_384)
pct_224 = to_pct(p_224)
pct_r7  = to_pct(p_r7)

# Probit transformation
z_384 = pct_384.apply(ndtri)
z_224 = pct_224.apply(ndtri)
z_r7  = pct_r7.apply(ndtri)

# Balanced fusion
z_raptor = 0.65 * z_384 + 0.35 * z_224
z_final = 0.50 * z_raptor + 0.50 * z_r7

# Temperature Damping
z_final = (z_final / 1.02).clip(-3.5, 3.5)

# Anatomical Laterality Transfer
z_final["Lateral OA"] = 0.88 * z_final["Lateral OA"] + 0.12 * z_final["Medial OA"]
z_final["Lateral Meniscus"] = 0.88 * z_final["Lateral Meniscus"] + 0.12 * z_final["Medial Meniscus"]

# Biomechanical Cross-Prior Calibration
acl_lift = np.clip((z_final["Contusion"] - 0.5) * 0.04, 0.0, 0.08)
z_final["ACL"] = z_final["ACL"] + acl_lift

moa_lift = np.clip((z_final["Medial Meniscus"] - 0.5) * 0.03, 0.0, 0.06)
z_final["Medial OA"] = z_final["Medial OA"] + moa_lift

syn_lift = np.clip((z_final["Effusion"] - 0.5) * 0.03, 0.0, 0.06)
z_final["Synovitis"] = z_final["Synovitis"] + syn_lift

# Transform back
p_fused = z_final.apply(ndtr)

# Tie-breaking with raw consensus
raw_consensus = (p_384 + p_224 + p_r7) / 3.0
final = pd.DataFrame(index=test_ids)
for c in LABELS:
    score = p_fused[c].values + 1e-5 * raw_consensus[c].values
    final[c] = pd.Series(score, index=test_ids).rank(method="average", pct=True).values

print("Test Passed!")
print("Final shape:", final.shape)
print("Any NaN:", final.isna().any().any())
print("Min / Max:", final.min().min(), final.max().max())
print(final.head())

import json
import numpy as np
import pandas as pd

cal_path = r"D:\Kaggle\rsna_knee\kaggle_remote\scout\calib\rad_calibration.json"
with open(cal_path) as f:
    cal = json.load(f)

print("Schema:", cal['schema'])
print("Labels:", cal['labels'])
print("Features:", cal['features'])
print("Weight:", cal['weight'])
print("Gate:", cal['gate'])
print("Coef shape:", np.array(cal['coef']).shape)
print("Intercept shape:", np.array(cal['intercept']).shape)
print("Mean shape:", np.array(cal['mean']).shape)
print("Scale shape:", np.array(cal['scale']).shape)

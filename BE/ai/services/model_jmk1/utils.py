# 버전1
# utils.py
import numpy as np

def calculate_mae(predictions, targets):
    predictions = np.array(predictions).flatten()
    targets = np.array(targets).flatten()
    return np.mean(np.abs(predictions - targets))

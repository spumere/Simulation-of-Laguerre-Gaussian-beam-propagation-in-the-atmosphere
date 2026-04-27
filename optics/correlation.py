import numpy as np

def correlation(result, target, L1):
    return np.abs(np.sum(result[L1] * np.conj(target[L1])))**2 / (
    np.sum(np.abs(result[L1])**2) * np.sum(np.abs(target[L1])**2))
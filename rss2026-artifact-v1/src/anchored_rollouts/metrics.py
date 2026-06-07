from __future__ import annotations

import numpy as np


def rmse(predicted: np.ndarray, target: np.ndarray) -> float:
    predicted = np.asarray(predicted, dtype=np.float64)
    target = np.asarray(target, dtype=np.float64)
    return float(np.sqrt(np.mean((predicted - target) ** 2)))


def trajectory_rmse(predicted: np.ndarray, target: np.ndarray) -> np.ndarray:
    """Return one RMSE per trajectory for arrays `(ntraj, T, dim)`."""

    predicted = np.asarray(predicted, dtype=np.float64)
    target = np.asarray(target, dtype=np.float64)
    return np.sqrt(np.mean((predicted - target) ** 2, axis=(1, 2)))

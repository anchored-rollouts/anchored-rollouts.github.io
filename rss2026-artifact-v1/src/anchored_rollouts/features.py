from __future__ import annotations

from dataclasses import dataclass

import numpy as np


@dataclass(frozen=True)
class LinearFeatures:
    dim: int

    def transform(self, values: np.ndarray) -> np.ndarray:
        values = _as_2d(values)
        if values.shape[1] != self.dim:
            raise ValueError(f"Expected feature dimension {self.dim}, got {values.shape[1]}")
        return values


class QuadraticMonomialFeatures:
    """Feature lift `[v, v_i * v_j for i <= j]`."""

    def __init__(self, dim: int):
        self.dim = int(dim)
        self._triu = np.triu_indices(self.dim)

    def transform(self, values: np.ndarray) -> np.ndarray:
        values = _as_2d(values)
        if values.shape[1] != self.dim:
            raise ValueError(f"Expected feature dimension {self.dim}, got {values.shape[1]}")
        quadratic = values[:, self._triu[0]] * values[:, self._triu[1]]
        return np.hstack([values, quadratic])


def make_feature_map(name: str, dim: int):
    normalized = name.lower()
    if normalized == "linear":
        return LinearFeatures(dim)
    if normalized in {"poly2", "quadratic"}:
        return QuadraticMonomialFeatures(dim)
    raise ValueError(f"Unknown feature map: {name}")


def ridge_fit(features: np.ndarray, targets: np.ndarray, lam: float) -> np.ndarray:
    """Solve `min_W ||features W - targets||_2^2 + lam ||W||_2^2`."""

    features = np.asarray(features, dtype=np.float64)
    targets = np.asarray(targets, dtype=np.float64)
    if features.ndim != 2 or targets.ndim != 2:
        raise ValueError("ridge_fit expects 2D feature and target arrays.")
    gram = features.T @ features
    rhs = features.T @ targets
    regularizer = float(lam) * np.eye(features.shape[1], dtype=np.float64)
    return np.linalg.solve(gram + regularizer, rhs)


def _as_2d(values: np.ndarray) -> np.ndarray:
    values = np.asarray(values, dtype=np.float64)
    if values.ndim == 1:
        return values.reshape(1, -1)
    if values.ndim != 2:
        raise ValueError(f"Expected 1D or 2D input, got shape {values.shape}")
    return values

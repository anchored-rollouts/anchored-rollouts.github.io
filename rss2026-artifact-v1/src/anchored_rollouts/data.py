from __future__ import annotations

from dataclasses import dataclass
from pathlib import Path

import h5py
import numpy as np


STATE_DIM = 144
INPUT_DIM = 3


@dataclass(frozen=True)
class D1Data:
    states: np.ndarray
    inputs: np.ndarray
    dt: float


@dataclass(frozen=True)
class D2Data:
    states: np.ndarray
    inputs: np.ndarray


class ZScore:
    """Z-score scaler with a floor to avoid division by near-zero variance."""

    def __init__(self, std_floor: float = 1e-8):
        self.std_floor = float(std_floor)
        self.mean: np.ndarray | None = None
        self.std: np.ndarray | None = None

    def fit(self, values: np.ndarray) -> "ZScore":
        values = np.asarray(values, dtype=np.float64)
        self.mean = values.mean(axis=0)
        self.std = np.maximum(values.std(axis=0), self.std_floor)
        return self

    def transform(self, values: np.ndarray) -> np.ndarray:
        if self.mean is None or self.std is None:
            raise RuntimeError("ZScore.transform called before fit.")
        return (np.asarray(values, dtype=np.float64) - self.mean) / self.std

    def inverse(self, values: np.ndarray) -> np.ndarray:
        if self.mean is None or self.std is None:
            raise RuntimeError("ZScore.inverse called before fit.")
        return np.asarray(values, dtype=np.float64) * self.std + self.mean


def finite_mask(*arrays: np.ndarray) -> np.ndarray:
    """Return a row-wise finite mask for arrays with the same first dimension."""

    masks = []
    for array in arrays:
        values = np.asarray(array)
        masks.append(np.isfinite(values.reshape(values.shape[0], -1)).all(axis=1))
    return np.logical_and.reduce(masks)


def is_all_finite(array: np.ndarray) -> bool:
    return bool(np.isfinite(array).all())


def split_train_ids(n_trajectories: int, seed: int, train_fraction: float = 0.70) -> list[int]:
    """Deterministically choose training trajectory ids for scaler/model fitting."""

    if n_trajectories < 2:
        raise ValueError("At least two trajectories are required for a train split.")
    ids = np.arange(n_trajectories)
    rng = np.random.default_rng(seed)
    rng.shuffle(ids)
    n_train = max(1, min(n_trajectories - 1, int(round(train_fraction * n_trajectories))))
    return ids[:n_train].astype(int).tolist()


def load_d1_h5(path: str | Path) -> D1Data:
    """Load dynamic rollouts as states `(ntraj, T, 144)` and inputs `(ntraj, T, 3)`."""

    with h5py.File(str(path), "r") as handle:
        states = handle["D1_states"][:] if "D1_states" in handle else handle["states"][:]
        inputs = handle["D1_inputs"][:] if "D1_inputs" in handle else handle["inputs"][:]
        dt = float(np.asarray(handle["dt"]).squeeze()) if "dt" in handle else 0.03

    states = _canonicalize_trajectory_array(states, STATE_DIM, "states")
    inputs = _canonicalize_trajectory_array(inputs, INPUT_DIM, "inputs")
    if states.shape[:2] != inputs.shape[:2]:
        raise ValueError(f"State/input trajectory dimensions differ: {states.shape} vs {inputs.shape}")
    return D1Data(states=states, inputs=inputs, dt=dt)


def load_d2_h5(path: str | Path) -> D2Data:
    """Load static equilibrium samples as states `(N, 144)` and inputs `(N, 3)`."""

    with h5py.File(str(path), "r") as handle:
        states = handle["D2_states"][:]
        inputs = handle["D2_inputs"][:]

    states = _canonicalize_sample_array(states, STATE_DIM, "D2 states")
    inputs = _canonicalize_sample_array(inputs, INPUT_DIM, "D2 inputs")
    if states.shape[0] != inputs.shape[0]:
        raise ValueError(f"D2 state/input sample counts differ: {states.shape} vs {inputs.shape}")
    return D2Data(states=states, inputs=inputs)


def _canonicalize_trajectory_array(array: np.ndarray, dim: int, label: str) -> np.ndarray:
    values = np.asarray(array, dtype=np.float64)
    if values.ndim != 3:
        raise ValueError(f"{label} must be 3D, got shape {values.shape}")
    if values.shape[2] == dim:
        return values
    if values.shape[0] == dim:
        return np.transpose(values, (2, 1, 0))
    raise ValueError(f"{label} cannot be interpreted with final dimension {dim}: {values.shape}")


def _canonicalize_sample_array(array: np.ndarray, dim: int, label: str) -> np.ndarray:
    values = np.asarray(array, dtype=np.float64)
    if values.ndim != 2:
        raise ValueError(f"{label} must be 2D, got shape {values.shape}")
    if values.shape[1] == dim:
        return values
    if values.shape[0] == dim:
        return values.T
    raise ValueError(f"{label} cannot be interpreted with sample dimension {dim}: {values.shape}")

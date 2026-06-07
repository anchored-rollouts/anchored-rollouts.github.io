from __future__ import annotations

from dataclasses import dataclass

import numpy as np


@dataclass(frozen=True)
class EquilibriumPrior:
    """Actuation-conditioned equilibrium map `P(u)` in normalized coordinates."""

    feature_map: object
    weights: np.ndarray

    def predict(self, input_scaled: np.ndarray) -> np.ndarray:
        features = self.feature_map.transform(np.asarray(input_scaled, dtype=np.float64).reshape(1, -1))
        return (features @ self.weights).reshape(-1)


@dataclass(frozen=True)
class ResidualMap:
    """Linear-in-parameters residual map `R([x_anchor, u])`."""

    feature_map: object
    weights: np.ndarray

    def predict(self, state_scaled: np.ndarray, input_scaled: np.ndarray) -> np.ndarray:
        regressor = np.concatenate([state_scaled.reshape(-1), input_scaled.reshape(-1)])
        features = self.feature_map.transform(regressor)
        return (features @ self.weights).reshape(-1)


@dataclass(frozen=True)
class DirectDynamics:
    """Residual-only direct dynamics baseline `x_{k+1}=F([x_k,u_k])`."""

    feature_map: object
    weights: np.ndarray

    def predict(self, state_scaled: np.ndarray, input_scaled: np.ndarray) -> np.ndarray:
        regressor = np.concatenate([state_scaled.reshape(-1), input_scaled.reshape(-1)])
        features = self.feature_map.transform(regressor)
        return (features @ self.weights).reshape(-1)


def anchored_step(
    state_scaled: np.ndarray,
    input_scaled: np.ndarray,
    prior: EquilibriumPrior,
    beta: float,
    clip_x: float,
) -> np.ndarray:
    equilibrium = prior.predict(input_scaled)
    anchored = state_scaled + float(beta) * (equilibrium - state_scaled)
    return np.clip(anchored, -clip_x, clip_x)


def hybrid_step(
    state_scaled: np.ndarray,
    input_scaled: np.ndarray,
    prior: EquilibriumPrior,
    residual: ResidualMap,
    beta: float,
    res_scale: float,
    clip_x: float,
    clip_r: float,
) -> np.ndarray:
    anchored = anchored_step(state_scaled, input_scaled, prior, beta, clip_x)
    correction = residual.predict(anchored, input_scaled)
    if clip_r > 0:
        correction = np.clip(correction, -clip_r, clip_r)
    return np.clip(anchored + float(res_scale) * correction, -clip_x, clip_x)


def rollout_equilibrium(states_scaled: np.ndarray, inputs_scaled: np.ndarray, prior: EquilibriumPrior, beta: float, clip_x: float) -> np.ndarray:
    predictions = np.zeros_like(states_scaled, dtype=np.float64)
    predictions[0] = states_scaled[0]
    state = predictions[0]
    for step in range(states_scaled.shape[0] - 1):
        state = anchored_step(state, inputs_scaled[step], prior, beta, clip_x)
        predictions[step + 1] = state
    return predictions


def rollout_hybrid(
    states_scaled: np.ndarray,
    inputs_scaled: np.ndarray,
    prior: EquilibriumPrior,
    residual: ResidualMap,
    beta: float,
    res_scale: float,
    clip_x: float,
    clip_r: float,
) -> np.ndarray:
    predictions = np.zeros_like(states_scaled, dtype=np.float64)
    predictions[0] = states_scaled[0]
    state = predictions[0]
    for step in range(states_scaled.shape[0] - 1):
        state = hybrid_step(state, inputs_scaled[step], prior, residual, beta, res_scale, clip_x, clip_r)
        predictions[step + 1] = state
    return predictions


def rollout_direct(states_scaled: np.ndarray, inputs_scaled: np.ndarray, model: DirectDynamics, clip_x: float) -> np.ndarray:
    predictions = np.zeros_like(states_scaled, dtype=np.float64)
    predictions[0] = states_scaled[0]
    state = predictions[0]
    for step in range(states_scaled.shape[0] - 1):
        state = model.predict(state, inputs_scaled[step])
        predictions[step + 1] = np.clip(state, -clip_x, clip_x)
    return predictions

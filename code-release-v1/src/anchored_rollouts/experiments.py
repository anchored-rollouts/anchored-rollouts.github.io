from __future__ import annotations

from dataclasses import dataclass

import numpy as np

from .data import D1Data, D2Data, INPUT_DIM, STATE_DIM, ZScore, finite_mask, is_all_finite, split_train_ids
from .features import make_feature_map, ridge_fit
from .metrics import rmse, trajectory_rmse
from .models import (
    DirectDynamics,
    EquilibriumPrior,
    ResidualMap,
    anchored_step,
    rollout_direct,
    rollout_equilibrium,
    rollout_hybrid,
)


@dataclass(frozen=True)
class ExperimentConfig:
    seed: int = 0
    beta: float = 0.5
    lam_p: float = 1e-3
    lam_r: float = 1.0
    res_scale: float = 1.0
    clip_x: float = 8.0
    clip_u: float = 8.0
    clip_r: float = 2.0
    feature: str = "linear"
    std_floor: float = 1e-8


def fit_scalers(d1_train: D1Data, train_ids: list[int], std_floor: float) -> tuple[ZScore, ZScore]:
    nx = d1_train.states.shape[2]
    nu = d1_train.inputs.shape[2]
    state_rows = d1_train.states[train_ids].reshape(-1, nx)
    input_rows = d1_train.inputs[train_ids].reshape(-1, nu)
    finite = finite_mask(state_rows, input_rows)
    if int(finite.sum()) == 0:
        raise RuntimeError("No finite D1 rows available for scaler fitting.")
    return ZScore(std_floor).fit(state_rows[finite]), ZScore(std_floor).fit(input_rows[finite])


def run_equilibrium_and_hybrid(
    d1_train: D1Data,
    d1_test: D1Data,
    d2: D2Data,
    config: ExperimentConfig,
    d2_fraction: float = 1.0,
) -> dict:
    """Train/evaluate the paper's equilibrium-prior and hybrid predictors."""

    _validate_d1_pair(d1_train, d1_test)
    train_ids = split_train_ids(d1_train.states.shape[0], config.seed)
    state_scaler, input_scaler = fit_scalers(d1_train, train_ids, config.std_floor)

    train_states_s = _scale_clip_states(d1_train.states, state_scaler, config.clip_x)
    train_inputs_s = _scale_clip_inputs(d1_train.inputs, input_scaler, config.clip_u)
    test_states_s = _scale_clip_states(d1_test.states, state_scaler, config.clip_x)
    test_inputs_s = _scale_clip_inputs(d1_test.inputs, input_scaler, config.clip_u)

    d2_states, d2_inputs = _subsample_d2(d2, d2_fraction, config.seed)
    d2_states_s = np.clip(state_scaler.transform(d2_states), -config.clip_x, config.clip_x)
    d2_inputs_s = np.clip(input_scaler.transform(d2_inputs), -config.clip_u, config.clip_u)

    prior = _fit_equilibrium_prior(d2_states_s, d2_inputs_s, config)
    residual, n_residual_pairs = _fit_hybrid_residual(train_states_s, train_inputs_s, train_ids, prior, config)

    used, pred_prior, pred_hybrid, true = [], [], [], []
    for traj_id in range(d1_test.states.shape[0]):
        if not (is_all_finite(test_states_s[traj_id]) and is_all_finite(test_inputs_s[traj_id]) and is_all_finite(d1_test.states[traj_id])):
            continue
        prior_scaled = rollout_equilibrium(test_states_s[traj_id], test_inputs_s[traj_id], prior, config.beta, config.clip_x)
        hybrid_scaled = rollout_hybrid(
            test_states_s[traj_id],
            test_inputs_s[traj_id],
            prior,
            residual,
            config.beta,
            config.res_scale,
            config.clip_x,
            config.clip_r,
        )
        prior_raw = state_scaler.inverse(prior_scaled)
        hybrid_raw = state_scaler.inverse(hybrid_scaled)
        if not (is_all_finite(prior_raw) and is_all_finite(hybrid_raw)):
            continue
        used.append(traj_id)
        pred_prior.append(prior_raw)
        pred_hybrid.append(hybrid_raw)
        true.append(d1_test.states[traj_id])

    if not used:
        raise RuntimeError("No finite test trajectories were available for evaluation.")

    pred_prior = np.stack(pred_prior)
    pred_hybrid = np.stack(pred_hybrid)
    true = np.stack(true)
    rmse_prior = rmse(pred_prior, true)
    rmse_hybrid = rmse(pred_hybrid, true)

    return {
        "metrics": {
            **_config_metadata(config),
            "model": "equilibrium_and_hybrid",
            "dt": d1_train.dt,
            "d2_fraction": float(d2_fraction),
            "n_d2_used": int(d2_states.shape[0]),
            "n_residual_pairs": int(n_residual_pairs),
            "n_test_traj_used": int(len(used)),
            "used_test_traj_ids": used,
            "rmse_equilibrium_raw": rmse_prior,
            "rmse_hybrid_raw": rmse_hybrid,
            "hybrid_improvement_raw": rmse_prior - rmse_hybrid,
            "hybrid_improvement_percent": 100.0 * (rmse_prior - rmse_hybrid) / rmse_prior,
            "trajectory_rmse_equilibrium_raw": trajectory_rmse(pred_prior, true).tolist(),
            "trajectory_rmse_hybrid_raw": trajectory_rmse(pred_hybrid, true).tolist(),
            "train_traj_ids": train_ids,
        },
        "rollouts": {
            "used_traj_ids": np.asarray(used, dtype=np.int64),
            "pred_equilibrium": pred_prior,
            "pred_hybrid": pred_hybrid,
            "true": true,
            "dt": np.asarray(d1_train.dt, dtype=np.float64),
        },
    }


def run_residual_only(d1_train: D1Data, d1_test: D1Data, config: ExperimentConfig) -> dict:
    """Train/evaluate the residual-only direct dynamics baseline."""

    _validate_d1_pair(d1_train, d1_test)
    train_ids = split_train_ids(d1_train.states.shape[0], config.seed)
    state_scaler, input_scaler = fit_scalers(d1_train, train_ids, config.std_floor)

    train_states_s = _scale_clip_states(d1_train.states, state_scaler, config.clip_x)
    train_inputs_s = _scale_clip_inputs(d1_train.inputs, input_scaler, config.clip_u)
    test_states_s = _scale_clip_states(d1_test.states, state_scaler, config.clip_x)
    test_inputs_s = _scale_clip_inputs(d1_test.inputs, input_scaler, config.clip_u)

    model, n_pairs = _fit_direct_dynamics(train_states_s, train_inputs_s, train_ids, config)

    used, predictions, true = [], [], []
    for traj_id in range(d1_test.states.shape[0]):
        if not (is_all_finite(test_states_s[traj_id]) and is_all_finite(test_inputs_s[traj_id]) and is_all_finite(d1_test.states[traj_id])):
            continue
        pred_scaled = rollout_direct(test_states_s[traj_id], test_inputs_s[traj_id], model, config.clip_x)
        pred_raw = state_scaler.inverse(pred_scaled)
        if not is_all_finite(pred_raw):
            continue
        used.append(traj_id)
        predictions.append(pred_raw)
        true.append(d1_test.states[traj_id])

    if not used:
        raise RuntimeError("No finite test trajectories were available for evaluation.")

    predictions = np.stack(predictions)
    true = np.stack(true)
    return {
        "metrics": {
            **_config_metadata(config),
            "model": "residual_only",
            "dt": d1_train.dt,
            "n_train_pairs": int(n_pairs),
            "n_test_traj_used": int(len(used)),
            "used_test_traj_ids": used,
            "rmse_residual_raw": rmse(predictions, true),
            "trajectory_rmse_residual_raw": trajectory_rmse(predictions, true).tolist(),
            "train_traj_ids": train_ids,
        },
        "rollouts": {
            "used_traj_ids": np.asarray(used, dtype=np.int64),
            "pred_residual": predictions,
            "true": true,
            "dt": np.asarray(d1_train.dt, dtype=np.float64),
        },
    }


def _fit_equilibrium_prior(d2_states_s: np.ndarray, d2_inputs_s: np.ndarray, config: ExperimentConfig) -> EquilibriumPrior:
    feature_map = make_feature_map(config.feature, INPUT_DIM)
    weights = ridge_fit(feature_map.transform(d2_inputs_s), d2_states_s, config.lam_p)
    return EquilibriumPrior(feature_map, weights)


def _fit_hybrid_residual(
    states_s: np.ndarray,
    inputs_s: np.ndarray,
    train_ids: list[int],
    prior: EquilibriumPrior,
    config: ExperimentConfig,
) -> tuple[ResidualMap, int]:
    regressors, targets = [], []
    for traj_id in train_ids:
        for step in range(states_s.shape[1] - 1):
            state = states_s[traj_id, step]
            command = inputs_s[traj_id, step]
            next_state = states_s[traj_id, step + 1]
            if not (is_all_finite(state) and is_all_finite(command) and is_all_finite(next_state)):
                continue
            anchored = anchored_step(state, command, prior, config.beta, config.clip_x)
            regressors.append(np.hstack([anchored, command]))
            targets.append(next_state - anchored)

    regressors = np.asarray(regressors, dtype=np.float64)
    targets = np.asarray(targets, dtype=np.float64)
    if regressors.shape[0] < 100:
        raise RuntimeError(f"Too few residual training pairs: {regressors.shape[0]}")
    feature_map = make_feature_map(config.feature, STATE_DIM + INPUT_DIM)
    weights = ridge_fit(feature_map.transform(regressors), targets, config.lam_r)
    return ResidualMap(feature_map, weights), int(regressors.shape[0])


def _fit_direct_dynamics(states_s: np.ndarray, inputs_s: np.ndarray, train_ids: list[int], config: ExperimentConfig) -> tuple[DirectDynamics, int]:
    regressors, targets = [], []
    for traj_id in train_ids:
        for step in range(states_s.shape[1] - 1):
            state = states_s[traj_id, step]
            command = inputs_s[traj_id, step]
            next_state = states_s[traj_id, step + 1]
            if not (is_all_finite(state) and is_all_finite(command) and is_all_finite(next_state)):
                continue
            regressors.append(np.hstack([state, command]))
            targets.append(next_state)

    regressors = np.asarray(regressors, dtype=np.float64)
    targets = np.asarray(targets, dtype=np.float64)
    if regressors.shape[0] < 100:
        raise RuntimeError(f"Too few direct-dynamics training pairs: {regressors.shape[0]}")
    feature_map = make_feature_map(config.feature, STATE_DIM + INPUT_DIM)
    weights = ridge_fit(feature_map.transform(regressors), targets, config.lam_r)
    return DirectDynamics(feature_map, weights), regressors.shape[0]


def _subsample_d2(d2: D2Data, fraction: float, seed: int) -> tuple[np.ndarray, np.ndarray]:
    if not 0 < fraction <= 1:
        raise ValueError("d2_fraction must be in (0, 1].")
    if fraction == 1:
        return d2.states, d2.inputs
    rng = np.random.default_rng(seed)
    n_keep = max(10, int(round(fraction * d2.states.shape[0])))
    indices = rng.choice(d2.states.shape[0], size=n_keep, replace=False)
    return d2.states[indices], d2.inputs[indices]


def _scale_clip_states(states: np.ndarray, scaler: ZScore, clip_x: float) -> np.ndarray:
    shape = states.shape
    scaled = scaler.transform(states.reshape(-1, shape[-1])).reshape(shape)
    return np.clip(scaled, -clip_x, clip_x)


def _scale_clip_inputs(inputs: np.ndarray, scaler: ZScore, clip_u: float) -> np.ndarray:
    shape = inputs.shape
    scaled = scaler.transform(inputs.reshape(-1, shape[-1])).reshape(shape)
    return np.clip(scaled, -clip_u, clip_u)


def _validate_d1_pair(train: D1Data, test: D1Data) -> None:
    if abs(train.dt - test.dt) > 1e-9:
        raise ValueError(f"Train/test dt mismatch: {train.dt} vs {test.dt}")
    if train.states.shape[1:] != test.states.shape[1:]:
        raise ValueError(f"Train/test state dimensions differ: {train.states.shape} vs {test.states.shape}")
    if train.inputs.shape[1:] != test.inputs.shape[1:]:
        raise ValueError(f"Train/test input dimensions differ: {train.inputs.shape} vs {test.inputs.shape}")


def _config_metadata(config: ExperimentConfig) -> dict:
    return {
        "seed": config.seed,
        "beta": config.beta,
        "lambda_p": config.lam_p,
        "lambda_r": config.lam_r,
        "res_scale": config.res_scale,
        "clip_x": config.clip_x,
        "clip_u": config.clip_u,
        "clip_r": config.clip_r,
        "feature": config.feature,
    }

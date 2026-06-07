# Anchored Rollouts Code Release

This folder contains the cleaned public code and data for the RSS 2026 paper
**Stabilizing 3D Continuum-Arm Rollouts via Equilibrium Anchoring and Feature-Lifted Residual Learning**.

The release focuses on the continuum-arm experiments present in this repository:

- equilibrium-prior prediction from the static dataset `D2`;
- residual-only dynamics from the dynamic dataset `D1`;
- the proposed equilibrium-anchored residual hybrid model;
- seed sweeps and `D2` fraction sweeps used for the paper ablations;
- compact result summaries for classical baselines that were already present in the raw artifact.

The soft-tail hardware and low-dimensional benchmark scripts/data described in the paper were not present in the raw local artifact, so they are not fabricated here.

## Layout

```text
code-release-v1/
  data/
    D1_train.h5
    D1_test.h5
    D2_dataset.h5
  results/
    compact paper result summaries
  scripts/
    run_continuum_models.py
    run_d2_fraction_sweep.py
    run_seed_sweep.py
    train_neural_prior.py
  src/anchored_rollouts/
    reusable loaders, features, metrics, and models
```

## Install

Use Python 3.10 or newer.

```bash
python -m venv .venv
.venv\Scripts\activate
pip install -r requirements.txt
```

On Linux or macOS, activate the environment with `source .venv/bin/activate`.
Install `requirements-neural.txt` only if you want to retrain the optional
neural equilibrium prior.

## Reproduce the Main Continuum-Arm Comparison

Run from this folder:

```bash
python scripts/run_continuum_models.py --model all
```

The command trains/evaluates:

- `equilibrium`: the anchor-only prior,
- `residual`: residual-only direct dynamics,
- `hybrid`: the proposed equilibrium-anchored residual predictor.

Outputs are written to `outputs/main_comparison/`.

Useful options:

```bash
python scripts/run_continuum_models.py --model hybrid --save-rollouts
python scripts/run_continuum_models.py --model all --seed 7 --lam-r 1.0 --clip-r 2.0
```

## Reproduce Seed and D2 Sweeps

```bash
python scripts/run_seed_sweep.py --seeds 30
python scripts/run_d2_fraction_sweep.py --seeds 5
```

The sweep defaults match the paper settings used in the raw artifact:
`beta=0.5`, linear features, `lambda_P=1e-3`, `lambda_R=1.0`,
`res_scale=1.0`, and residual clipping at `2.0`.

## Neural Equilibrium Prior

The paper also reports a neural-anchor variant. Train the neural equilibrium
map from `D2` with:

```bash
pip install -r requirements-neural.txt
python scripts/train_neural_prior.py
```

This produces `outputs/neural_prior/nn_prior.pt`. The main paper result is the
linear equilibrium-anchor hybrid; the neural prior script is included to make
the anchor-parameterization comparison reproducible.

## Data Notes

- `D1_train.h5`: dynamic training rollouts.
- `D1_test.h5`: held-out out-of-distribution dynamic rollouts.
- `D2_dataset.h5`: static input-equilibrium samples.

The loaders accept both MATLAB-style arrays `(state_dim, T, ntraj)` and
canonical Python arrays `(ntraj, T, state_dim)`.

## Result Summaries

The `results/` folder keeps small CSV/JSON summaries from the raw artifact.
Large generated rollouts, per-seed caches, plots, duplicate datasets, and
intermediate scratch files were removed because they can be regenerated from
the retained scripts and data.

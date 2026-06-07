# RSS 2026 Research Artifact V1

This artifact contains the code, datasets, and compact result summaries for the continuum-arm experiments in:

**Stabilizing 3D Continuum-Arm Rollouts via Equilibrium Anchoring and Feature-Lifted Residual Learning**

## Layout

```text
rss2026-artifact-v1/
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
    data loading, features, models, metrics, and experiment code
```

## Install

Use Python 3.10 or newer.

```bash
python -m venv .venv
.venv\Scripts\activate
pip install -r requirements.txt
```

On Linux or macOS:

```bash
source .venv/bin/activate
pip install -r requirements.txt
```

The optional neural-prior trainer requires PyTorch:

```bash
pip install -r requirements-neural.txt
```

## Main Continuum-Arm Comparison

Run from this folder:

```bash
python scripts/run_continuum_models.py --model all
```

This trains and evaluates:

- equilibrium prior;
- residual-only direct dynamics;
- equilibrium-anchored residual hybrid.

Outputs are written to `outputs/main_comparison/`.

Useful variants:

```bash
python scripts/run_continuum_models.py --model hybrid --save-rollouts
python scripts/run_continuum_models.py --model all --seed 7 --lam-r 1.0 --clip-r 2.0
```

## Seed And Equilibrium-Data Sweeps

```bash
python scripts/run_seed_sweep.py --seeds 30
python scripts/run_d2_fraction_sweep.py --seeds 5
```

Default settings match the continuum-arm artifact configuration:

- anchoring strength: 0.5;
- linear feature maps;
- equilibrium ridge penalty: 1e-3;
- residual ridge penalty: 1.0;
- residual scale: 1.0;
- residual clipping: 2.0.

## Neural Equilibrium Prior

The neural-anchor comparison can be retrained with:

```bash
python scripts/train_neural_prior.py
```

This writes `outputs/neural_prior/nn_prior.pt`.

## Data Files

- `D1_train.h5`: dynamic training rollouts.
- `D1_test.h5`: held-out out-of-distribution dynamic rollouts.
- `D2_dataset.h5`: static input-equilibrium samples.

The loaders accept MATLAB-style trajectory arrays and canonical Python trajectory arrays.

## Result Summaries

The `results/` folder contains small CSV and JSON summaries for:

- main continuum-arm model metrics;
- multi-seed summaries;
- equilibrium-data fraction sweep;
- baseline comparisons.

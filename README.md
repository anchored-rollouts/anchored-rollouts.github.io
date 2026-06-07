# Stabilizing 3D Continuum-Arm Rollouts

Project page and code release for the RSS 2026 paper:

**Stabilizing 3D Continuum-Arm Rollouts via Equilibrium Anchoring and Feature-Lifted Residual Learning**

Authors: Ahsan Tanveer, Rahdar Hussain Afridi, Waqar Hussain Afridi, Feitian Zhang, Guangming Xie

Website: <https://anchored-rollouts.github.io/>

## Overview

This work studies multi-step open-loop prediction for continuum and soft robots under actuation distribution shift. Recursive rollouts can accumulate small one-step errors, bias the predicted steady response, and eventually drift into implausible configurations.

The paper proposes an **equilibrium-anchored residual-learning** framework:

1. Learn an actuation-conditioned equilibrium map `P(u)` from static data `D2`.
2. Pull each recursive prediction toward this steady-state reference with a contractive anchor update.
3. Learn a feature-lifted residual correction from dynamic rollouts `D1`.
4. Evaluate all models under the same 200-step out-of-distribution rollout protocol.

## What Is In This Repository

- `index.html`: public project page.
- `assets/`: selected paper figures used by the website.
- `graphicalabstract.png`: hero/overview figure.
- `code-release-v1/`: cleaned public code and data release.

## Code Release

The cleaned release keeps the materials needed to reproduce the continuum-arm experiments available in the raw artifact:

- one canonical copy of `D1_train.h5`, `D1_test.h5`, and `D2_dataset.h5`;
- reusable Python modules for loading data, fitting models, and evaluating rollouts;
- scripts for the main continuum-arm comparison, multi-seed sweep, and `D2` fraction sweep;
- compact CSV/JSON summaries for paper results and baselines.

Large generated rollout caches, duplicate datasets, scratch folders, and intermediate plotting outputs were intentionally removed.

Main entry point:

```bash
cd code-release-v1
pip install -r requirements.txt
python scripts/run_continuum_models.py --model all
```

See [`code-release-v1/README.md`](code-release-v1/README.md) for full reproduction instructions.

## Paper Results Covered By The Release

The code release focuses on the continuum-arm experiments present in this repository:

- equilibrium-prior baseline,
- residual-only baseline,
- proposed equilibrium-anchored residual hybrid,
- neural-anchor variant summaries,
- `D2` fraction sweep,
- compact summaries for EDMDc, N4SID, SINDYc, MLP, and fish baseline comparisons.

The soft-tail hardware and low-dimensional benchmark code/data described in the paper were not present in the raw local artifact, so they are discussed on the project page but not fabricated in the release.

## Citation

```bibtex
@inproceedings{tanveer2026stabilizing,
  title={Stabilizing 3D Continuum-Arm Rollouts via Equilibrium Anchoring and Feature-Lifted Residual Learning},
  author={Tanveer, Ahsan and Afridi, Rahdar Hussain and Afridi, Waqar Hussain and Zhang, Feitian and Xie, Guangming},
  booktitle={Robotics: Science and Systems},
  year={2026}
}
```

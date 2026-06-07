# Stabilizing 3D Continuum-Arm Rollouts

Project page and RSS 2026 research artifact for:

**Stabilizing 3D Continuum-Arm Rollouts via Equilibrium Anchoring and Feature-Lifted Residual Learning**

Authors: Ahsan Tanveer, Rahdar Hussain Afridi, Waqar Hussain Afridi, Feitian Zhang, Guangming Xie

Website: <https://anchored-rollouts.github.io/>

## Repository Contents

- `index.html`: project page.
- `assets/`: figures used on the project page.
- `graphicalabstract.png`: overview image used in the page header.
- `rss2026-artifact-v1/`: paper artifact with code, datasets, and result summaries.

## Paper Artifact

The artifact contains the continuum-arm reproduction package:

- dynamic rollout datasets `D1_train.h5` and `D1_test.h5`;
- static equilibrium dataset `D2_dataset.h5`;
- reusable Python source for data loading, feature construction, model fitting, rollouts, and metrics;
- scripts for the main comparison, multi-seed sweep, and equilibrium-data fraction sweep;
- compact result summaries for the continuum-arm and baseline comparisons.

Main command:

```bash
cd rss2026-artifact-v1
pip install -r requirements.txt
python scripts/run_continuum_models.py --model all
```

See [`rss2026-artifact-v1/README.md`](rss2026-artifact-v1/README.md) for detailed artifact instructions.

## Citation

```bibtex
@inproceedings{tanveer2026stabilizing,
  title={Stabilizing 3D Continuum-Arm Rollouts via Equilibrium Anchoring and Feature-Lifted Residual Learning},
  author={Tanveer, Ahsan and Afridi, Rahdar Hussain and Afridi, Waqar Hussain and Zhang, Feitian and Xie, Guangming},
  booktitle={Robotics: Science and Systems},
  year={2026}
}
```

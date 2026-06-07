from __future__ import annotations

import argparse
import json
import sys
from pathlib import Path

import numpy as np

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT / "src"))

from anchored_rollouts import load_d1_h5, load_d2_h5
from anchored_rollouts.experiments import ExperimentConfig, run_equilibrium_and_hybrid, run_residual_only


def main() -> None:
    parser = argparse.ArgumentParser(description="Run continuum-arm predictors from the RSS paper.")
    parser.add_argument("--d1-train", default=str(ROOT / "data" / "D1_train.h5"))
    parser.add_argument("--d1-test", default=str(ROOT / "data" / "D1_test.h5"))
    parser.add_argument("--d2", default=str(ROOT / "data" / "D2_dataset.h5"))
    parser.add_argument("--model", choices=["equilibrium", "residual", "hybrid", "all"], default="all")
    parser.add_argument("--outdir", default=str(ROOT / "outputs" / "main_comparison"))
    parser.add_argument("--seed", type=int, default=0)
    parser.add_argument("--beta", type=float, default=0.5)
    parser.add_argument("--lam-p", type=float, default=1e-3)
    parser.add_argument("--lam-r", type=float, default=1.0)
    parser.add_argument("--res-scale", type=float, default=1.0)
    parser.add_argument("--clip-x", type=float, default=8.0)
    parser.add_argument("--clip-u", type=float, default=8.0)
    parser.add_argument("--clip-r", type=float, default=2.0)
    parser.add_argument("--feature", choices=["linear", "poly2"], default="linear")
    parser.add_argument("--save-rollouts", action="store_true", help="Save compressed rollout arrays in addition to metrics.")
    args = parser.parse_args()

    config = ExperimentConfig(
        seed=args.seed,
        beta=args.beta,
        lam_p=args.lam_p,
        lam_r=args.lam_r,
        res_scale=args.res_scale,
        clip_x=args.clip_x,
        clip_u=args.clip_u,
        clip_r=args.clip_r,
        feature=args.feature,
    )

    d1_train = load_d1_h5(args.d1_train)
    d1_test = load_d1_h5(args.d1_test)
    d2 = load_d2_h5(args.d2)
    outdir = Path(args.outdir)
    outdir.mkdir(parents=True, exist_ok=True)

    combined = {}
    if args.model in {"equilibrium", "hybrid", "all"}:
        result = run_equilibrium_and_hybrid(d1_train, d1_test, d2, config)
        metrics = result["metrics"]
        if args.model == "equilibrium":
            metrics = {
                key: value
                for key, value in metrics.items()
                if "hybrid" not in key and not key.startswith("trajectory_rmse_hybrid")
            }
        _write_result(outdir, "equilibrium_hybrid", metrics, result["rollouts"], args.save_rollouts)
        combined["equilibrium_hybrid"] = metrics

    if args.model in {"residual", "all"}:
        result = run_residual_only(d1_train, d1_test, config)
        _write_result(outdir, "residual_only", result["metrics"], result["rollouts"], args.save_rollouts)
        combined["residual_only"] = result["metrics"]

    (outdir / "summary.json").write_text(json.dumps(combined, indent=2), encoding="utf-8")
    print(json.dumps(combined, indent=2))
    print(f"Wrote metrics to {outdir}")


def _write_result(outdir: Path, name: str, metrics: dict, rollouts: dict, save_rollouts: bool) -> None:
    (outdir / f"{name}_metrics.json").write_text(json.dumps(metrics, indent=2), encoding="utf-8")
    if save_rollouts:
        np.savez_compressed(outdir / f"{name}_rollouts.npz", **rollouts)


if __name__ == "__main__":
    main()

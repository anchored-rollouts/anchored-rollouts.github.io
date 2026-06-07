from __future__ import annotations

import argparse
import json
import sys
from pathlib import Path

import pandas as pd

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT / "src"))

from anchored_rollouts import load_d1_h5, load_d2_h5
from anchored_rollouts.experiments import ExperimentConfig, run_equilibrium_and_hybrid


def main() -> None:
    parser = argparse.ArgumentParser(description="Run the D2 fraction ablation for the hybrid predictor.")
    parser.add_argument("--d1-train", default=str(ROOT / "data" / "D1_train.h5"))
    parser.add_argument("--d1-test", default=str(ROOT / "data" / "D1_test.h5"))
    parser.add_argument("--d2", default=str(ROOT / "data" / "D2_dataset.h5"))
    parser.add_argument("--outdir", default=str(ROOT / "outputs" / "d2_fraction_sweep"))
    parser.add_argument("--fractions", type=float, nargs="+", default=[0.10, 0.25, 0.50, 0.75, 1.00])
    parser.add_argument("--seeds", type=int, default=5)
    parser.add_argument("--beta", type=float, default=0.5)
    parser.add_argument("--lam-p", type=float, default=1e-3)
    parser.add_argument("--lam-r", type=float, default=1.0)
    parser.add_argument("--res-scale", type=float, default=1.0)
    parser.add_argument("--clip-r", type=float, default=2.0)
    args = parser.parse_args()

    d1_train = load_d1_h5(args.d1_train)
    d1_test = load_d1_h5(args.d1_test)
    d2 = load_d2_h5(args.d2)
    outdir = Path(args.outdir)
    outdir.mkdir(parents=True, exist_ok=True)

    rows = []
    for fraction in args.fractions:
        for seed in range(args.seeds):
            config = ExperimentConfig(
                seed=seed,
                beta=args.beta,
                lam_p=args.lam_p,
                lam_r=args.lam_r,
                res_scale=args.res_scale,
                clip_r=args.clip_r,
            )
            metrics = run_equilibrium_and_hybrid(d1_train, d1_test, d2, config, d2_fraction=fraction)["metrics"]
            rows.append(
                {
                    "d2_fraction": fraction,
                    "seed": seed,
                    "n_d2_used": metrics["n_d2_used"],
                    "rmse_equilibrium_raw": metrics["rmse_equilibrium_raw"],
                    "rmse_hybrid_raw": metrics["rmse_hybrid_raw"],
                    "hybrid_improvement_percent": metrics["hybrid_improvement_percent"],
                    "n_test_traj_used": metrics["n_test_traj_used"],
                }
            )
            pd.DataFrame(rows).to_csv(outdir / "d2_fraction_sweep.csv", index=False)
            print(f"fraction={fraction:.2f} seed={seed:02d} rmse={metrics['rmse_hybrid_raw']:.6f}")

    frame = pd.DataFrame(rows)
    summary = frame.groupby("d2_fraction")["rmse_hybrid_raw"].agg(["mean", "std", "count"]).reset_index()
    frame.to_csv(outdir / "d2_fraction_sweep.csv", index=False)
    summary.to_csv(outdir / "d2_fraction_summary.csv", index=False)
    (outdir / "meta.json").write_text(json.dumps(vars(args), indent=2), encoding="utf-8")
    print(summary.to_string(index=False))


if __name__ == "__main__":
    main()

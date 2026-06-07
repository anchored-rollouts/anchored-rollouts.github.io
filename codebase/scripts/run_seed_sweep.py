from __future__ import annotations

import argparse
import json
import sys
from pathlib import Path

import pandas as pd

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT / "src"))

from anchored_rollouts import load_d1_h5, load_d2_h5
from anchored_rollouts.experiments import ExperimentConfig, run_equilibrium_and_hybrid, run_residual_only


def main() -> None:
    parser = argparse.ArgumentParser(description="Run a multi-seed comparison of continuum-arm predictors.")
    parser.add_argument("--d1-train", default=str(ROOT / "data" / "D1_train.h5"))
    parser.add_argument("--d1-test", default=str(ROOT / "data" / "D1_test.h5"))
    parser.add_argument("--d2", default=str(ROOT / "data" / "D2_dataset.h5"))
    parser.add_argument("--outdir", default=str(ROOT / "outputs" / "seed_sweep"))
    parser.add_argument("--seeds", type=int, default=30)
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
    for seed in range(args.seeds):
        config = ExperimentConfig(
            seed=seed,
            beta=args.beta,
            lam_p=args.lam_p,
            lam_r=args.lam_r,
            res_scale=args.res_scale,
            clip_r=args.clip_r,
        )
        hybrid_metrics = run_equilibrium_and_hybrid(d1_train, d1_test, d2, config)["metrics"]
        residual_metrics = run_residual_only(d1_train, d1_test, config)["metrics"]
        rows.append(
            {
                "seed": seed,
                "rmse_equilibrium_raw": hybrid_metrics["rmse_equilibrium_raw"],
                "rmse_hybrid_raw": hybrid_metrics["rmse_hybrid_raw"],
                "rmse_residual_raw": residual_metrics["rmse_residual_raw"],
                "hybrid_improvement_percent": hybrid_metrics["hybrid_improvement_percent"],
                "n_test_traj_used": hybrid_metrics["n_test_traj_used"],
            }
        )
        pd.DataFrame(rows).to_csv(outdir / "per_seed.csv", index=False)
        print(f"seed={seed:02d} hybrid={hybrid_metrics['rmse_hybrid_raw']:.6f} residual={residual_metrics['rmse_residual_raw']:.6f}")

    frame = pd.DataFrame(rows)
    summary = {
        "seeds": args.seeds,
        "rmse_equilibrium_mean": float(frame["rmse_equilibrium_raw"].mean()),
        "rmse_residual_mean": float(frame["rmse_residual_raw"].mean()),
        "rmse_hybrid_mean": float(frame["rmse_hybrid_raw"].mean()),
        "hybrid_win_rate_vs_equilibrium": float((frame["rmse_hybrid_raw"] < frame["rmse_equilibrium_raw"]).mean()),
        "hybrid_win_rate_vs_residual": float((frame["rmse_hybrid_raw"] < frame["rmse_residual_raw"]).mean()),
    }
    (outdir / "summary.json").write_text(json.dumps(summary, indent=2), encoding="utf-8")
    print(json.dumps(summary, indent=2))


if __name__ == "__main__":
    main()

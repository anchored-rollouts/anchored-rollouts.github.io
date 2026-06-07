from __future__ import annotations

import argparse
import json
import sys
from pathlib import Path

import numpy as np
import torch
import torch.nn as nn
from torch.utils.data import DataLoader, TensorDataset

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT / "src"))

from anchored_rollouts import load_d1_h5, load_d2_h5
from anchored_rollouts.data import split_train_ids
from anchored_rollouts.experiments import fit_scalers


class MLP(nn.Module):
    def __init__(self, hidden: tuple[int, ...]):
        super().__init__()
        layers: list[nn.Module] = []
        in_dim = 3
        for width in hidden:
            layers.extend([nn.Linear(in_dim, width), nn.ReLU()])
            in_dim = width
        layers.append(nn.Linear(in_dim, 144))
        self.net = nn.Sequential(*layers)

    def forward(self, inputs: torch.Tensor) -> torch.Tensor:
        return self.net(inputs)


def main() -> None:
    parser = argparse.ArgumentParser(description="Train the neural equilibrium prior P_theta(u).")
    parser.add_argument("--d1-train", default=str(ROOT / "data" / "D1_train.h5"))
    parser.add_argument("--d2", default=str(ROOT / "data" / "D2_dataset.h5"))
    parser.add_argument("--outdir", default=str(ROOT / "outputs" / "neural_prior"))
    parser.add_argument("--seed", type=int, default=0)
    parser.add_argument("--epochs", type=int, default=80)
    parser.add_argument("--batch-size", type=int, default=1024)
    parser.add_argument("--lr", type=float, default=1e-3)
    parser.add_argument("--hidden", default="128,256,256")
    args = parser.parse_args()

    torch.manual_seed(args.seed)
    np.random.seed(args.seed)

    d1 = load_d1_h5(args.d1_train)
    d2 = load_d2_h5(args.d2)
    train_ids = split_train_ids(d1.states.shape[0], args.seed)
    state_scaler, input_scaler = fit_scalers(d1, train_ids, std_floor=1e-8)

    inputs = input_scaler.transform(d2.inputs).astype(np.float32)
    states = state_scaler.transform(d2.states).astype(np.float32)
    dataset = TensorDataset(torch.from_numpy(inputs), torch.from_numpy(states))
    loader = DataLoader(dataset, batch_size=args.batch_size, shuffle=True)

    hidden = tuple(int(part) for part in args.hidden.split(",") if part.strip())
    model = MLP(hidden)
    optimizer = torch.optim.Adam(model.parameters(), lr=args.lr)
    loss_fn = nn.MSELoss()

    for epoch in range(1, args.epochs + 1):
        losses = []
        model.train()
        for batch_inputs, batch_states in loader:
            loss = loss_fn(model(batch_inputs), batch_states)
            optimizer.zero_grad()
            loss.backward()
            optimizer.step()
            losses.append(float(loss.item()))
        if epoch == 1 or epoch % 10 == 0:
            print(f"epoch={epoch:03d} mse={np.mean(losses):.6e}")

    outdir = Path(args.outdir)
    outdir.mkdir(parents=True, exist_ok=True)
    torch.save(
        {
            "state_dict": model.state_dict(),
            "hidden": hidden,
            "state_mean": state_scaler.mean,
            "state_std": state_scaler.std,
            "input_mean": input_scaler.mean,
            "input_std": input_scaler.std,
        },
        outdir / "nn_prior.pt",
    )
    (outdir / "meta.json").write_text(json.dumps({"hidden": hidden, "epochs": args.epochs, "seed": args.seed}, indent=2), encoding="utf-8")
    print(f"Wrote {outdir / 'nn_prior.pt'}")


if __name__ == "__main__":
    main()

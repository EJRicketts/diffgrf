"""Inverse problem: recover the correlation length from a target field."""

import torch

from diffgrf import DifferentiableGRF


def main():
    grid = [128, 128]
    seed = 0

    target = (
        DifferentiableGRF(
            "gau",
            2,
            1.0,
            12.0,
            seed=seed,
        )
        .structured(grid)
        .detach()
    )

    ell = torch.tensor(5.0, requires_grad=True)
    opt = torch.optim.Adam([ell], lr=0.2)

    for step in range(200):
        opt.zero_grad()
        pred = DifferentiableGRF(
            "gau",
            2,
            1.0,
            ell,
            seed=seed,
        ).structured(grid)
        loss = (pred - target).pow(2).mean()
        loss.backward()
        opt.step()
        if step % 20 == 0:
            print(f"step {step:3d}  loss={loss.item():.5f}  ell={ell.item():.3f}")

    print(f"\nrecovered ell = {ell.item():.3f}  (truth = 12.000)")


if __name__ == "__main__":
    main()

"""Anisotropic 2D field: per-axis correlation lengths with rotation."""

import math

import matplotlib.pyplot as plt

from diffgrf import DifferentiableGRF


def main():
    len_scale = [20.0, 5.0]  # stretched in x, thin in y before rotation
    angle = math.pi / 4  # rotate 45 degrees

    grf = DifferentiableGRF(
        kernel="gau",
        dim=2,
        variance=1.0,
        len_scale=len_scale,
        angles=angle,
        seed=0,
        mode_no=2000,
    )
    field = grf.structured([200, 200])

    plt.figure(figsize=(4, 4))
    plt.imshow(field.detach().numpy(), cmap="viridis", origin="lower")
    plt.colorbar(label="field value")
    plt.title(f"Anisotropic: len_scale={len_scale}, angle=45 deg")
    plt.tight_layout()
    plt.savefig("anisotropy.png", dpi=150)
    print("wrote anisotropy.png")


if __name__ == "__main__":
    main()

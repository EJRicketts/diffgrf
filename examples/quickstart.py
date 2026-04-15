"""Quickstart: generate a 2D Gaussian random field and plot it."""

import matplotlib.pyplot as plt

from diffgrf import DifferentiableGRF


def main():
    grf = DifferentiableGRF(
        kernel="gau",
        dim=2,
        variance=1.0,
        len_scale=10.0,
        seed=0,
    )
    field = grf.structured([128, 128])

    plt.figure(figsize=(4, 4))
    plt.imshow(field.detach().numpy(), cmap="viridis", origin="lower")
    plt.colorbar(label="field value")
    plt.title("DiffGRF: 2D Gaussian kernel, len_scale=10")
    plt.tight_layout()
    plt.savefig("quickstart.png", dpi=150)
    print("wrote quickstart.png")


if __name__ == "__main__":
    main()

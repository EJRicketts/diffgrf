"""Gradient tests: autograd gradcheck + explicit backward smoke tests."""

import torch

from diffgrf import DifferentiableGRF


def test_gradcheck_len_scale_gaussian_1d():
    def f(ell):
        return DifferentiableGRF(
            "gau",
            1,
            1.0,
            ell,
            seed=0,
            mode_no=200,
        ).structured([16])

    ell = torch.tensor(3.0, dtype=torch.float64, requires_grad=True)
    assert torch.autograd.gradcheck(f, (ell,), eps=1e-5, atol=1e-4, rtol=1e-3)


def test_gradcheck_len_scale_gaussian_2d():
    def f(ell):
        return DifferentiableGRF(
            "gau",
            2,
            1.0,
            ell,
            seed=0,
            mode_no=200,
        ).structured([8, 8])

    ell = torch.tensor(5.0, dtype=torch.float64, requires_grad=True)
    assert torch.autograd.gradcheck(f, (ell,), eps=1e-5, atol=1e-4, rtol=1e-3)


def test_gradcheck_variance_gaussian_2d():
    def f(var):
        return DifferentiableGRF(
            "gau",
            2,
            var,
            5.0,
            seed=0,
            mode_no=200,
        ).structured([8, 8])

    var = torch.tensor(1.5, dtype=torch.float64, requires_grad=True)
    assert torch.autograd.gradcheck(f, (var,), eps=1e-5, atol=1e-4, rtol=1e-3)


def test_gradcheck_angle_gaussian_2d():
    def f(ang):
        return DifferentiableGRF(
            "gau",
            2,
            1.0,
            [10.0, 3.0],
            seed=0,
            mode_no=200,
            angles=ang,
        ).structured([8, 8])

    ang = torch.tensor(0.4, dtype=torch.float64, requires_grad=True)
    assert torch.autograd.gradcheck(f, (ang,), eps=1e-5, atol=1e-4, rtol=1e-3)


def test_gradcheck_len_scale_matern_2d():
    def f(ell):
        return DifferentiableGRF(
            "mat",
            2,
            1.0,
            ell,
            seed=0,
            mode_no=200,
            nu=1.5,
        ).structured([8, 8])

    ell = torch.tensor(5.0, dtype=torch.float64, requires_grad=True)
    assert torch.autograd.gradcheck(f, (ell,), eps=1e-5, atol=1e-4, rtol=1e-3)


def test_gradcheck_anisotropy_tensor_gaussian_2d():
    """Per-axis len_scale as a tensor leaf: gradients should flow to both axes."""

    def f(lx, ly):
        return DifferentiableGRF(
            "gau",
            2,
            1.0,
            [lx, ly],
            seed=0,
            mode_no=200,
        ).structured([8, 8])

    lx = torch.tensor(6.0, dtype=torch.float64, requires_grad=True)
    ly = torch.tensor(3.0, dtype=torch.float64, requires_grad=True)
    assert torch.autograd.gradcheck(f, (lx, ly), eps=1e-5, atol=1e-4, rtol=1e-3)


def test_gradcheck_rotation_3d_gaussian():
    """3D Tait--Bryan angles as tensor leaves."""

    def f(ang):
        return DifferentiableGRF(
            "gau",
            3,
            1.0,
            [6.0, 3.0, 2.0],
            seed=0,
            mode_no=200,
            angles=ang,
        ).structured([8, 8, 8])

    ang = torch.tensor([0.3, 0.2, 0.1], dtype=torch.float64, requires_grad=True)
    assert torch.autograd.gradcheck(f, (ang,), eps=1e-5, atol=1e-4, rtol=1e-3)


def test_backward_smoke():
    """Full-size backward pass produces finite gradient."""
    ell = torch.tensor(8.0, requires_grad=True)
    f = DifferentiableGRF(
        "gau",
        2,
        1.0,
        ell,
        seed=0,
        mode_no=1000,
    ).structured([64, 64])
    loss = f.pow(2).mean()
    loss.backward()
    assert ell.grad is not None and torch.isfinite(ell.grad).all()


def test_recover_len_scale_via_adam():
    """Short optimisation should reduce the loss monotonically."""
    grid = [64, 64]
    target = (
        DifferentiableGRF("gau", 2, 1.0, 10.0, seed=1, mode_no=1000)
        .structured(grid)
        .detach()
    )

    def loss_fn(ell):
        pred = DifferentiableGRF("gau", 2, 1.0, ell, seed=1, mode_no=1000).structured(
            grid
        )
        return (pred - target).pow(2).mean()

    ell = torch.tensor(8.0, requires_grad=True)
    loss0 = float(loss_fn(ell).detach())
    opt = torch.optim.Adam([ell], lr=0.2)
    for _ in range(60):
        opt.zero_grad()
        loss_fn(ell).backward()
        opt.step()
    loss_final = float(loss_fn(ell).detach())
    assert (
        loss_final < 0.5 * loss0
    ), f"loss did not halve: initial={loss0:.4f}, final={loss_final:.4f}"
    assert abs(ell.detach().item() - 10.0) < abs(
        8.0 - 10.0
    ), f"final ell = {ell.detach().item():.3f} did not move toward truth"

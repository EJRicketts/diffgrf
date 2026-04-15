"""Tests for kernel support (Gaussian, Matern) and reproducibility."""

import torch

from diffgrf import DifferentiableGRF


def _ensemble_var(kernel, nu=1.0, n_real=60):
    vs = []
    for seed in range(n_real):
        f = DifferentiableGRF(
            kernel,
            2,
            1.0,
            10.0,
            seed=seed,
            mode_no=2000,
            nu=nu,
        ).structured([64, 64])
        vs.append(float(f.var()))
    return torch.tensor(vs).mean().item()


def test_gaussian_runs():
    f = DifferentiableGRF("gau", 2, 1.0, 10.0, seed=0).structured([32, 32])
    assert torch.isfinite(f).all()


def test_matern_runs_at_nu_0p5():
    f = DifferentiableGRF("mat", 2, 1.0, 10.0, seed=0, nu=0.5).structured([32, 32])
    assert torch.isfinite(f).all()


def test_matern_runs_at_nu_1p5():
    f = DifferentiableGRF("mat", 2, 1.0, 10.0, seed=0, nu=1.5).structured([32, 32])
    assert torch.isfinite(f).all()


def test_matern_runs_at_nu_2p5():
    f = DifferentiableGRF("mat", 2, 1.0, 10.0, seed=0, nu=2.5).structured([32, 32])
    assert torch.isfinite(f).all()


def test_gaussian_ensemble_variance_approaches_variance_param():
    mean_var = _ensemble_var("gau")
    assert 0.75 < mean_var < 1.25, f"ensemble var = {mean_var}"


def test_matern_ensemble_variance_approaches_variance_param():
    mean_var = _ensemble_var("mat", nu=1.5)
    assert 0.75 < mean_var < 1.25, f"ensemble var = {mean_var}"


def test_unknown_kernel_raises():
    try:
        DifferentiableGRF("bad", 2, 1.0, 10.0, seed=0).structured([8, 8])
    except ValueError:
        return
    raise AssertionError("expected ValueError for unknown kernel")


def test_same_seed_deterministic():
    f1 = DifferentiableGRF("gau", 2, 1.0, 10.0, seed=42).structured([32, 32])
    f2 = DifferentiableGRF("gau", 2, 1.0, 10.0, seed=42).structured([32, 32])
    assert torch.allclose(f1, f2)


def test_different_seeds_differ():
    f1 = DifferentiableGRF("gau", 2, 1.0, 10.0, seed=1).structured([32, 32])
    f2 = DifferentiableGRF("gau", 2, 1.0, 10.0, seed=2).structured([32, 32])
    assert not torch.allclose(f1, f2)

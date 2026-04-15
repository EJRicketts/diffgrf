"""Tests for differentiable rotation."""

import math

import torch

from diffgrf import DifferentiableGRF


def test_zero_angle_equals_no_angle_2d():
    f0 = DifferentiableGRF("gau", 2, 1.0, 10.0, seed=0).structured([32, 32])
    fz = DifferentiableGRF("gau", 2, 1.0, 10.0, seed=0, angles=0.0).structured([32, 32])
    assert torch.allclose(f0, fz)


def test_rotation_2pi_equals_no_rotation_2d():
    f0 = DifferentiableGRF("gau", 2, 1.0, 10.0, seed=0).structured([32, 32])
    f_full = DifferentiableGRF(
        "gau", 2, 1.0, 10.0, seed=0, angles=2 * math.pi
    ).structured([32, 32])
    assert torch.allclose(f0, f_full, atol=1e-6)


def test_rotation_changes_field_when_anisotropic():
    base = DifferentiableGRF("gau", 2, 1.0, [20.0, 4.0], seed=0, angles=0.0).structured(
        [32, 32]
    )
    rot = DifferentiableGRF(
        "gau", 2, 1.0, [20.0, 4.0], seed=0, angles=math.pi / 2
    ).structured([32, 32])
    # rotated anisotropy should produce a different field
    assert not torch.allclose(base, rot)


def test_3d_rotation_runs():
    f = DifferentiableGRF(
        "gau", 3, 1.0, 5.0, seed=0, angles=[0.3, 0.5, 0.2]
    ).structured([16, 16, 16])
    assert f.shape == torch.Size([16, 16, 16])
    assert torch.isfinite(f).all()


def test_differentiable_rotation_2d():
    angle = torch.tensor(0.3, requires_grad=True, dtype=torch.float64)
    f = DifferentiableGRF("gau", 2, 1.0, [20.0, 4.0], seed=0, angles=angle).structured(
        [32, 32]
    )
    loss = f.pow(2).mean()
    loss.backward()
    assert angle.grad is not None
    assert torch.isfinite(angle.grad).all()

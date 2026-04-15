"""Tests for 1D / 2D / 3D structured and unstructured generation."""

import math

import torch

from diffgrf import DifferentiableGRF


def test_structured_1d_shape():
    f = DifferentiableGRF("gau", 1, 1.0, 5.0, seed=0).structured([64])
    assert f.shape == torch.Size([64])
    assert torch.isfinite(f).all()


def test_structured_2d_shape():
    f = DifferentiableGRF("gau", 2, 1.0, 10.0, seed=0).structured([32, 48])
    assert f.shape == torch.Size([32, 48])
    assert torch.isfinite(f).all()


def test_structured_3d_shape():
    f = DifferentiableGRF("gau", 3, 1.0, 4.0, seed=0).structured([16, 20, 24])
    assert f.shape == torch.Size([16, 20, 24])
    assert torch.isfinite(f).all()


def test_unstructured_1d():
    pts = torch.linspace(0, 10, 50).unsqueeze(1)
    f = DifferentiableGRF("gau", 1, 1.0, 3.0, seed=0).unstructured(pts)
    assert f.shape == torch.Size([50])
    assert torch.isfinite(f).all()


def test_unstructured_2d():
    pts = torch.rand(100, 2) * 20
    f = DifferentiableGRF("gau", 2, 1.0, 5.0, seed=0).unstructured(pts)
    assert f.shape == torch.Size([100])
    assert torch.isfinite(f).all()


def test_unstructured_3d():
    pts = torch.rand(50, 3) * 10
    f = DifferentiableGRF("gau", 3, 1.0, 2.0, seed=0).unstructured(pts)
    assert f.shape == torch.Size([50])
    assert torch.isfinite(f).all()


def test_variance_positive_finite():
    f = DifferentiableGRF("gau", 2, 1.0, 10.0, seed=0, mode_no=2000).structured(
        [128, 128]
    )
    v = float(f.var())
    assert v > 0 and math.isfinite(v)
    # single-realisation sample variance is noisy, but should be O(1)
    assert 0.5 < v < 1.5

"""Device-equivalence tests for CUDA and Apple MPS (skipped when unavailable)."""

import pytest
import torch

from diffgrf import DifferentiableGRF

cuda_only = pytest.mark.skipif(
    not torch.cuda.is_available(),
    reason="CUDA not available",
)

mps_only = pytest.mark.skipif(
    not (torch.backends.mps.is_available() and torch.backends.mps.is_built()),
    reason="Apple MPS not available",
)


@cuda_only
def test_cuda_matches_cpu_gaussian_2d():
    f_cpu = DifferentiableGRF(
        "gau",
        2,
        1.0,
        10.0,
        seed=0,
        mode_no=500,
        device="cpu",
    ).structured([32, 32])
    f_gpu = (
        DifferentiableGRF(
            "gau",
            2,
            1.0,
            10.0,
            seed=0,
            mode_no=500,
            device="cuda",
        )
        .structured([32, 32])
        .cpu()
    )
    assert torch.allclose(f_cpu, f_gpu, atol=1e-6)


@cuda_only
def test_cuda_matches_cpu_matern_3d():
    f_cpu = DifferentiableGRF(
        "mat",
        3,
        1.0,
        4.0,
        seed=0,
        mode_no=500,
        nu=1.5,
        device="cpu",
    ).structured([12, 12, 12])
    f_gpu = (
        DifferentiableGRF(
            "mat",
            3,
            1.0,
            4.0,
            seed=0,
            mode_no=500,
            nu=1.5,
            device="cuda",
        )
        .structured([12, 12, 12])
        .cpu()
    )
    assert torch.allclose(f_cpu, f_gpu, atol=1e-6)


@mps_only
def test_mps_matches_cpu_gaussian_2d():
    """MPS lacks float64, so run both paths at float32 and compare."""
    f_cpu = DifferentiableGRF(
        "gau",
        2,
        1.0,
        10.0,
        seed=0,
        mode_no=500,
        device="cpu",
        dtype=torch.float32,
    ).structured([32, 32])
    f_mps = (
        DifferentiableGRF(
            "gau",
            2,
            1.0,
            10.0,
            seed=0,
            mode_no=500,
            device="mps",
            dtype=torch.float32,
        )
        .structured([32, 32])
        .cpu()
    )
    assert torch.allclose(f_cpu, f_mps, atol=1e-4)


@mps_only
def test_mps_matches_cpu_matern_3d():
    f_cpu = DifferentiableGRF(
        "mat",
        3,
        1.0,
        4.0,
        seed=0,
        mode_no=500,
        nu=1.5,
        device="cpu",
        dtype=torch.float32,
    ).structured([12, 12, 12])
    f_mps = (
        DifferentiableGRF(
            "mat",
            3,
            1.0,
            4.0,
            seed=0,
            mode_no=500,
            nu=1.5,
            device="mps",
            dtype=torch.float32,
        )
        .structured([12, 12, 12])
        .cpu()
    )
    assert torch.allclose(f_cpu, f_mps, atol=1e-4)


def test_dtype_cast_cpu_float32():
    """Exercise the non-float64 dtype path on CPU. This covers the
    dtype-casting branch used by MPS/CUDA without requiring a GPU."""
    f = DifferentiableGRF(
        "gau",
        2,
        1.0,
        10.0,
        seed=0,
        mode_no=500,
        device="cpu",
        dtype=torch.float32,
    ).structured([32, 32])
    assert f.dtype == torch.float32
    assert torch.isfinite(f).all()


@mps_only
def test_mps_gradient_flows_gaussian_2d():
    ell = torch.tensor(10.0, requires_grad=True, dtype=torch.float32, device="mps")
    f = DifferentiableGRF(
        "gau",
        2,
        1.0,
        ell,
        seed=0,
        mode_no=500,
        device="mps",
        dtype=torch.float32,
    ).structured([32, 32])
    loss = (f**2).mean()
    loss.backward()
    assert ell.grad is not None and torch.isfinite(ell.grad).all()

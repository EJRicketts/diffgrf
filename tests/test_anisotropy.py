"""Tests that per-axis len_scale stretches the empirical correlation."""

import torch

from diffgrf import DifferentiableGRF


def _empirical_acf_axis(field, axis, max_lag):
    """Ensemble-normalised autocorrelation along a given axis."""
    f = field - field.mean()
    lags = list(range(max_lag + 1))
    acf = []
    for k in lags:
        shifted = torch.roll(f, shifts=k, dims=axis)
        c = (f * shifted).mean()
        acf.append(float(c))
    acf0 = acf[0]
    return [a / acf0 for a in acf]


def _mean_acf_over_seeds(axis, len_scale, n_real=40, max_lag=15):
    acfs = []
    for seed in range(n_real):
        f = DifferentiableGRF(
            "gau",
            2,
            1.0,
            len_scale,
            seed=seed,
            mode_no=2000,
        ).structured([128, 128])
        acfs.append(_empirical_acf_axis(f, axis, max_lag))
    return [sum(a[k] for a in acfs) / n_real for k in range(max_lag + 1)]


def test_anisotropic_long_axis_decays_slower():
    # Long along axis 0 (20), short along axis 1 (4)
    acf0 = _mean_acf_over_seeds(axis=0, len_scale=[20.0, 4.0])
    acf1 = _mean_acf_over_seeds(axis=1, len_scale=[20.0, 4.0])
    # axis 0 autocorrelation at lag 5 should be > axis 1 autocorrelation at lag 5
    assert (
        acf0[5] > acf1[5] + 0.1
    ), f"axis0 acf[5]={acf0[5]:.3f}, axis1 acf[5]={acf1[5]:.3f}"


def test_isotropic_vs_anisotropic_differ():
    f_iso = DifferentiableGRF("gau", 2, 1.0, 10.0, seed=0).structured([64, 64])
    f_ani = DifferentiableGRF("gau", 2, 1.0, [20.0, 4.0], seed=0).structured([64, 64])
    assert not torch.allclose(f_iso, f_ani)

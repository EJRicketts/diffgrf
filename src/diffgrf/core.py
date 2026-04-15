"""
DiffGRF: Differentiable Gaussian random field generation via the spectral method.

Uses the reparameterization trick in spectral space: frozen base samples
are transformed through the covariance-model-dependent spectral density,
making the field differentiable w.r.t. variance, correlation length(s),
anisotropy, and rotation angles via PyTorch autograd.

Mirrors the randomization method of Kraichnan (1970) and Hesse et al.
(2014) used by GSTools (Mueller et al. 2022), implemented in PyTorch for
analytical autograd gradients and GPU/MPS support.

Spectral derivations (target conventions match GSTools):

    Gaussian   C(r) = sigma^2 * exp(-pi/4 * (r/ell)^2)
               => k_i ~ N(0, pi/(2 ell^2)) i.i.d.
               => |k| = (sqrt(pi)/ell) * Chi(d)/sqrt(2)

    Matern     C(r) = sigma^2 * (2^(1-nu)/Gamma(nu))(alpha r)^nu K_nu(alpha r)
               with alpha = sqrt(nu)/ell.
               Spectral density  S(k) ~ (alpha^2 + |k|^2)^-(nu + d/2)
               (Whittle-Matern; see Stein 1999, "Interpolation of Spatial Data").
               Substituting v = |k|^2/alpha^2 gives v ~ BetaPrime(d/2, nu) =
               Chi^2(d)/Chi^2(2*nu), hence
               => |k| = alpha * Chi(d)/sqrt(Chi^2(2*nu))
                      = (sqrt(nu)/ell) * Chi(d)/sqrt(Chi^2(2*nu))

In both cases the frozen "base radii" carry the kernel-specific Chi-ratio
distribution, and the differentiable `*_spectral_transform` applies the
len_scale (and, for Matern, nu) scaling. This factorisation keeps every
parameter-dependent operation on the autograd graph.
"""

import math
import pathlib

import numpy as np
import torch

# ---------------------------------------------------------------------------
# Gaussian kernel: |k| = (sqrt(pi)/ell) * base, where base = Chi(d)/sqrt(2).
# ---------------------------------------------------------------------------


def _gaussian_base_radii_1d(mode_no, rng):
    """Chi(1)/sqrt(2) = |N(0,1)|/sqrt(2) sampled via inverse CDF.

    For u ~ U(0,1), erfinv(u) = |N(0,1)|/sqrt(2). Combined with the
    shared transform (multiplication by sqrt(pi)/ell), this reproduces
    k ~ N(0, pi/(2*ell^2)) as required by the 1D Gaussian spectrum.
    """
    u = torch.rand(mode_no, generator=rng, dtype=torch.float64)
    u = torch.clamp(u, min=1e-10, max=1.0 - 1e-10)
    return torch.erfinv(u)


def _gaussian_base_radii_2d(mode_no, rng):
    """Chi(2)/sqrt(2) (Rayleigh, sigma=1) via inverse CDF.

    For u ~ U(0,1), sqrt(-log(1-u)) is the inverse CDF of a Rayleigh
    variate with scale 1, which equals Chi(2)/sqrt(2).
    """
    u = torch.rand(mode_no, generator=rng, dtype=torch.float64)
    u = torch.clamp(u, min=1e-10, max=1.0 - 1e-10)
    return torch.sqrt(-torch.log(1.0 - u))


def _gaussian_base_radii_3d(mode_no, rng):
    """Chi(3)/sqrt(2): draw three independent N(0,1) and combine.

    There is no closed-form inverse CDF for Chi(3)/sqrt(2), so we use the
    direct sum-of-squares construction.
    """
    x = torch.randn(3, mode_no, generator=rng, dtype=torch.float64)
    return torch.sqrt((x**2).sum(dim=0) / 2.0)


def gaussian_spectral_transform(base_radii, len_scale):
    """|k| = (sqrt(pi)/len_scale) * base_radii.

    base_radii must already encode Chi(d)/sqrt(2) (see helpers above).
    The result samples the radial spectral density of the Gaussian
    kernel C(r) = exp(-pi/4 (r/ell)^2).
    """
    return (math.sqrt(math.pi) / len_scale) * base_radii


# ---------------------------------------------------------------------------
# Matern kernel: |k| = (sqrt(nu)/ell) * base, where base = Chi(d)/sqrt(Chi^2(2*nu)).
# ---------------------------------------------------------------------------


def _matern_base_radii(mode_no, nu, dim, rng):
    """Chi(d) / sqrt(Chi^2(2*nu)) -- the BetaPrime(d/2, nu) ratio.

    See Stein (1999, "Interpolation of Spatial Data", Sec. 2.7) for the
    Whittle-Matern spectral density. The Gamma draw for Chi^2(2*nu) uses
    a numpy Generator so that it can be deterministically seeded from the
    torch Generator (torch.distributions.Gamma.sample() does not accept a
    generator argument, which would otherwise break reproducibility and
    the reparameterisation trick).
    """
    x = torch.randn(dim, mode_no, generator=rng, dtype=torch.float64)
    chi_dim = torch.sqrt((x**2).sum(dim=0))

    np_seed = int(
        torch.randint(
            0,
            2**31 - 1,
            (1,),
            generator=rng,
            dtype=torch.int64,
        ).item()
    )
    nprng = np.random.default_rng(np_seed)
    chi2_2nu = torch.tensor(
        2.0 * nprng.gamma(nu, 1.0, size=mode_no),
        dtype=torch.float64,
    )
    return chi_dim / torch.sqrt(chi2_2nu)


def matern_spectral_transform(base_radii, len_scale, nu):
    """|k| = (sqrt(nu)/len_scale) * base_radii.

    base_radii must already encode Chi(d)/sqrt(Chi^2(2*nu)) (see helper
    above). The result samples the radial spectral density of the Matern
    kernel under the convention alpha = sqrt(nu)/ell, matching the
    GSTools Matern parameterisation.
    """
    return (math.sqrt(nu) / len_scale) * base_radii


# ---------------------------------------------------------------------------
# Sphere sampling
# ---------------------------------------------------------------------------


def _sample_sphere(dim, mode_no, rng):
    """Sample uniform directions on the unit sphere. Returns (dim, mode_no)."""
    if dim == 1:
        # Random +/-1
        signs = (
            2.0
            * torch.bernoulli(
                torch.full((mode_no,), 0.5, dtype=torch.float64), generator=rng
            )
            - 1.0
        )
        return signs.unsqueeze(0)
    if dim == 2:
        phi = 2.0 * math.pi * torch.rand(mode_no, generator=rng, dtype=torch.float64)
        return torch.stack([torch.cos(phi), torch.sin(phi)])
    if dim == 3:
        phi = 2.0 * math.pi * torch.rand(mode_no, generator=rng, dtype=torch.float64)
        cos_theta = 2.0 * torch.rand(mode_no, generator=rng, dtype=torch.float64) - 1.0
        sin_theta = torch.sqrt(1.0 - cos_theta**2)
        return torch.stack(
            [
                sin_theta * torch.cos(phi),
                sin_theta * torch.sin(phi),
                cos_theta,
            ]
        )
    raise ValueError(f"dim must be 1, 2, or 3, got {dim}")


# ---------------------------------------------------------------------------
# Rotation matrices
# ---------------------------------------------------------------------------


def _rotation_matrix_2d(angle):
    """2D rotation matrix from angle (radians). Differentiable."""
    c = torch.cos(angle) if isinstance(angle, torch.Tensor) else math.cos(angle)
    s = torch.sin(angle) if isinstance(angle, torch.Tensor) else math.sin(angle)
    if isinstance(angle, torch.Tensor):
        return torch.stack(
            [
                torch.stack([c, -s]),
                torch.stack([s, c]),
            ]
        )
    return torch.tensor([[c, -s], [s, c]], dtype=torch.float64)


def _rotation_matrix_3d(angles):
    """3D rotation matrix from Tait-Bryan angles (radians). Differentiable.

    Follows GSTools convention: rotations in planes (0,1), (0,2), (1,2).
    """
    if not isinstance(angles, (list, tuple, torch.Tensor)):
        angles = [angles, 0.0, 0.0]
    while len(angles) < 3:
        angles = list(angles) + [0.0]

    R = torch.eye(3, dtype=torch.float64)
    planes = [(0, 1), (0, 2), (1, 2)]
    for ang, (i, j) in zip(angles, planes):
        c = torch.cos(ang) if isinstance(ang, torch.Tensor) else math.cos(ang)
        s = torch.sin(ang) if isinstance(ang, torch.Tensor) else math.sin(ang)
        G = torch.eye(3, dtype=torch.float64)
        if isinstance(ang, torch.Tensor):
            G = G.clone()
            G[i, i] = c
            G[i, j] = -s
            G[j, i] = s
            G[j, j] = c
        else:
            G[i, i] = c
            G[i, j] = -s
            G[j, i] = s
            G[j, j] = c
        R = R @ G
    return R


# ---------------------------------------------------------------------------
# Mesh helpers
# ---------------------------------------------------------------------------

_CELL_TOPO_DIM = {
    "vertex": 0,
    "line": 1,
    "line3": 1,
    "triangle": 2,
    "triangle6": 2,
    "quad": 2,
    "quad8": 2,
    "quad9": 2,
    "tetra": 3,
    "tetra10": 3,
    "hexahedron": 3,
    "hexahedron20": 3,
    "hexahedron27": 3,
    "wedge": 3,
    "wedge15": 3,
    "pyramid": 3,
    "pyramid13": 3,
    "pyramid14": 3,
}


def _cell_centroids(mesh, dim):
    """Cell centroids from the highest-dimensional cell block.

    Gmsh ``.msh`` files typically export both boundary blocks (e.g.\
    ``line`` in 2D, ``triangle`` in 3D) and interior blocks (``triangle``
    in 2D, ``tetra`` in 3D). Taking the highest-topological-dimension
    block picks the interior cells that the user means by "cell
    centroid". If several blocks share the highest dimension (mixed
    triangle + quad in 2D, etc.), centroids from each are concatenated.
    """
    if len(mesh.cells) == 0:
        raise ValueError("Mesh has no cell blocks; cannot extract centroids.")
    best_bd = max(_CELL_TOPO_DIM.get(b.type, -1) for b in mesh.cells)
    if best_bd < 0:
        raise ValueError(
            "Mesh has no recognised cell topology; cannot extract centroids."
        )
    pts = np.asarray(mesh.points)
    chunks = []
    for block in mesh.cells:
        if _CELL_TOPO_DIM.get(block.type, -1) == best_bd:
            verts = pts[block.data]  # (n_cells, n_per_cell, 3)
            chunks.append(verts.mean(axis=1))  # (n_cells, 3)
    centroids = np.concatenate(chunks, axis=0)
    return centroids[:, :dim]


# ---------------------------------------------------------------------------
# Main generator class
# ---------------------------------------------------------------------------


class DifferentiableGRF:
    """
    Differentiable Gaussian random field generator via the spectral method.

    Parameters
    ----------
    kernel : str
        'gau' for Gaussian, 'mat' for Matern.
    dim : int
        Spatial dimensionality (1, 2, or 3).
    variance : float or torch.Tensor
        Field variance. If Tensor with requires_grad, gradients flow.
    len_scale : float, list, or torch.Tensor
        Correlation length(s). Scalar for isotropic, list/tensor for
        per-axis anisotropic lengths.
    seed : int
        Random seed for frozen noise (reproducibility).
    mode_no : int
        Number of Fourier modes.
    angles : float, list, or torch.Tensor
        Rotation angle(s) in radians. Scalar for 2D, list of 3 for 3D.
    nu : float
        Matern smoothness parameter (only used when kernel='mat').
    dtype : torch.dtype
    device : str
    """

    def __init__(
        self,
        kernel,
        dim,
        variance,
        len_scale,
        seed=0,
        mode_no=1000,
        angles=0,
        nu=1.0,
        dtype=torch.float64,
        device="cpu",
    ):
        self.kernel = kernel
        self.dim = dim
        self.variance = variance
        self.mode_no = mode_no
        self.angles = angles
        self.nu = nu
        self.dtype = dtype
        self.device = device

        # Parse len_scale into main scale + anisotropy ratios
        if isinstance(len_scale, (list, tuple)):
            self.len_scale = (
                len_scale[0]
                if not isinstance(len_scale[0], torch.Tensor)
                else len_scale[0]
            )
            self.anis = [
                (
                    ls / self.len_scale
                    if not isinstance(ls, torch.Tensor)
                    else ls / self.len_scale
                )
                for ls in len_scale[1:]
            ]
            self._per_axis = True
        elif isinstance(len_scale, torch.Tensor) and len_scale.ndim > 0:
            self.len_scale = len_scale[0]
            self.anis = [len_scale[i] / len_scale[0] for i in range(1, len(len_scale))]
            self._per_axis = True
        else:
            self.len_scale = len_scale
            self.anis = None
            self._per_axis = False

        # Freeze noise
        rng = torch.Generator(device="cpu").manual_seed(seed)

        self.z1 = torch.randn(mode_no, generator=rng, dtype=dtype)
        self.z2 = torch.randn(mode_no, generator=rng, dtype=dtype)
        self.directions = _sample_sphere(dim, mode_no, rng)

        # Base radii (kernel-dependent, frozen)
        if kernel == "gau":
            if dim == 1:
                self.base_radii = _gaussian_base_radii_1d(mode_no, rng)
            elif dim == 2:
                self.base_radii = _gaussian_base_radii_2d(mode_no, rng)
            elif dim == 3:
                self.base_radii = _gaussian_base_radii_3d(mode_no, rng)
        elif kernel == "mat":
            self.base_radii = _matern_base_radii(mode_no, nu, dim, rng)
        else:
            raise ValueError(f"Unknown kernel '{kernel}'. Use 'gau' or 'mat'.")

        # Move frozen tensors to device and cast to module dtype. Base
        # radii and direction samplers generate in float64 on CPU for
        # numerical reproducibility; cast here so that float32/MPS users
        # get consistent state across devices.
        if device != "cpu":
            self.z1 = self.z1.to(dtype=self.dtype, device=device)
            self.z2 = self.z2.to(dtype=self.dtype, device=device)
            self.directions = self.directions.to(dtype=self.dtype, device=device)
            self.base_radii = self.base_radii.to(dtype=self.dtype, device=device)
        else:
            # Still cast in case user passed a dtype ≠ float64
            self.z1 = self.z1.to(dtype=self.dtype)
            self.z2 = self.z2.to(dtype=self.dtype)
            self.directions = self.directions.to(dtype=self.dtype)
            self.base_radii = self.base_radii.to(dtype=self.dtype)

    def _compute_wave_vectors(self):
        """Compute wave vectors k_m from frozen base + differentiable params."""
        # Spectral radii (differentiable w.r.t. len_scale)
        if self.kernel == "gau":
            radii = gaussian_spectral_transform(self.base_radii, self.len_scale)
        elif self.kernel == "mat":
            radii = matern_spectral_transform(self.base_radii, self.len_scale, self.nu)
        else:
            raise ValueError(f"kernel must be 'gau' or 'mat', got {self.kernel!r}")

        # k = radii * directions  ->  (dim, mode_no)
        k = radii.unsqueeze(0) * self.directions

        # Apply anisotropy (per-axis scaling in spectral space)
        if self._per_axis and self.anis:
            # Scale axes 1+ by 1/anis_i (spectral space is inverse of spatial)
            rows = [k[0]]
            for i, a in enumerate(self.anis):
                rows.append(k[i + 1] / a)
            k = torch.stack(rows)

        # Apply rotation if non-zero
        angles = self.angles
        has_rotation = False
        if isinstance(angles, torch.Tensor):
            has_rotation = True
        elif isinstance(angles, (list, tuple)):
            has_rotation = any(a != 0 for a in angles)
        elif angles != 0:
            has_rotation = True

        if has_rotation:
            if self.dim == 2:
                ang = (
                    angles
                    if isinstance(angles, (float, int, torch.Tensor))
                    else angles[0]
                )
                R = _rotation_matrix_2d(ang)
            else:
                R = _rotation_matrix_3d(angles)
            if isinstance(R, torch.Tensor):
                R = R.to(dtype=self.dtype, device=self.device)
            k = R.T @ k

        return k

    def _summate(self, k, pos):
        """Core spectral summation. k: (dim, M), pos: (dim, N). Returns (N,)."""
        variance = self.variance
        if not isinstance(variance, torch.Tensor):
            variance = torch.tensor(variance, dtype=self.dtype, device=self.device)

        amp = torch.sqrt(variance / self.mode_no)

        # phase = k^T @ pos  ->  (M, N)
        phase = k.T @ pos

        # field = amp * (z1 @ cos(phase) + z2 @ sin(phase))
        field = amp * (self.z1 @ torch.cos(phase) + self.z2 @ torch.sin(phase))
        return field

    def structured(self, grid_dim):
        """Generate field on a regular grid.

        Parameters
        ----------
        grid_dim : list of int
            Grid dimensions, e.g. [100, 100] or [50, 50, 50].

        Returns
        -------
        field : Tensor, shape (*grid_dim)
        """
        axes = [torch.arange(n, dtype=self.dtype, device=self.device) for n in grid_dim]
        grids = torch.meshgrid(*axes, indexing="ij")
        pos = torch.stack([g.reshape(-1) for g in grids])

        k = self._compute_wave_vectors()
        field = self._summate(k, pos)
        return field.reshape(*grid_dim)

    def unstructured(self, points):
        """Generate field at arbitrary spatial points.

        Parameters
        ----------
        points : Tensor or ndarray, shape (n_points, dim)

        Returns
        -------
        field : Tensor, shape (n_points,)
        """
        if not isinstance(points, torch.Tensor):
            points = torch.tensor(points, dtype=self.dtype, device=self.device)
        else:
            points = points.to(dtype=self.dtype, device=self.device)
        pos = points.T

        k = self._compute_wave_vectors()
        return self._summate(k, pos)

    def on_mesh(self, mesh, location="points"):
        """Evaluate the field on a meshio mesh or a file that meshio can read.

        Parameters
        ----------
        mesh : meshio.Mesh | str | pathlib.Path
            A ``meshio.Mesh`` object, or a filesystem path to any
            meshio-supported mesh file (``.msh`` from gmsh, ``.vtk``,
            ``.xdmf``, etc.). Paths are read via ``meshio.read``.
        location : {'points', 'cells'}
            Where to evaluate the field. ``'points'`` returns one value
            per mesh vertex; ``'cells'`` returns one value per cell
            centroid (computed from the first cell block).

        Returns
        -------
        field : Tensor, shape (n_sites,)
            ``n_sites`` is the number of vertices (``location='points'``)
            or cells (``location='cells'``).

        Notes
        -----
        Requires the optional ``meshio`` dependency. Install with
        ``pip install diffgrf[mesh]``.
        """
        try:
            import meshio
        except ImportError as e:
            raise ImportError(
                "diffgrf.on_mesh requires the optional `meshio` "
                "dependency. Install with: pip install diffgrf[mesh]"
            ) from e

        if isinstance(mesh, (str, pathlib.Path)):
            mesh = meshio.read(str(mesh))

        if location == "points":
            pts = np.asarray(mesh.points)[:, : self.dim]
        elif location == "cells":
            pts = _cell_centroids(mesh, self.dim)
        else:
            raise ValueError(f"location must be 'points' or 'cells', got {location!r}")

        return self.unstructured(pts)

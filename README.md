# DiffGRF

**Differentiable Gaussian random field generation in PyTorch.**

DiffGRF implements the spectral randomization method for Gaussian random field
(GRF) synthesis (Kraichnan 1970; Hesse et al. 2014) with full PyTorch autograd
support. Gradients flow analytically through every physical parameter —
variance, correlation length(s), anisotropy, rotation angles and the Matérn
smoothness — so the generator can be used as a differentiable building block in
inverse problems, design optimisation, and torch-native training pipelines.

## Features

- Gaussian and Matérn kernels
- 1D, 2D, and 3D spatial domains
- Isotropic or per-axis anisotropic correlation lengths
- Differentiable rotation (2D angle, 3D Tait–Bryan triplet)
- Structured (regular grid), unstructured (arbitrary points), and
  mesh-native (`meshio` / gmsh `.msh`) sampling
- CPU, CUDA, and Apple MPS device support
- End-to-end autograd through all physical parameters

## Installation

```bash
pip install diffgrf
```

Or from source:

```bash
git clone https://github.com/EJRicketts/diffgrf.git
cd diffgrf
pip install -e .[dev]
```

Requires Python ≥ 3.10, PyTorch ≥ 2.0, NumPy ≥ 1.22.

GPU acceleration is available via `device="cuda"` (NVIDIA) or
`device="mps"` (Apple Silicon). MPS does not support `float64`, so
pass `dtype=torch.float32` when using MPS:

```python
import torch
from diffgrf import DifferentiableGRF

grf = DifferentiableGRF(
    kernel="gau", dim=2, variance=1.0, len_scale=10.0, seed=0,
    device="mps", dtype=torch.float32,
)
field = grf.structured([256, 256])
```

Observed speedups on Apple M-series over CPU: 4–5× at 128²–256² 2D,
~10× at 96³ 3D.

## Quickstart

Generate a 2D Gaussian random field on a regular grid:

```python
from diffgrf import DifferentiableGRF

grf = DifferentiableGRF(
    kernel="gau", dim=2, variance=1.0, len_scale=10.0, seed=0,
)
field = grf.structured([128, 128])
# field is a torch.Tensor of shape (128, 128)
```

Flow gradients through the correlation length:

```python
import torch
from diffgrf import DifferentiableGRF

len_scale = torch.tensor(10.0, requires_grad=True)
grf = DifferentiableGRF(
    kernel="gau", dim=2, variance=1.0, len_scale=len_scale, seed=0,
)
field = grf.structured([128, 128])
loss = field.var()
loss.backward()
print(len_scale.grad)   # analytical gradient via spectral reparameterisation
```

Evaluate a field directly on an unstructured mesh (requires `meshio`,
`pip install diffgrf[mesh]`):

```python
import meshio
from diffgrf import DifferentiableGRF

grf = DifferentiableGRF(
    kernel="mat", dim=2, variance=1.0, len_scale=5.0, nu=1.5, seed=0,
)
# Accepts a meshio.Mesh OR any path readable by meshio.read.
# Evaluate at vertices (default) or cell centroids.
field_at_nodes = grf.on_mesh(meshio.read("domain.msh"))
field_at_cells = grf.on_mesh("domain.msh", location="cells")
```

Recover a correlation length from a target sample by gradient descent:

```python
import torch
from diffgrf import DifferentiableGRF

target = DifferentiableGRF(
    "gau", 2, 1.0, 12.0, seed=0,
).structured([128, 128])

ell = torch.tensor(5.0, requires_grad=True)
opt = torch.optim.Adam([ell], lr=0.2)
for _ in range(200):
    opt.zero_grad()
    pred = DifferentiableGRF("gau", 2, 1.0, ell, seed=0).structured([128, 128])
    loss = (pred - target).pow(2).mean()
    loss.backward()
    opt.step()
print(float(ell))   # -> ~12
```

## API

### `DifferentiableGRF(kernel, dim, variance, len_scale, ...)`

The single entry point. See `diffgrf/core.py` for full signature and
docstrings.

Methods:

- `.structured(grid_dim)` — generate on a regular grid.
- `.unstructured(points)` — generate at arbitrary spatial points.
- `.on_mesh(mesh, location='points')` — generate on a `meshio.Mesh`
  or a path to any meshio-compatible file (`.msh`, `.vtk`, `.xdmf`,
  ...); evaluate at vertices (`location='points'`) or cell centroids
  (`location='cells'`). Requires the optional `meshio` dependency
  (`pip install diffgrf[mesh]`).

## Citation

If you use DiffGRF in published work, please cite:

```bibtex
@software{diffgrf,
  author = {Ricketts, Evan John},
  title  = {DiffGRF: Differentiable Gaussian random field generation in PyTorch},
  year   = {2026},
  url    = {https://github.com/EJRicketts/diffgrf},
}
```

A peer-reviewed SoftwareX paper describing DiffGRF is in preparation; the
citation will be updated once published.

## License

MIT. See [LICENSE](LICENSE).

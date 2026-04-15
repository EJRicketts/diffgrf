"""Mesh path: meshio.Mesh objects, file paths, cell centroids, autograd."""

import sys

import numpy as np
import pytest
import torch

from diffgrf import DifferentiableGRF

meshio = pytest.importorskip("meshio")


def _tri_mesh():
    """A single-triangle mesh with three vertices in the xy-plane."""
    points = np.array(
        [
            [0.0, 0.0, 0.0],
            [1.0, 0.0, 0.0],
            [0.0, 1.0, 0.0],
        ]
    )
    cells = [("triangle", np.array([[0, 1, 2]]))]
    return meshio.Mesh(points=points, cells=cells)


def test_on_mesh_from_meshio_object():
    mesh = _tri_mesh()
    grf = DifferentiableGRF("gau", 2, 1.0, 10.0, seed=0, mode_no=200)
    f = grf.on_mesh(mesh)
    assert f.shape == (3,)
    expected = grf.unstructured(mesh.points[:, :2])
    assert torch.allclose(f, expected, atol=1e-10)


def test_on_mesh_from_path(tmp_path):
    mesh = _tri_mesh()
    path = tmp_path / "tri.msh"
    meshio.write(str(path), mesh, file_format="gmsh")

    grf = DifferentiableGRF("gau", 2, 1.0, 10.0, seed=0, mode_no=200)
    f = grf.on_mesh(str(path))
    assert f.shape == (3,)
    assert torch.isfinite(f).all()


def test_on_mesh_cells():
    mesh = _tri_mesh()
    grf = DifferentiableGRF("gau", 2, 1.0, 10.0, seed=0, mode_no=200)
    f = grf.on_mesh(mesh, location="cells")
    # One triangle → one centroid → one field value.
    assert f.shape == (1,)


def test_on_mesh_cells_mixed_topology():
    """gmsh .msh exports interleave boundary (line) + interior (triangle)
    blocks. Centroids must come from the highest-dim block (triangles),
    not the boundary segments."""
    points = np.array(
        [
            [0.0, 0.0, 0.0],
            [1.0, 0.0, 0.0],
            [0.0, 1.0, 0.0],
        ]
    )
    mesh = meshio.Mesh(
        points=points,
        cells=[
            ("line", np.array([[0, 1], [1, 2], [2, 0]])),
            ("triangle", np.array([[0, 1, 2]])),
        ],
    )
    grf = DifferentiableGRF("gau", 2, 1.0, 10.0, seed=0, mode_no=200)
    f = grf.on_mesh(mesh, location="cells")
    # One triangle (the highest-dim block); the 3 boundary lines are skipped.
    assert f.shape == (1,)


def test_on_mesh_gradient():
    mesh = _tri_mesh()
    ell = torch.tensor(10.0, requires_grad=True, dtype=torch.float64)
    grf = DifferentiableGRF("gau", 2, 1.0, ell, seed=0, mode_no=200)
    f = grf.on_mesh(mesh)
    (f**2).mean().backward()
    assert ell.grad is not None and torch.isfinite(ell.grad)


def test_on_mesh_missing_meshio(monkeypatch):
    """Informative ImportError when meshio is absent."""
    monkeypatch.setitem(sys.modules, "meshio", None)
    grf = DifferentiableGRF("gau", 2, 1.0, 10.0, seed=0, mode_no=200)
    with pytest.raises(ImportError, match="meshio"):
        grf.on_mesh("dummy.msh")

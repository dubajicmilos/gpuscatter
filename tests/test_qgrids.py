"""Tests for q-grid construction helpers."""
import numpy as np

from gpuscatter import make_qgrid_HK_plane, make_qgrid_BZ


def test_HK_plane_shape():
    h, q, sh = make_qgrid_HK_plane(L_value=1.5, a_cub=6.0, n_grid=11)
    assert h.shape == (11,)
    assert q.shape == (121, 3)
    assert sh == (11, 11)


def test_HK_plane_L_constant():
    """All q-points should have the same L (qz)."""
    L = 1.5
    a = 6.0
    h, q, sh = make_qgrid_HK_plane(L_value=L, a_cub=a, n_grid=11)
    expected_qz = L * 2 * np.pi / a
    assert np.allclose(q[:, 2], expected_qz)


def test_HK_plane_origin():
    """The center q-point of an odd-grid should be (0, 0, qz_L)."""
    h, q, _ = make_qgrid_HK_plane(L_value=1.5, a_cub=6.0, n_grid=11)
    center = q[len(q) // 2]
    assert abs(center[0]) < 1e-6
    assert abs(center[1]) < 1e-6


def test_BZ_grid():
    n_cells = 6
    a = 6.0
    q1d, q_red, q_cart, sh = make_qgrid_BZ(n_cells, a)
    assert q1d.shape == (n_cells,)
    assert q_red.shape == (3, n_cells ** 3)
    assert q_cart.shape == (n_cells ** 3, 3)
    assert sh == (n_cells, n_cells, n_cells)


def test_BZ_grid_in_range():
    """Reduced q values should be in [-1/2, 1/2)."""
    q1d, _, _, _ = make_qgrid_BZ(8, 6.0)
    assert q1d.min() == -0.5
    assert q1d.max() < 0.5

"""Tests for the dispersion-projection module."""
import numpy as np
import pytest

from gpuscatter import (
    HIGH_SYMMETRY_POINTS_CUBIC, make_path_indices, project_dispersion,
)


def test_high_sym_points():
    assert np.allclose(HIGH_SYMMETRY_POINTS_CUBIC['Gamma'], [0, 0, 0])
    assert np.allclose(HIGH_SYMMETRY_POINTS_CUBIC['X'], [0.5, 0, 0])
    assert np.allclose(HIGH_SYMMETRY_POINTS_CUBIC['M'], [0.5, 0.5, 0])
    assert np.allclose(HIGH_SYMMETRY_POINTS_CUBIC['R'], [0.5, 0.5, 0.5])


def test_path_indices_shape():
    n_cells = 24
    pts = [HIGH_SYMMETRY_POINTS_CUBIC[s]
           for s in ['Gamma', 'X', 'M', 'R', 'Gamma']]
    q_red, q_idx, breaks = make_path_indices(pts, n_cells, n_per_seg=11)
    assert q_red.shape[1] == 3
    assert len(q_idx) == q_red.shape[0]
    assert breaks[0] == 0
    assert breaks[-1] == q_red.shape[0] - 1


def test_project_dispersion_correct_shape():
    n_cells = 8
    n_omega = 50
    S = np.random.rand(n_cells ** 3, n_omega).astype(np.float32)
    pts = [HIGH_SYMMETRY_POINTS_CUBIC[s]
           for s in ['Gamma', 'X', 'M']]
    S_path, q_red, breaks = project_dispersion(
        S, (n_cells,) * 3, pts, n_per_seg=5
    )
    assert S_path.shape[1] == n_omega
    assert S_path.shape[0] == q_red.shape[0]


def test_project_dispersion_4D_input():
    n_cells = 8
    n_omega = 30
    S = np.random.rand(n_cells, n_cells, n_cells, n_omega).astype(np.float32)
    pts = [HIGH_SYMMETRY_POINTS_CUBIC[s] for s in ['Gamma', 'R']]
    S_path, _, _ = project_dispersion(S, (n_cells,) * 3, pts, n_per_seg=4)
    assert S_path.shape[1] == n_omega


def test_project_dispersion_wrong_dim():
    """1D input is rejected."""
    S = np.random.rand(50)
    pts = [HIGH_SYMMETRY_POINTS_CUBIC[s] for s in ['Gamma', 'X']]
    with pytest.raises(ValueError, match='must be'):
        project_dispersion(S, (5, 5, 5), pts)

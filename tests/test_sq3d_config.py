"""Tests for Sq3DConfig and Sq3DResult helpers (no GPU required)."""
import numpy as np
import pytest

from gpuscatter import Sq3DConfig, Sq3DResult


def test_n_total_basic():
    cfg = Sq3DConfig(n_cells=24, n_voxels_per_cell=8)
    assert cfg.n_total == 192


def test_n_total_oversampled():
    cfg = Sq3DConfig(n_cells=24, n_voxels_per_cell=16)
    assert cfg.n_total == 384


def test_q_max_clean_default():
    """q_max_clean = 0.85 * n_voxels_per_cell / 2.

    For the headline-demo n_voxels_per_cell=8, q_Nyq = 4 r.l.u. and
    q_max_clean = 3.4 r.l.u.
    """
    res = Sq3DResult(
        h_arr=np.zeros(192, dtype=np.float32),
        k_arr=np.zeros(192, dtype=np.float32),
        L_arr=np.zeros(97, dtype=np.float32),
        a_cub=6.187, L_box=148.5, n_frames=10, n_cells=24,
        n_voxels_per_cell=8, n_regions=1, method='test',
    )
    assert res.q_max_clean == pytest.approx(3.4, rel=1e-6)


def test_q_max_clean_oversampled():
    """Doubling n_voxels_per_cell doubles q_max_clean."""
    res = Sq3DResult(
        h_arr=np.zeros(384, dtype=np.float32),
        k_arr=np.zeros(384, dtype=np.float32),
        L_arr=np.zeros(193, dtype=np.float32),
        a_cub=6.187, L_box=148.5, n_frames=10, n_cells=24,
        n_voxels_per_cell=16, n_regions=1, method='test',
    )
    assert res.q_max_clean == pytest.approx(6.8, rel=1e-6)


def test_q_max_clean_below_q_nyquist():
    """q_max_clean must always be strictly below q_Nyquist."""
    for n_v in (4, 8, 10, 16, 32):
        res = Sq3DResult(
            h_arr=np.zeros(2, dtype=np.float32),
            k_arr=np.zeros(2, dtype=np.float32),
            L_arr=np.zeros(2, dtype=np.float32),
            a_cub=6.0, L_box=24.0, n_frames=1, n_cells=4,
            n_voxels_per_cell=n_v, n_regions=1, method='test',
        )
        q_nyq = n_v / 2
        assert res.q_max_clean < q_nyq
        assert res.q_max_clean > 0.5 * q_nyq  # but not absurdly conservative

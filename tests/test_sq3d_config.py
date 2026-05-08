"""Tests for Sq3DConfig and Sq3DResult helpers (no GPU required)."""
import numpy as np
import pytest

from gpuscatter import Sq3DConfig, Sq3DResult


def _make_dummy_result(n_cells: int = 24, n_voxels_per_cell: int = 8,
                       a_cub: float = 6.187):
    """Build a fake :class:`Sq3DResult` whose arrays match the FFT layout.

    ``h_arr`` runs from -N/2 to N/2 - 1 in steps of 1/n_cells r.l.u.;
    ``L_arr`` runs from 0 to N/2 in steps of 1/n_cells (rfft half-spectrum).
    ``total`` and one fake partial are filled with a known pattern so we
    can verify trimming preserves values.
    """
    N = n_cells * n_voxels_per_cell
    h_arr = (np.arange(-N // 2, N // 2) / n_cells).astype(np.float32)
    L_arr = (np.arange(0, N // 2 + 1) / n_cells).astype(np.float32)
    H, K, L = np.meshgrid(h_arr, h_arr, L_arr, indexing='ij')
    total = (H * 100 + K + L * 0.01).astype(np.float32)
    partial = (total * 0.5).astype(np.float32)
    return Sq3DResult(
        h_arr=h_arr, k_arr=h_arr.copy(), L_arr=L_arr,
        a_cub=a_cub, L_box=n_cells * a_cub,
        n_frames=10, n_cells=n_cells,
        n_voxels_per_cell=n_voxels_per_cell, n_regions=1,
        method='dummy',
        partials={('Pb', 'Pb'): partial},
        total=total,
    )


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


# ---------------------------------------------------------------------------
# trim() tests
# ---------------------------------------------------------------------------
def test_trim_default_uses_q_max_clean():
    """Default q_max should match q_max_clean."""
    res = _make_dummy_result(n_cells=24, n_voxels_per_cell=8)
    trimmed = res.trim()
    # q_max_clean = 0.85 * 4 = 3.4 r.l.u.
    assert trimmed.h_arr.max() <= res.q_max_clean + 1e-6
    assert (-trimmed.h_arr.min()) <= res.q_max_clean + 1e-6
    assert trimmed.L_arr.max() <= res.q_max_clean + 1e-6


def test_trim_custom_q_max():
    res = _make_dummy_result(n_cells=24, n_voxels_per_cell=8)
    trimmed = res.trim(q_max=2.0)
    assert trimmed.h_arr.max() <= 2.0 + 1e-6
    assert (-trimmed.h_arr.min()) <= 2.0 + 1e-6
    assert trimmed.L_arr.max() <= 2.0 + 1e-6


def test_trim_preserves_values():
    """Values inside the kept region must be unchanged."""
    res = _make_dummy_result(n_cells=24, n_voxels_per_cell=8)
    trimmed = res.trim(q_max=1.5)
    # spot-check: at the center (H=0, K=0, L=0)
    h0 = int(np.argmin(np.abs(res.h_arr)))
    L0 = int(np.argmin(np.abs(res.L_arr)))
    h0_t = int(np.argmin(np.abs(trimmed.h_arr)))
    L0_t = int(np.argmin(np.abs(trimmed.L_arr)))
    assert trimmed.total[h0_t, h0_t, L0_t] == res.total[h0, h0, L0]


def test_trim_shapes_consistent():
    """Trimmed arrays must have shapes matching trimmed axes."""
    res = _make_dummy_result(n_cells=24, n_voxels_per_cell=8)
    trimmed = res.trim(q_max=2.0)
    nh, nk, nL = trimmed.h_arr.size, trimmed.k_arr.size, trimmed.L_arr.size
    assert trimmed.total.shape == (nh, nk, nL)
    for S in trimmed.partials.values():
        assert S.shape == (nh, nk, nL)


def test_trim_preserves_metadata():
    res = _make_dummy_result(n_cells=24, n_voxels_per_cell=8)
    trimmed = res.trim(q_max=2.0)
    assert trimmed.a_cub == res.a_cub
    assert trimmed.L_box == res.L_box
    assert trimmed.n_frames == res.n_frames
    assert trimmed.n_cells == res.n_cells
    assert trimmed.n_voxels_per_cell == res.n_voxels_per_cell
    # q_max_clean should still reflect the original grid
    assert trimmed.q_max_clean == res.q_max_clean


def test_trim_method_string_records_q_max():
    res = _make_dummy_result(n_cells=24, n_voxels_per_cell=8)
    trimmed = res.trim(q_max=2.5)
    assert 'trimmed' in trimmed.method
    assert '2.5' in trimmed.method


def test_trim_rejects_q_max_above_nyquist():
    res = _make_dummy_result(n_cells=24, n_voxels_per_cell=8)
    # q_Nyquist = 4 r.l.u.; 5 should raise.
    with pytest.raises(ValueError):
        res.trim(q_max=5.0)


def test_trim_rejects_non_positive_q_max():
    res = _make_dummy_result(n_cells=24, n_voxels_per_cell=8)
    with pytest.raises(ValueError):
        res.trim(q_max=0.0)
    with pytest.raises(ValueError):
        res.trim(q_max=-1.0)


def test_trim_idempotent_at_q_max_clean():
    """Calling trim() twice with the same q_max should be a no-op the
    second time."""
    res = _make_dummy_result(n_cells=24, n_voxels_per_cell=8)
    once = res.trim()
    twice = once.trim(q_max=once.h_arr.max())
    assert once.h_arr.size == twice.h_arr.size
    assert once.L_arr.size == twice.L_arr.size
    assert np.array_equal(once.total, twice.total)

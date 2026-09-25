"""Tests for Sq3DResult.save_rspace3d (no GPU required; needs h5py)."""
import dataclasses

import numpy as np
import pytest

from gpuscatter import Sq3DResult

h5py = pytest.importorskip('h5py')

A_CUB = 6.0
PEAK = 1000.0
PEAK_HKL = (0.5, -0.75, 1.0)        # l > 0 and inside |q| <= 1.5


def _index(axis, value):
    return int(np.flatnonzero(np.isclose(axis, value))[0])


def _make_result():
    """Sq3DResult on the Sq3D grid for n_cells = 4, n_voxels_per_cell = 4.

    H and K run over the untrimmed [-N/2, N/2) grid, arange(-8, 8) / 4, and
    L over the rfft half-spectrum, arange(0, 9) / 4. ``total`` holds small
    random values and one peak at PEAK_HKL; there is one partial ('A', 'B').
    """
    rng = np.random.default_rng(0)
    h = (np.arange(-8, 8) / 4).astype(np.float32)
    L = (np.arange(0, 9) / 4).astype(np.float32)
    total = rng.uniform(0.0, 1.0, (h.size, h.size, L.size)).astype(np.float32)
    # Sq3D output obeys S(-q) = S(q) inside the L = 0 plane (rfft of a real
    # field). Impose it where -q is on the grid (indices 1..15 pair with
    # 15..1), so that the Friedel check below can cover every voxel.
    plane = total[1:, 1:, 0]
    total[1:, 1:, 0] = 0.5 * (plane + plane[::-1, ::-1])
    total[_index(h, PEAK_HKL[0]), _index(h, PEAK_HKL[1]),
          _index(L, PEAK_HKL[2])] = PEAK
    partial = rng.uniform(-1.0, 1.0, total.shape).astype(np.float32)
    return Sq3DResult(
        h_arr=h, k_arr=h.copy(), L_arr=L,
        a_cub=A_CUB, L_box=4 * A_CUB,
        n_frames=10, n_cells=4, n_voxels_per_cell=4, n_regions=1,
        method='test method',
        partials={('A', 'B'): partial},
        total=total,
    )


def test_half_spectrum_datasets_and_attributes(tmp_path):
    """full_l=False writes the rspace3d layout and the data unchanged."""
    res = _make_result()
    path = tmp_path / 'half.h5'
    res.save_rspace3d(path, full_l=False, wavelength=0.71)

    with h5py.File(path, 'r') as f:
        assert set(f.keys()) == {'data', 'H', 'K', 'L', 'UB'}
        assert f['data'].dtype == np.float32
        assert f['data'].shape == (16, 16, 9)
        np.testing.assert_array_equal(f['data'][...], res.total)
        for name, axis in (('H', res.h_arr), ('K', res.k_arr), ('L', res.L_arr)):
            assert f[name].dtype == np.float64
            assert f[name].shape == axis.shape
            np.testing.assert_array_equal(f[name][...], axis.astype(np.float64))
        for name in ('data', 'H', 'K', 'L'):
            assert f[name].compression == 'gzip'
            assert f[name].compression_opts == 4
        assert f['UB'].dtype == np.float64
        assert f['UB'].shape == (3, 3)
        np.testing.assert_array_equal(f['UB'][...], 0.71 * np.eye(3) / A_CUB)
        attrs = dict(f.attrs)

    assert set(attrs) == {
        'wavelength', 'grid_kind', 'plane_type',
        'cell_a', 'cell_b', 'cell_c', 'cell_alpha', 'cell_beta', 'cell_gamma',
        'gpuscatter_channel', 'gpuscatter_method', 'gpuscatter_n_frames',
    }
    assert isinstance(attrs['wavelength'], np.floating)
    assert attrs['wavelength'] == 0.71
    assert attrs['grid_kind'] == 'hkl_regular'
    assert attrs['plane_type'] == 'HK'
    for key in ('cell_a', 'cell_b', 'cell_c'):
        assert isinstance(attrs[key], np.floating)
        assert attrs[key] == A_CUB
    for key in ('cell_alpha', 'cell_beta', 'cell_gamma'):
        assert isinstance(attrs[key], np.floating)
        assert attrs[key] == 90.0
    assert attrs['gpuscatter_channel'] == 'total'
    assert attrs['gpuscatter_method'] == 'test method'
    assert isinstance(attrs['gpuscatter_n_frames'], np.integer)
    assert attrs['gpuscatter_n_frames'] == 10


def test_full_l_needs_trim_and_is_friedel_symmetric(tmp_path):
    """full_l=True rejects the untrimmed grid and expands a trimmed one."""
    res = _make_result()
    untrimmed = tmp_path / 'untrimmed.h5'
    with pytest.raises(ValueError, match=r'\.trim\(\)'):
        res.save_rspace3d(untrimmed)
    assert not untrimmed.exists()

    clean = res.trim(q_max=1.5)
    path = tmp_path / 'full.h5'
    clean.save_rspace3d(path)
    with h5py.File(path, 'r') as f:
        data, H, K, L = (f[name][...] for name in ('data', 'H', 'K', 'L'))

    axis = np.arange(-6, 7) / 4                          # -1.5 .. 1.5
    np.testing.assert_array_equal(H, axis)
    np.testing.assert_array_equal(K, axis)
    np.testing.assert_array_equal(L, axis)
    np.testing.assert_array_equal(L, -L[::-1])
    assert data.dtype == np.float32
    assert data.shape == (13, 13, 13)
    # The L >= 0 half is the trimmed input as it is.
    np.testing.assert_array_equal(data[:, :, 6:], clean.total)
    # Friedel symmetry for every voxel: data[h, k, l] == data[-h, -k, -l].
    np.testing.assert_array_equal(data, data[::-1, ::-1, ::-1])
    # The peak sits at (h, k, l) and at (-h, -k, -l), and nowhere else.
    h0, k0, l0 = PEAK_HKL
    assert data[_index(H, h0), _index(K, k0), _index(L, l0)] == PEAK
    assert data[_index(H, -h0), _index(K, -k0), _index(L, -l0)] == PEAK
    assert np.count_nonzero(data == PEAK) == 2


def test_channel_selection_and_missing_channels(tmp_path):
    res = _make_result()
    path = tmp_path / 'AB.h5'
    res.save_rspace3d(path, channel=('A', 'B'), full_l=False, compression=None)
    with h5py.File(path, 'r') as f:
        np.testing.assert_array_equal(f['data'][...], res.partials[('A', 'B')])
        assert f.attrs['gpuscatter_channel'] == 'AB'
        assert f['data'].compression is None

    with pytest.raises(ValueError, match='not in this result'):
        res.save_rspace3d(tmp_path / 'AC.h5', channel=('A', 'C'), full_l=False)
    no_total = dataclasses.replace(res, total=None)
    with pytest.raises(ValueError, match='no total'):
        no_total.save_rspace3d(tmp_path / 'total.h5', full_l=False)


def test_float32_axes_are_written_exactly_on_the_grid(tmp_path):
    """Sq3D keeps its axes in float32; the file gets exact i / n_cells.

    xrays-on-detector rejects an axis whose step varies by more than 1e-6
    (relative). For n_cells = 24 a plain float64 cast of the float32 axes
    varies by more than that, so the writer rebuilds them from the grid.
    """
    n_cells, N = 24, 96                                  # n_voxels_per_cell = 4
    h = (np.arange(-N // 2, N // 2) / n_cells).astype(np.float32)
    L = (np.arange(0, N // 2 + 1) / n_cells).astype(np.float32)
    res = Sq3DResult(
        h_arr=h, k_arr=h.copy(), L_arr=L,
        a_cub=6.19, L_box=n_cells * 6.19,
        n_frames=1, n_cells=n_cells, n_voxels_per_cell=4, n_regions=1,
        method='test',
        total=np.ones((h.size, h.size, L.size), dtype=np.float32),
    )
    clean = res.trim(q_max=1.7)                          # |i| <= 40
    cast = np.diff(clean.h_arr.astype(np.float64))
    assert not np.allclose(cast, cast[0], rtol=1e-6, atol=0.0)

    path = tmp_path / 'grid.h5'
    clean.save_rspace3d(path)
    with h5py.File(path, 'r') as f:
        H, K, L_full = (f[name][...] for name in ('H', 'K', 'L'))
    axis = np.arange(-40, 41) / n_cells
    for written in (H, K, L_full):
        np.testing.assert_array_equal(written, axis)
        step = np.diff(written)
        assert np.allclose(step, step[0], rtol=1e-6, atol=0.0)

    off_grid = dataclasses.replace(_make_result(), n_cells=3)
    with pytest.raises(ValueError, match='1/n_cells grid'):
        off_grid.save_rspace3d(tmp_path / 'off.h5', full_l=False)

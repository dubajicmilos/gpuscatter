"""Tests for Laue-group symmetry averaging (no GPU required)."""
import numpy as np
import pytest

from gpuscatter import (
    symmetrize_volume, expand_rfft_L, get_symmetry_operations,
)

EXPECTED_ORDERS = {
    '-1': 2, '2/m': 4, 'mmm': 8, '4/m': 8, '4/mmm': 16,
    'm-3': 24, 'm-3m': 48,
}


def test_group_orders_and_orthogonality():
    for g, n in EXPECTED_ORDERS.items():
        ops = get_symmetry_operations(g)
        assert len(ops) == n, f'{g}: got {len(ops)} expected {n}'
        for op in ops:
            # every op is an orthogonal signed permutation of (h, k, l)
            assert np.array_equal(op @ op.T, np.eye(3, dtype=int))
        # all ops distinct
        assert len({tuple(o.flatten()) for o in ops}) == n


def test_unknown_group_raises():
    with pytest.raises(ValueError):
        get_symmetry_operations('6/mmm')


def test_synthetic_mmm_orbit_mean():
    # symmetric 5-grid; mmm orbit of a voxel = its 8 sign-flip partners.
    ax = np.arange(-2, 3, dtype=float)
    rng = np.random.default_rng(0)
    v = rng.random((5, 5, 5)).astype(np.float32)
    sym = symmetrize_volume(ax, ax, ax, v, 'mmm', sigma=None)
    i, j, k = 3, 4, 1                          # coords (1, 2, -1)
    fi, fj, fk = 4 - i, 4 - j, 4 - k           # sign-flip index = n-1-m, n=5
    orbit = [(a, b, c) for a in (i, fi) for b in (j, fj) for c in (k, fk)]
    manual = np.mean([v[a, b, c] for (a, b, c) in orbit])
    assert sym[i, j, k] == pytest.approx(manual, abs=1e-6)
    # an mmm-averaged volume is invariant under each independent sign flip
    assert np.allclose(sym, sym[::-1], atol=1e-6)
    assert np.allclose(sym, sym[:, ::-1], atol=1e-6)
    assert np.allclose(sym, sym[:, :, ::-1], atol=1e-6)


def test_expand_rfft_L_friedel_roundtrip():
    h = (np.arange(-3, 4) / 3.0)               # symmetric about 0
    L = (np.arange(0, 4) / 3.0)                # rfft half-spectrum, starts at 0
    rng = np.random.default_rng(2)
    half = rng.random((h.size, h.size, L.size)).astype(np.float32)
    # a real |F|^2 spectrum has a centrosymmetric L=0 plane (|F(q)|=|F(-q)|);
    # only then is the Friedel-expanded volume fully centrosymmetric.
    half[:, :, 0] = 0.5 * (half[:, :, 0] + half[::-1, ::-1, 0])
    hf, kf, Lf, full = expand_rfft_L(h, h, L, half)
    assert full.shape == (h.size, h.size, 2 * L.size - 1)
    assert np.allclose(Lf, -Lf[::-1])          # symmetric L axis
    # L >= 0 half is unchanged; whole volume is centrosymmetric (Friedel)
    assert np.array_equal(full[:, :, L.size - 1:], half)
    assert np.allclose(full, full[::-1, ::-1, ::-1])


def test_expand_rfft_L_rejects_bad_axes():
    L = np.arange(0, 4) / 3.0
    bad = np.arange(0, 7) / 3.0                # not symmetric about 0
    good = (np.arange(-3, 4) / 3.0)
    vol = np.zeros((good.size, good.size, L.size), dtype=np.float32)
    with pytest.raises(ValueError):
        expand_rfft_L(bad, good, L, vol)
    with pytest.raises(ValueError):
        expand_rfft_L(good, good, good, vol)   # L must start at 0


def _synthetic_full_volume(n=6, seed=1):
    ax = np.arange(-n, n + 1, dtype=float)     # symmetric, spans +-q
    rng = np.random.default_rng(seed)
    vol = rng.random((ax.size, ax.size, ax.size)).astype(np.float32)
    return ax, vol


def test_4mmm_idempotent_and_invariant():
    ax, vol = _synthetic_full_volume()
    sym = symmetrize_volume(ax, ax, ax, vol, '4/mmm', sigma=None)
    # idempotent: symmetrizing the symmetric result changes nothing
    sym2 = symmetrize_volume(ax, ax, ax, sym, '4/mmm', sigma=None)
    denom = np.abs(sym).max()
    assert np.abs(sym2 - sym).max() / denom < 1e-5
    # invariant under the 4-fold about c (rot90 in the H,K plane) and L-mirror
    assert np.abs(np.rot90(sym, 1, axes=(0, 1)) - sym).max() / denom < 1e-4
    assert np.abs(sym[:, :, ::-1] - sym).max() / denom < 1e-4


def test_averaging_reduces_orbit_scatter():
    ax, vol = _synthetic_full_volume()
    sym = symmetrize_volume(ax, ax, ax, vol, '4/mmm', sigma=None)
    # the symmetric result has near-zero spread across its own orbit
    rot = np.rot90(sym, 1, axes=(0, 1))
    assert np.std(rot - sym) < 0.01 * np.std(vol)

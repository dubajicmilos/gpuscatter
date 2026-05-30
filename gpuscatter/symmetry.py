"""Laue-group symmetry averaging of 3D reciprocal-space volumes.

Ported from rspace3d (https://github.com/dubajicmilos/rspace3d): for every
voxel, gather the intensities at all symmetry-equivalent positions (the
*orbit*) and average them. Folding the ``N_ops`` crystallographically
equivalent copies together cuts the per-voxel noise of a ``<|F|^2>``-type
diffuse map by up to ``sqrt(N_ops)``, without changing the q-grid (contrast
with merely computing on a finer grid, which leaves per-pixel noise intact).

The averaging is exact on a regular reciprocal grid: each Laue operation of
the cubic/tetragonal/orthorhombic groups is a *signed permutation* of
``(h, k, l)``, so it maps grid nodes onto grid nodes with no interpolation,
provided the grid spans both ``+q`` and ``-q`` for every axis the group
flips. Use :func:`expand_rfft_L` first to turn an rfft half-spectrum
(``L >= 0``, as produced by :class:`gpuscatter.Sq3D`) into a full symmetric
volume.

Only the signed-permutation Laue groups are implemented (no trigonal /
hexagonal), which covers every cubic perovskite phase.
"""
from __future__ import annotations
import numpy as np

# ---- generators: signed-permutation matrices in the r.l.u. crystal basis ----
_INV = -np.eye(3, dtype=int)                                    # inversion
_C2a = np.diag([1, -1, -1]).astype(int)                         # 2-fold || a
_C2b = np.diag([-1, 1, -1]).astype(int)                         # 2-fold || b
_C4c = np.array([[0, -1, 0], [1, 0, 0], [0, 0, 1]], dtype=int)  # 4-fold || c
_C3_111 = np.array([[0, 0, 1], [1, 0, 0], [0, 1, 0]], dtype=int)  # 3-fold || [111]

_EXPECTED_ORDERS = {
    '-1': 2, '2/m': 4, 'mmm': 8, '4/m': 8, '4/mmm': 16,
    'm-3': 24, 'm-3m': 48,
}

_LAUE_GENERATORS = {
    '-1':     [_INV],
    '2/m':    [_C2b, _INV],
    'mmm':    [_C2a, _C2b, _INV],
    '4/m':    [_C4c, _INV],
    '4/mmm':  [_C4c, _C2a, _INV],
    'm-3':    [_C3_111, _C2b, _INV],
    'm-3m':   [_C3_111, _C4c, _INV],
}

_GROUPS_CACHE: dict[str, list[np.ndarray]] = {}


def _generate_group(generators: list[np.ndarray],
                    max_iter: int = 200) -> list[np.ndarray]:
    """Close a set of generator matrices into the full point group."""
    ops = {tuple(np.eye(3, dtype=int).flatten())}
    queue = [g.astype(int) for g in generators]
    for _ in range(max_iter):
        new_ops = set()
        for g in queue:
            for o_flat in list(ops):
                o = np.array(o_flat, dtype=int).reshape(3, 3)
                for product in (g @ o, o @ g):
                    key = tuple(product.flatten())
                    if key not in ops:
                        new_ops.add(key)
        if not new_ops:
            break
        ops.update(new_ops)
        queue = [np.array(k, dtype=int).reshape(3, 3) for k in new_ops]
    return [np.array(o, dtype=int).reshape(3, 3) for o in ops]


def get_symmetry_operations(laue_group: str) -> list[np.ndarray]:
    """Return the list of 3x3 integer operation matrices for a Laue group.

    Supported groups: ``-1, 2/m, mmm, 4/m, 4/mmm, m-3, m-3m``.
    """
    if laue_group not in _GROUPS_CACHE:
        if laue_group not in _LAUE_GENERATORS:
            raise ValueError(
                f"Unknown / unsupported Laue group {laue_group!r}. "
                f"Valid: {list(_LAUE_GENERATORS)}"
            )
        ops = _generate_group(_LAUE_GENERATORS[laue_group])
        if len(ops) != _EXPECTED_ORDERS[laue_group]:
            raise RuntimeError(
                f"Generated {len(ops)} ops for {laue_group!r}, "
                f"expected {_EXPECTED_ORDERS[laue_group]}."
            )
        _GROUPS_CACHE[laue_group] = ops
    return _GROUPS_CACHE[laue_group]


def expand_rfft_L(h_arr: np.ndarray, k_arr: np.ndarray, L_arr: np.ndarray,
                  vol: np.ndarray) -> tuple[np.ndarray, np.ndarray,
                                            np.ndarray, np.ndarray]:
    """Expand an rfft half-spectrum (``L >= 0``) to a full symmetric volume.

    Uses Friedel symmetry of an intensity ``S(H, K, -L) = S(-H, -K, L)``
    (true for ``|F|^2`` and for the Bragg-subtracted diffuse, both even
    under ``q -> -q``). The in-plane axes ``h_arr`` and ``k_arr`` must be
    symmetric about 0 so that reversing them negates the coordinate; trim
    the Sq3D result to a symmetric ``|q| <= q_max`` first.

    Parameters
    ----------
    h_arr, k_arr : 1D arrays
        Symmetric, regularly-spaced H and K axes (r.l.u.).
    L_arr : 1D array
        Half-spectrum L axis starting at 0 (``[0, dL, 2 dL, ...]``).
    vol : (nH, nK, nL) array
        Intensity on the half-spectrum grid.

    Returns
    -------
    (h_arr, k_arr, L_full, vol_full)
        ``L_full`` spans ``[-Lmax, ..., 0, ..., Lmax]`` and ``vol_full``
        has shape ``(nH, nK, 2 nL - 1)``.
    """
    h_arr = np.asarray(h_arr)
    k_arr = np.asarray(k_arr)
    L_arr = np.asarray(L_arr)
    if not np.allclose(h_arr, -h_arr[::-1]):
        raise ValueError('h_arr must be symmetric about 0 (trim to |q|<=q_max).')
    if not np.allclose(k_arr, -k_arr[::-1]):
        raise ValueError('k_arr must be symmetric about 0 (trim to |q|<=q_max).')
    if not np.isclose(L_arr[0], 0.0):
        raise ValueError('L_arr must start at 0 (rfft half-spectrum).')

    m = L_arr.size
    L_full = np.concatenate([-L_arr[1:][::-1], L_arr])
    nH, nK = h_arr.size, k_arr.size
    out = np.empty((nH, nK, 2 * m - 1), dtype=vol.dtype)
    out[:, :, m - 1:] = vol                              # L >= 0
    out[:, :, :m - 1] = vol[::-1, ::-1, 1:][:, :, ::-1]  # L < 0 via Friedel
    return h_arr, k_arr, L_full, out


def _gather_op(data: np.ndarray, op: np.ndarray,
               axes, steps, origins, sizes) -> np.ndarray:
    """Gather ``S(R q)`` for one operation ``R`` (a signed permutation).

    Returns an array the same shape as ``data`` with out-of-range orbit
    members set to NaN. Identity axis mapping (a true h,k,l volume).
    """
    # Row a of op picks source axis src_a with sign sgn_a: (R q)_a = sgn_a q_{src_a}.
    idx_arrays = [None, None, None]
    valid_arrays = [None, None, None]
    for a in range(3):
        nz = np.nonzero(op[a])[0]
        src = int(nz[0])
        sgn = int(op[a, src])
        # Source data-index along axis a for the coordinate sgn * axes[src][.]:
        src_idx = np.round((sgn * axes[src] - origins[a]) / steps[a]).astype(np.intp)
        valid = (src_idx >= 0) & (src_idx < sizes[a])
        src_idx = np.clip(src_idx, 0, sizes[a] - 1)
        # This index varies along OUTPUT axis `src`; reshape to broadcast there.
        shape = [1, 1, 1]
        shape[src] = sizes[src]
        idx_arrays[a] = src_idx.reshape(shape)
        valid_arrays[a] = valid.reshape(shape)

    gathered = data[idx_arrays[0], idx_arrays[1], idx_arrays[2]].astype(np.float32)
    valid = valid_arrays[0] & valid_arrays[1] & valid_arrays[2]
    return np.where(valid, gathered, np.float32(np.nan))


def symmetrize_volume(h_arr: np.ndarray, k_arr: np.ndarray, L_arr: np.ndarray,
                      intensity: np.ndarray, laue_group: str,
                      sigma: float | None = None,
                      min_valid: int = 3) -> np.ndarray:
    """Orbit-average ``intensity`` over a Laue group on a regular grid.

    For each voxel the function gathers the full symmetry orbit
    ``{S(R q) : R in group}``, optionally MAD-rejects outliers, then writes
    the (nan)mean of the orbit. Returns a new array; ``intensity`` is not
    modified.

    Parameters
    ----------
    h_arr, k_arr, L_arr : 1D arrays
        Regularly-spaced reciprocal axes (r.l.u.). For groups that flip an
        axis (``4/mmm``, ``m-3m``, ... flip L) the grid must span both signs
        of that axis -- expand an rfft half-spectrum with
        :func:`expand_rfft_L` first.
    intensity : (nH, nK, nL) array
        Volume to average.
    laue_group : str
        One of the groups in :func:`get_symmetry_operations`.
    sigma : float or None, default None
        MAD outlier-rejection multiplier applied per orbit. ``None`` (the
        default for computed S(q), which has no measurement outliers) does a
        pure orbit mean. A finite value flags orbit members deviating by more
        than ``sigma * MAD`` from the orbit median and drops them.
    min_valid : int, default 3
        Minimum finite orbit members required for outlier flagging to apply.

    Returns
    -------
    (nH, nK, nL) float32 array
        The symmetry-averaged volume.
    """
    ops = get_symmetry_operations(laue_group)
    axes = [np.asarray(h_arr, dtype=np.float64),
            np.asarray(k_arr, dtype=np.float64),
            np.asarray(L_arr, dtype=np.float64)]
    steps = [float(a[1] - a[0]) for a in axes]
    origins = [float(a[0]) for a in axes]
    sizes = [int(a.size) for a in axes]
    data = np.asarray(intensity, dtype=np.float32)

    n_ops = len(ops)
    equiv = np.empty((n_ops,) + data.shape, dtype=np.float32)
    for oi, op in enumerate(ops):
        equiv[oi] = _gather_op(data, op, axes, steps, origins, sizes)

    if sigma is not None:
        MAD_SCALE = 1.4826
        with np.errstate(invalid='ignore'):
            n_fin = np.sum(np.isfinite(equiv), axis=0)
            med = np.nanmedian(equiv, axis=0)
            dev = np.abs(equiv - med[None])
            mad = np.nanmedian(dev, axis=0) * MAD_SCALE
            outlier = (dev > sigma * mad[None]) & (n_fin[None] >= min_valid)
        equiv = np.where(outlier, np.float32(np.nan), equiv)

    with np.errstate(invalid='ignore'):
        avg = np.nanmean(equiv, axis=0)
    return np.where(np.isfinite(avg), avg, np.float32(0.0)).astype(np.float32)

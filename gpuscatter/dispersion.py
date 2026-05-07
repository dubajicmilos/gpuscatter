"""Phonon dispersion projection from BZ-folded S(q, omega).

Given a 3D ``S(q, omega)`` cube on the first BZ (output of
:class:`gpuscatter.sqw.Sqw` with :func:`make_qgrid_BZ`), extract
2D dispersion sheets along high-symmetry paths.

Standard cubic BZ paths:

* ``Gamma = (0, 0, 0)``
* ``X = (1/2, 0, 0)``
* ``M = (1/2, 1/2, 0)``
* ``R = (1/2, 1/2, 1/2)``

Path naming follows the Setyawan-Curtarolo convention.

This module also supports custom paths defined as endpoints in r.l.u.
"""
from __future__ import annotations
from dataclasses import dataclass, field
from pathlib import Path
from typing import Sequence
import numpy as np


# Cubic BZ high-symmetry points (in r.l.u., relative to a primitive RLV).
HIGH_SYMMETRY_POINTS_CUBIC = {
    'Gamma': np.array([0.0, 0.0, 0.0]),
    'X':     np.array([0.5, 0.0, 0.0]),
    'M':     np.array([0.5, 0.5, 0.0]),
    'R':     np.array([0.5, 0.5, 0.5]),
}


def _q_red_to_index(q_red, n_cells):
    """Map a reduced q in [-1/2, 1/2) to BZ-grid index in [0, n_cells)."""
    return int(np.round(q_red * n_cells)) % n_cells


def make_path_indices(path_pts: Sequence[np.ndarray], n_cells: int,
                      n_per_seg: int = 11) -> tuple[np.ndarray, np.ndarray, list[int]]:
    """Build piecewise-linear path through reduced q-points.

    Parameters
    ----------
    path_pts
        List of (3,) reduced-q endpoints in [-1/2, 1/2).
    n_cells
        Supercell size per dimension (BZ grid is ``n_cells^3``).
    n_per_seg
        Number of grid points per segment (inclusive of endpoints, except
        joining points are counted once).

    Returns
    -------
    path_q_red
        (n_pts, 3) reduced q for every point on the path.
    path_idx
        (n_pts,) flattened BZ-grid index for each path point.
    seg_breaks
        List of indices where new segments start (for plotting tick marks).
    """
    pts = []
    breaks = [0]
    for i in range(len(path_pts) - 1):
        a = np.asarray(path_pts[i])
        b = np.asarray(path_pts[i + 1])
        for k in range(n_per_seg if i == 0 else n_per_seg):
            t = k / max(n_per_seg - 1, 1)
            q = a + t * (b - a)
            pts.append(q)
        breaks.append(len(pts) - 1)
    path_q_red = np.asarray(pts, dtype=np.float32)
    path_idx = np.asarray(
        [_q_red_to_index(q[0], n_cells) * n_cells * n_cells
         + _q_red_to_index(q[1], n_cells) * n_cells
         + _q_red_to_index(q[2], n_cells)
         for q in path_q_red]
    )
    return path_q_red, path_idx, breaks


def project_dispersion(S_qw: np.ndarray, q_grid_shape: tuple[int, int, int],
                       path_pts: Sequence[np.ndarray],
                       n_per_seg: int = 11
                       ) -> tuple[np.ndarray, np.ndarray, list[int]]:
    """Sample a (n_cells^3, n_omega) cube along a path.

    Parameters
    ----------
    S_qw
        3D-or-flat S(q, omega). Shape ``(n_cells, n_cells, n_cells, n_omega)``
        or ``(n_cells**3, n_omega)``.
    q_grid_shape
        ``(n_cells, n_cells, n_cells)`` shape of the BZ grid.
    path_pts
        Endpoints in r.l.u. (shape (3,) each).
    n_per_seg
        Points per segment.

    Returns
    -------
    S_path
        ``(n_path_pts, n_omega)`` slice along the path.
    path_q_red
        ``(n_path_pts, 3)`` reduced q.
    seg_breaks
        Where each segment starts (for tick marks).
    """
    n_cells = q_grid_shape[0]
    if S_qw.ndim == 4:
        S_flat = S_qw.reshape(n_cells ** 3, S_qw.shape[-1])
    elif S_qw.ndim == 2:
        S_flat = S_qw
    else:
        raise ValueError('S_qw must be (n_cells^3, n_omega) or '
                         '(n_cells, n_cells, n_cells, n_omega)')

    path_q_red, path_idx, breaks = make_path_indices(
        path_pts, n_cells, n_per_seg
    )
    S_path = S_flat[path_idx]
    return S_path, path_q_red, breaks


def cubic_path_GXMR_Gamma(n_cells: int, n_per_seg: int = 11):
    """Default cubic Gamma-X-M-R-Gamma path.

    Returns ``(S_path, path_q_red, breaks, labels, label_pos)``.
    Apply via:

    >>> path_pts = [HIGH_SYMMETRY_POINTS_CUBIC[s]
    ...             for s in ['Gamma', 'X', 'M', 'R', 'Gamma']]
    >>> S_path, q_red, breaks = project_dispersion(S_qw, (n_cells,)*3, path_pts)
    """
    labels = ['Gamma', 'X', 'M', 'R', 'Gamma']
    pts = [HIGH_SYMMETRY_POINTS_CUBIC[s] for s in labels]
    return pts, labels


@dataclass
class DispersionProjection:
    """Convenience wrapper for projecting a BZ-folded S(q, omega)."""

    S_qw: np.ndarray
    q_grid_shape: tuple[int, int, int]
    E_axis_meV: np.ndarray

    def project(self, path_labels: Sequence[str], n_per_seg: int = 11):
        pts = [HIGH_SYMMETRY_POINTS_CUBIC[s] for s in path_labels]
        S_path, q_red, breaks = project_dispersion(
            self.S_qw, self.q_grid_shape, pts, n_per_seg
        )
        return S_path, q_red, breaks

    def project_along(self, path_pts: Sequence[np.ndarray],
                      n_per_seg: int = 11):
        return project_dispersion(self.S_qw, self.q_grid_shape,
                                  path_pts, n_per_seg)

"""Example 5 — Phonon dispersion projection along Gamma-X-M-R-Gamma.

Loads the BZ-folded ``S(q, omega)`` cube produced by Example 4 and
projects it onto the standard cubic high-symmetry path. Plots a
dispersion sheet per partial.

Expected wall time: a few seconds.
"""
from __future__ import annotations
from pathlib import Path
import sys

import numpy as np
import matplotlib.pyplot as plt

_REPO = Path(__file__).resolve().parent.parent
if str(_REPO) not in sys.path:
    sys.path.insert(0, str(_REPO))

from gpuscatter import (                                 # noqa: E402
    DispersionProjection, HIGH_SYMMETRY_POINTS_CUBIC,
)


def main(npz_path: Path,
         labels=('Gamma', 'X', 'M', 'R', 'Gamma'),
         n_per_seg: int = 11,
         e_max_meV: float = 8.0,
         out_path: Path | None = None):
    print(f'Loading {npz_path.name}...')
    d = np.load(npz_path, allow_pickle=True)
    n_cells = int(d['n_cells'])
    E = d['E_axis_meV']

    partial_keys = [k for k in d.files
                    if k.startswith('S_') and k != 'S_total']
    print(f'  partials: {partial_keys}')

    # Build the standard path
    path_pts = [HIGH_SYMMETRY_POINTS_CUBIC[s] for s in labels]

    panels = ['S_total'] + sorted(partial_keys)
    n = len(panels)
    ncols = 4
    nrows = int(np.ceil(n / ncols))
    fig, axes = plt.subplots(nrows, ncols, figsize=(ncols * 3, nrows * 2.5),
                              sharex=True, sharey=True)
    axes = np.atleast_1d(axes).flatten()

    e_mask = E <= e_max_meV
    E_plot = E[e_mask]

    for i, key in enumerate(panels):
        ax = axes[i]
        S_qw = d[key]
        proj = DispersionProjection(S_qw=S_qw,
                                    q_grid_shape=(n_cells,) * 3,
                                    E_axis_meV=E)
        S_path, q_red, breaks = proj.project(labels, n_per_seg=n_per_seg)
        S_path = S_path[:, e_mask]
        # log10 of |S| for visibility
        img = np.log10(np.abs(S_path).T + 1e-3)
        ax.imshow(img, aspect='auto', origin='lower',
                   extent=[0, S_path.shape[0] - 1, E_plot[0], E_plot[-1]],
                   cmap='inferno')
        # Tick labels at segment breaks
        ax.set_xticks(breaks)
        ax.set_xticklabels([s.replace('Gamma', r'$\Gamma$') for s in labels],
                            fontsize=8)
        ax.set_title(key.replace('S_', ''), fontsize=9)
        if i % ncols == 0:
            ax.set_ylabel('E [meV]')
    for j in range(i + 1, len(axes)):
        axes[j].axis('off')

    fig.suptitle(
        f'BZ-folded S(q, $\\omega$) along {"-".join([s.replace("Gamma", "$\\Gamma$") for s in labels])} '
        f'— CsPbI$_3$ 600 K',
        fontsize=11
    )
    fig.tight_layout()
    if out_path is None:
        out_path = npz_path.with_name('dispersion_paths.png')
    fig.savefig(out_path, dpi=150, bbox_inches='tight')
    print(f'Saved {out_path.name}')


if __name__ == '__main__':
    import argparse
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument('--npz', type=Path,
                        default=Path(__file__).parent
                        / 'sqw_BZ_CsPbI3_600K.npz')
    parser.add_argument('--e-max', type=float, default=8.0)
    parser.add_argument('--out', type=Path, default=None)
    args = parser.parse_args()
    main(args.npz, e_max_meV=args.e_max, out_path=args.out)

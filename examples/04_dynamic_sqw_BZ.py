"""Example 4 — Full first-BZ S(q, omega) for phonon-dispersion analysis.

Computes ``S(q, omega)`` on the first Brillouin zone via direct atomic
Fourier sum on a 24 x 24 x 24 q-grid (one BZ point per supercell-RLV).
Output is fed to ``05_dispersion_paths.py`` to extract dispersion sheets
along high-symmetry paths.

Expected wall time on a GTX 1070: ~11 min for the 600 K demo
(13 824 q-points x 5001 frames x 3 species).
"""
from __future__ import annotations
from pathlib import Path
import sys

import numpy as np

_REPO = Path(__file__).resolve().parent.parent
if str(_REPO) not in sys.path:
    sys.path.insert(0, str(_REPO))

from gpuscatter import (                             # noqa: E402
    NpzTrajectory, Sqw, SqwConfig, make_qgrid_BZ
)


DEFAULT_TRAJ_DIR = Path(
    r'C:/Claude projects/MD trajectories/trajectories_CsPbI3_paper/600K'
)


def main(traj_dir: Path = DEFAULT_TRAJ_DIR,
         n_cells: int = 24,
         dt_fs: float = 200.0,
         out_path: Path | None = None):
    files = sorted(traj_dir.glob('nptraj*.npz'))
    if not files:
        raise FileNotFoundError(f'No nptraj*.npz files in {traj_dir}')
    traj = NpzTrajectory(files)
    a_cub = traj.L_box / n_cells
    print(f'Loaded {len(files)} files, {traj.n_frames} frames, '
          f'a_cub = {a_cub:.4f} A, n_cells = {n_cells}')

    q_red_1d, q_red, q_cart, grid_shape = make_qgrid_BZ(n_cells, a_cub)
    print(f'q-grid: full BZ {n_cells}^3 = {q_cart.shape[0]} q-points')

    cfg = SqwConfig(q_vecs=q_cart, dt_fs=dt_fs, weighting='xray',
                    sign_convention='minus')
    result = Sqw(traj, cfg).run()
    print(f'\nDone in {result.elapsed_s:.1f}s')

    n_omega = result.E_axis_meV.size
    for key in list(result.partials):
        result.partials[key] = result.partials[key].reshape(
            n_cells, n_cells, n_cells, n_omega
        )
    if result.total is not None:
        result.total = result.total.reshape(
            n_cells, n_cells, n_cells, n_omega
        )

    if out_path is None:
        out_path = Path(__file__).parent / 'sqw_BZ_CsPbI3_600K.npz'

    out = {
        'q_red_1d': q_red_1d, 'q_red': q_red, 'q_cart': q_cart,
        'E_axis_meV': result.E_axis_meV,
        'n_frames': result.n_frames, 'dt_fs': result.dt_fs,
        'a_cub': a_cub, 'n_cells': n_cells,
        'method': result.method,
        'S_total': result.total,
    }
    for (a, b), S in result.partials.items():
        out[f'S_{a}{b}'] = S
    np.savez(out_path, **out)
    size_mb = out_path.stat().st_size / 1e6
    print(f'Saved {out_path.name} ({size_mb:.0f} MB)')


if __name__ == '__main__':
    import argparse
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument('--traj-dir', type=Path, default=DEFAULT_TRAJ_DIR)
    parser.add_argument('--n-cells', type=int, default=24)
    parser.add_argument('--dt-fs', type=float, default=200.0)
    parser.add_argument('--out', type=Path, default=None)
    args = parser.parse_args()
    main(traj_dir=args.traj_dir, n_cells=args.n_cells,
         dt_fs=args.dt_fs, out_path=args.out)

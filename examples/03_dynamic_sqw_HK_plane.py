"""Example 3 — Compute S(q, omega) on an HK plane at fixed L.

For CsPbI3 600 K HK1.5 (L = 1.5 r.l.u.). Output is a 161 x 161 x 2501
cube of the X-ray-weighted S(q, omega) plus the 6 partials.

Expected wall time on a GTX 1070: ~21 min for 5001 frames.
On a single-CPU dynasor run, the same calculation takes about 5 hours.
"""
from __future__ import annotations
from pathlib import Path
import sys

_REPO = Path(__file__).resolve().parent.parent
if str(_REPO) not in sys.path:
    sys.path.insert(0, str(_REPO))

from gpuscatter import (                             # noqa: E402
    NpzTrajectory, Sqw, SqwConfig, make_qgrid_HK_plane
)


DEFAULT_TRAJ_DIR = Path(
    r'C:/Claude projects/MD trajectories/trajectories_CsPbI3_paper/600K'
)


def main(traj_dir: Path = DEFAULT_TRAJ_DIR,
         L_value: float = 1.5,
         n_grid: int = 161,
         h_max: float = 4.0,
         dt_fs: float = 200.0,
         out_path: Path | None = None):
    files = sorted(traj_dir.glob('nptraj*.npz'))
    if not files:
        raise FileNotFoundError(f'No nptraj*.npz files in {traj_dir}')
    traj = NpzTrajectory(files)
    a_cub = traj.L_box / 24
    print(f'Loaded {len(files)} files, {traj.n_frames} frames, '
          f'a_cub = {a_cub:.4f} A')

    h_arr, q_vecs, grid_shape = make_qgrid_HK_plane(
        L_value=L_value, a_cub=a_cub, h_max=h_max, n_grid=n_grid
    )
    print(f'q-grid: HK{L_value} plane, {q_vecs.shape[0]} q-points')

    cfg = SqwConfig(q_vecs=q_vecs, dt_fs=dt_fs, weighting='xray')
    result = Sqw(traj, cfg).run()
    print(f'\nDone in {result.elapsed_s:.1f}s')

    # Reshape partials to (n_grid, n_grid, n_omega)
    n_omega = result.E_axis_meV.size
    for key in list(result.partials):
        result.partials[key] = result.partials[key].reshape(
            n_grid, n_grid, n_omega
        )
    if result.total is not None:
        result.total = result.total.reshape(n_grid, n_grid, n_omega)

    if out_path is None:
        out_path = (Path(__file__).parent
                    / f'sqw_CsPbI3_600K_HK{L_value}.npz')
    # Custom save with reshape metadata.
    import numpy as np
    out = {
        'h_arr': h_arr, 'k_arr': h_arr.copy(),
        'L_value': L_value,
        'E_axis_meV': result.E_axis_meV,
        'n_frames': result.n_frames,
        'dt_fs': result.dt_fs,
        'a_cub': a_cub,
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
    parser.add_argument('--L', type=float, default=1.5)
    parser.add_argument('--n-grid', type=int, default=161)
    parser.add_argument('--h-max', type=float, default=4.0)
    parser.add_argument('--dt-fs', type=float, default=200.0)
    parser.add_argument('--out', type=Path, default=None)
    args = parser.parse_args()
    main(traj_dir=args.traj_dir, L_value=args.L, n_grid=args.n_grid,
         h_max=args.h_max, dt_fs=args.dt_fs, out_path=args.out)

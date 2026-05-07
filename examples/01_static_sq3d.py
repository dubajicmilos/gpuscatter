"""Example 1 — Compute the full 3D static S(q) cube for CsPbI3 600 K.

This is the headline-feature demo. It computes the X-ray-weighted
partial S(q) on a 192 x 192 x 97 q-grid (covering all 97 unique L
planes simultaneously) using GPU density binning + 3D rFFT.

Expected wall time on a GTX 1070:
* 1 region (full box): ~1.7 min for 5001 frames
* 8 sub-regions (ripple suppression): ~14 min

Output: ``sq3d_CsPbI3_600K_8reg.npz`` with 6 X-ray partials + total.
"""
from __future__ import annotations
from pathlib import Path
import sys

# When run from the repo root without installing the package, fall back
# to in-place import.
_REPO = Path(__file__).resolve().parent.parent
if str(_REPO) not in sys.path:
    sys.path.insert(0, str(_REPO))

from gpuscatter import NpzTrajectory, Sq3D, Sq3DConfig    # noqa: E402


# Edit this path to point at the trajectory directory on your machine.
DEFAULT_TRAJ_DIR = Path(
    r'C:/Claude projects/MD trajectories/trajectories_CsPbI3_paper/600K'
)


def main(traj_dir: Path = DEFAULT_TRAJ_DIR,
         n_regions: int = 8, sub_region_cells: int = 8,
         out_path: Path | None = None):
    files = sorted(traj_dir.glob('nptraj*.npz'))
    if not files:
        raise FileNotFoundError(f'No nptraj*.npz files in {traj_dir}')
    traj = NpzTrajectory(files)
    print(f'Loaded {len(files)} trajectory files, '
          f'{traj.n_frames} frames, {traj.n_atoms} atoms')

    cfg = Sq3DConfig(
        n_cells=24,
        n_voxels_per_cell=8,
        sub_regions=n_regions,
        sub_region_cells=sub_region_cells if n_regions > 1 else None,
        weighting='xray',
    )

    result = Sq3D(traj, cfg).run()
    print(f'\nDone in {result.elapsed_s:.1f}s')
    print(f'Output partials: {[f"{a}{b}" for a, b in result.partials]}')

    if out_path is None:
        suffix = f'_{n_regions}reg' if n_regions > 1 else '_full_box'
        out_path = Path(__file__).parent / f'sq3d_CsPbI3_600K{suffix}.npz'
    result.save(out_path)
    size_mb = out_path.stat().st_size / 1e6
    print(f'Saved {out_path.name} ({size_mb:.1f} MB)')


if __name__ == '__main__':
    import argparse
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument('--traj-dir', type=Path, default=DEFAULT_TRAJ_DIR)
    parser.add_argument('--n-regions', type=int, default=8,
                        help='Sub-regions for ripple suppression. '
                             '1 = full box only.')
    parser.add_argument('--sub-region-cells', type=int, default=8)
    parser.add_argument('--out', type=Path, default=None)
    args = parser.parse_args()
    main(traj_dir=args.traj_dir, n_regions=args.n_regions,
         sub_region_cells=args.sub_region_cells, out_path=args.out)

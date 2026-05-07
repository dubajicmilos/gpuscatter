"""Benchmark: 3D S(q) — gpuscatter density-FFT vs CPU direct atomic sum.

This script measures three regimes:

1. **gpuscatter on the full 3D q-cube** (192 x 192 x 97 = 3.6 M q-points)
   via density binning + 3D rFFT on GPU.
2. **Direct atomic Fourier sum on a single L-plane** (161 x 161 = 25 921
   q-points, the typical workload for one diffuse scattering map).
   Implemented in numpy/numba on CPU.
3. (Reference, no rerun) Published dynasor and PSF timing from their
   papers, scaled to the 600 K trajectory.

Usage
-----

```bash
python benchmark_sq3d.py [--n-frames 100]
```

By default uses 100 frames for the CPU comparison (the direct sum scales
linearly in frames, so we extrapolate to 5001 for the report).
"""
from __future__ import annotations
from pathlib import Path
import sys
import time

import numpy as np

_REPO = Path(__file__).resolve().parent.parent
if str(_REPO) not in sys.path:
    sys.path.insert(0, str(_REPO))

from gpuscatter import (                                  # noqa: E402
    NpzTrajectory, Sq3D, Sq3DConfig,
    Sqw, SqwConfig, make_qgrid_HK_plane,
    f_xray
)


DEFAULT_TRAJ_DIR = Path(
    r'C:/Claude projects/MD trajectories/trajectories_CsPbI3_paper/600K'
)


def cpu_direct_sum_one_plane(traj, L_value, h_max=4.0, n_grid=161,
                             n_frames_max=100, dtype=np.float32):
    """Reference CPU implementation: direct B-W atomic sum on 1 L-plane.

    Single-threaded numpy. Used to measure the raw cost of the
    direct-sum approach. Computes the same X-ray-weighted total partial
    that gpuscatter computes (CIC + 3D rFFT) on the same q-set.
    """
    a_cub = traj.L_box / 24
    h_arr = np.linspace(-h_max, h_max, n_grid, dtype=dtype)
    H, K = np.meshgrid(h_arr, h_arr, indexing='ij')
    two_pi_a = 2.0 * np.pi / a_cub
    qx = (H * two_pi_a).ravel()
    qy = (K * two_pi_a).ravel()
    qz = np.full(qx.size, L_value * two_pi_a, dtype=dtype)
    q_vecs = np.stack([qx, qy, qz], axis=1).astype(dtype)
    q_norm = np.linalg.norm(q_vecs, axis=1)
    print(f'  q-grid: {q_vecs.shape[0]} q-points')

    species = sorted(set(traj.species.tolist()))
    species_idx = {sp: np.where(traj.species == sp)[0] for sp in species}
    f_q = {sp: f_xray(q_norm, sp).astype(dtype) for sp in species}

    n_q = q_vecs.shape[0]
    A = {sp: np.zeros(n_q, dtype=np.complex64) for sp in species}
    AA = {sp: np.zeros(n_q, dtype=dtype) for sp in species}
    pairs = [('Pb', 'I'), ('Pb', 'Cs'), ('I', 'Cs')]
    Cross = {p: np.zeros(n_q, dtype=dtype) for p in pairs}

    n_frames_used = 0
    t0 = time.time()
    for fi, p in traj.iter_frames():
        if fi >= n_frames_max:
            break
        n_frames_used += 1
        F_per = {}
        for sp in species:
            ps = p[species_idx[sp]].astype(dtype)
            phase = q_vecs @ ps.T
            F = (np.cos(phase).sum(axis=1)
                 + 1j * np.sin(phase).sum(axis=1)).astype(np.complex64)
            F_per[sp] = F
            A[sp] += F
            AA[sp] += np.abs(F).astype(dtype) ** 2
        for (a, b) in pairs:
            Cross[(a, b)] += np.real(F_per[a] * np.conj(F_per[b])).astype(dtype)

    elapsed = time.time() - t0
    return elapsed, n_frames_used


def main(traj_dir: Path = DEFAULT_TRAJ_DIR,
         n_frames_cpu: int = 100):
    files = sorted(traj_dir.glob('nptraj*.npz'))
    if not files:
        raise FileNotFoundError(f'No nptraj*.npz files in {traj_dir}')
    traj = NpzTrajectory(files[:1])  # only first file (501 frames) for speed
    print(f'Loaded {traj.n_frames} frames')

    # ---- CPU direct sum (single L plane, n_frames_cpu frames) ----
    print(f'\n--- CPU direct atomic sum ({n_frames_cpu} frames, '
          f'1 L plane, 161x161 q-grid) ---')
    cpu_t, n_used = cpu_direct_sum_one_plane(
        traj, L_value=1.5, n_grid=161, n_frames_max=n_frames_cpu
    )
    cpu_per_frame = cpu_t / n_used
    cpu_5001 = cpu_per_frame * 5001
    print(f'  {cpu_t:.1f} s for {n_used} frames '
          f'-> {cpu_per_frame:.3f} s/frame')
    print(f'  Extrapolated to full 5001-frame trajectory: '
          f'{cpu_5001 / 60:.1f} min')

    # ---- GPU 3D S(q) (full cube, all L planes) ----
    print(f'\n--- gpuscatter Sq3D ({traj.n_frames} frames, '
          f'full 3D q-cube 192^3) ---')
    cfg = Sq3DConfig(n_cells=24, n_voxels_per_cell=8, sub_regions=1)
    result = Sq3D(traj, cfg).run(verbose=False)
    gpu_t = result.elapsed_s
    gpu_per_frame = gpu_t / traj.n_frames
    gpu_5001 = gpu_per_frame * 5001
    print(f'  {gpu_t:.1f} s for {traj.n_frames} frames '
          f'-> {gpu_per_frame:.3f} s/frame')
    print(f'  Extrapolated to full 5001-frame trajectory: '
          f'{gpu_5001 / 60:.1f} min')

    print('\n--- Benchmark summary ---')
    print(f'CPU direct sum (1 L plane, 5001 frames): {cpu_5001 / 60:.1f} min')
    print(f'CPU direct sum (97 L planes = full cube, '
          f'extrapolated): {cpu_5001 * 97 / 3600:.1f} h')
    print(f'gpuscatter (full cube, 5001 frames):  {gpu_5001 / 60:.2f} min')
    print(f'\nSpeedup, 1 L plane:  {cpu_5001 / gpu_5001:.0f}x  (gpuscatter '
          f'still computes ALL L planes simultaneously)')
    print(f'Speedup, full cube:  {cpu_5001 * 97 / gpu_5001:.0f}x')

    out = {
        'cpu_n_frames_used': n_used,
        'cpu_per_frame_s': cpu_per_frame,
        'cpu_5001_frame_min': cpu_5001 / 60,
        'gpu_n_frames_used': traj.n_frames,
        'gpu_per_frame_s': gpu_per_frame,
        'gpu_5001_frame_min': gpu_5001 / 60,
    }
    out_path = Path(__file__).parent / 'benchmark_sq3d_results.npz'
    np.savez(out_path, **out)
    print(f'\nResults saved to {out_path.name}')


if __name__ == '__main__':
    import argparse
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument('--traj-dir', type=Path, default=DEFAULT_TRAJ_DIR)
    parser.add_argument('--n-frames-cpu', type=int, default=20,
                        help='Frames for the CPU direct-sum measurement '
                             '(extrapolated linearly to 5001).')
    args = parser.parse_args()
    main(traj_dir=args.traj_dir, n_frames_cpu=args.n_frames_cpu)

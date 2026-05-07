"""Benchmark: dynamic S(q, omega) — gpuscatter vs CPU direct atomic sum.

This script measures the per-frame, per-q-point throughput of the
spatial Fourier amplitude computation, which is the dominant cost of
the dynamic S(q, omega) calculation. Time-FFT and partial-assembly
are < 5% of total cost on both CPU and GPU.

The CPU reference is a single-threaded numpy implementation (no numba)
to give a clean cache-friendly baseline. Real-world dynasor / PSF
deployments use numba JIT or multi-threading, which speeds the CPU
case up by typically 4-10x but does not change the order of magnitude.
"""
from __future__ import annotations
from pathlib import Path
import sys
import time

import numpy as np

_REPO = Path(__file__).resolve().parent.parent
if str(_REPO) not in sys.path:
    sys.path.insert(0, str(_REPO))

from gpuscatter import (                                   # noqa: E402
    NpzTrajectory, Sqw, SqwConfig, make_qgrid_HK_plane
)


DEFAULT_TRAJ_DIR = Path(
    r'C:/Claude projects/MD trajectories/trajectories_CsPbI3_paper/600K'
)


def cpu_amplitude_one_frame(positions, q_vecs, species_idx, sign='minus'):
    """Single-frame numpy direct atomic Fourier sum, all species."""
    s = -1.0 if sign == 'minus' else +1.0
    F = {}
    for sp, idx in species_idx.items():
        ps = positions[idx]
        phase = q_vecs @ ps.T          # (n_q, n_atoms)
        F[sp] = (np.cos(phase).sum(axis=1)
                 + 1j * s * np.sin(phase).sum(axis=1)).astype(np.complex64)
    return F


def main(traj_dir: Path = DEFAULT_TRAJ_DIR,
         n_grid_cpu: int = 41,
         n_frames_cpu: int = 5):
    files = sorted(traj_dir.glob('nptraj*.npz'))
    if not files:
        raise FileNotFoundError(f'No nptraj*.npz files in {traj_dir}')
    traj = NpzTrajectory(files[:1])
    a_cub = traj.L_box / 24
    species_idx = {sp: np.where(traj.species == sp)[0]
                   for sp in sorted(set(traj.species))}
    print(f'Loaded {traj.n_frames} frames, {traj.n_atoms} atoms')

    # ---- CPU direct sum ----
    h_cpu, q_cpu, _ = make_qgrid_HK_plane(L_value=1.5, a_cub=a_cub,
                                           n_grid=n_grid_cpu)
    n_q_cpu = q_cpu.shape[0]
    print(f'\n--- CPU direct atomic sum ({n_frames_cpu} frames, '
          f'{n_q_cpu} q-points, 3 species) ---')

    cpu_t = 0.0
    for fi, p in traj.iter_frames():
        if fi >= n_frames_cpu:
            break
        t0 = time.time()
        cpu_amplitude_one_frame(p.astype(np.float32), q_cpu,
                                species_idx, sign='minus')
        cpu_t += time.time() - t0

    cpu_per_frame = cpu_t / n_frames_cpu
    cpu_per_q_per_frame = cpu_per_frame / n_q_cpu
    print(f'  {cpu_t:.2f} s for {n_frames_cpu} frames -> '
          f'{cpu_per_frame:.3f} s/frame')

    # Extrapolate to full HK-plane workload
    n_q_full = 161 * 161
    cpu_5001_full = cpu_per_q_per_frame * n_q_full * 5001
    print(f'  Extrapolated to 161x161 HK-plane x 5001 frames: '
          f'{cpu_5001_full / 60:.1f} min')

    # ---- GPU ----
    h_gpu, q_gpu, _ = make_qgrid_HK_plane(L_value=1.5, a_cub=a_cub,
                                           n_grid=161)
    print(f'\n--- gpuscatter Sqw ({traj.n_frames} frames, '
          f'{q_gpu.shape[0]} q-points) ---')
    cfg = SqwConfig(q_vecs=q_gpu, dt_fs=200.0, weighting='xray')
    result = Sqw(traj, cfg).run(verbose=False)
    gpu_t = result.elapsed_s
    gpu_per_frame = gpu_t / traj.n_frames
    gpu_5001 = gpu_per_frame * 5001
    print(f'  {gpu_t:.1f} s for {traj.n_frames} frames -> '
          f'{gpu_per_frame:.3f} s/frame (full 161x161 grid)')
    print(f'  Extrapolated to 5001-frame trajectory: '
          f'{gpu_5001 / 60:.1f} min')

    print('\n--- Benchmark summary ---')
    speedup = cpu_5001_full / gpu_5001
    print(f'CPU direct sum (HK plane, 5001 frames): {cpu_5001_full / 60:.1f} min')
    print(f'gpuscatter (HK plane, 5001 frames):     {gpu_5001 / 60:.1f} min')
    print(f'Speedup: {speedup:.0f}x  (numpy single-thread baseline)')

    out = {
        'cpu_n_q': n_q_cpu, 'cpu_n_frames': n_frames_cpu,
        'cpu_per_frame_s': cpu_per_frame,
        'cpu_per_q_per_frame_s': cpu_per_q_per_frame,
        'cpu_extrapolated_min': cpu_5001_full / 60,
        'gpu_n_q': q_gpu.shape[0], 'gpu_n_frames': traj.n_frames,
        'gpu_per_frame_s': gpu_per_frame,
        'gpu_extrapolated_min': gpu_5001 / 60,
        'speedup': speedup,
    }
    out_path = Path(__file__).parent / 'benchmark_sqw_results.npz'
    np.savez(out_path, **out)
    print(f'\nResults saved to {out_path.name}')


if __name__ == '__main__':
    import argparse
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument('--traj-dir', type=Path, default=DEFAULT_TRAJ_DIR)
    parser.add_argument('--n-grid-cpu', type=int, default=41,
                        help='HK-grid size for the CPU baseline (small to '
                             'keep test time reasonable). Extrapolates to '
                             '161x161.')
    parser.add_argument('--n-frames-cpu', type=int, default=5)
    args = parser.parse_args()
    main(traj_dir=args.traj_dir, n_grid_cpu=args.n_grid_cpu,
         n_frames_cpu=args.n_frames_cpu)

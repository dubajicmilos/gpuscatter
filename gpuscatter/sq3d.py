"""3D static partial S(q) on the supercell-reciprocal-lattice grid.

Algorithm
---------
For each frame:

1. Bin atomic positions per species onto a 3D real-space grid using
   cloud-in-cell (CIC) interpolation.
2. 3D rFFT each species' density grid to get F_a(q).
3. Accumulate <F_a>, <|F_a|^2>, <Re[F_a F_b*]> over all frames.

After all frames:

* Auto:  S_aa(q) = <|F_a|^2> - |<F_a>|^2     (exact Bragg subtraction)
* Cross: S_ab(q) = <Re[F_a F_b*]> - Re[<F_a><F_b>*]

Then deconvolve the CIC kernel and apply X-ray Cromer-Mann form factors.

The output q-grid covers ``H, K`` on ``[-N/2, N/2)`` and ``L`` on
``[0, N/2]`` (rfft half-spectrum) at step ``1 / N_cells`` r.l.u.

Optionally, sub-region averaging suppresses long-vector finite-N Fourier
ripples by computing per-frame densities on a small subcube of the
supercell, FFTing each region separately, and averaging.

This module computes in 1.7 min on a GTX 1070 vs ~34 h via direct sum
for the 600 K CsPbI3 demo (5001 frames, full 3D q-cube).
"""
from __future__ import annotations
from dataclasses import dataclass, field
from pathlib import Path
import time
from typing import Sequence
import numpy as np

from .form_factors import f_xray
from .trajectory import BaseTrajectory


@dataclass
class Sq3DConfig:
    """Configuration for :class:`Sq3D`."""
    n_cells: int                                    # supercell size (per dim)
    n_voxels_per_cell: int = 8                      # binning resolution
    species: Sequence[str] | None = None            # default: discover from traj
    cross_pairs: Sequence[tuple[str, str]] | None = None  # default: all pairs
    sub_regions: int = 1                            # 1 = full box (no averaging)
    sub_region_cells: int | None = None             # cells per sub-region side
    rng_seed: int = 7
    weighting: str = 'xray'                         # 'xray' or 'unit'

    @property
    def n_total(self) -> int:
        return self.n_cells * self.n_voxels_per_cell


def _cic_bin_gpu(positions, N, dx):
    """Cloud-in-cell binning of (n_atoms, 3) positions onto (N, N, N).

    Positions in angstroms; ``dx = L_box / N``. Returns float32 grid.
    """
    import cupy as cp
    fi = positions / dx
    base = cp.floor(fi)
    i0 = base.astype(cp.int32) % N
    df = fi - base
    n = positions.shape[0]
    all_w = cp.empty(8 * n, dtype=cp.float32)
    all_idx = cp.empty(8 * n, dtype=cp.int64)
    pos = 0
    for ox in (0, 1):
        wx = df[:, 0] if ox else (1.0 - df[:, 0])
        for oy in (0, 1):
            wy = df[:, 1] if oy else (1.0 - df[:, 1])
            for oz in (0, 1):
                wz = df[:, 2] if oz else (1.0 - df[:, 2])
                w = wx * wy * wz
                ix = (i0[:, 0] + ox) % N
                iy = (i0[:, 1] + oy) % N
                iz = (i0[:, 2] + oz) % N
                idx = (ix.astype(cp.int64) * (N * N)
                       + iy.astype(cp.int64) * N
                       + iz.astype(cp.int64))
                all_w[pos:pos + n] = w
                all_idx[pos:pos + n] = idx
                pos += n
    grid_flat = cp.bincount(all_idx, weights=all_w, minlength=N**3)
    return grid_flat.reshape(N, N, N).astype(cp.float32)


@dataclass
class Sq3DResult:
    """Container for 3D static S(q) output."""
    h_arr: np.ndarray
    k_arr: np.ndarray
    L_arr: np.ndarray
    a_cub: float
    L_box: float
    n_frames: int
    n_cells: int
    n_voxels_per_cell: int
    n_regions: int
    method: str
    partials: dict[tuple[str, str], np.ndarray] = field(default_factory=dict)
    total: np.ndarray | None = None
    elapsed_s: float = 0.0
    region_origins: np.ndarray | None = None

    def save(self, path: str | Path):
        """Save to NPZ. Partials become ``S_{a}{b}`` keys."""
        out = {
            'h_arr': self.h_arr, 'k_arr': self.k_arr, 'L_arr': self.L_arr,
            'a_cub': self.a_cub, 'L_box': self.L_box,
            'n_frames': self.n_frames,
            'n_cells': self.n_cells,
            'n_voxels_per_cell': self.n_voxels_per_cell,
            'n_regions': self.n_regions,
            'method': self.method,
        }
        if self.region_origins is not None:
            out['region_origins'] = self.region_origins
        if self.total is not None:
            out['S_total'] = self.total
        for (a, b), S in self.partials.items():
            out[f'S_{a}{b}'] = S
        np.savez(Path(path), **out)


class Sq3D:
    """3D static S(q) computation by density binning + GPU rFFT.

    Parameters
    ----------
    trajectory
        Any :class:`gpuscatter.trajectory.BaseTrajectory`.
    config
        :class:`Sq3DConfig` controlling the q-grid, species, sub-region
        averaging, and weighting.

    Examples
    --------
    >>> traj = NpzTrajectory(Path('600K').glob('nptraj*.npz'))
    >>> cfg = Sq3DConfig(n_cells=24, n_voxels_per_cell=8, sub_regions=8,
    ...                  sub_region_cells=8)
    >>> result = Sq3D(traj, cfg).run()
    >>> result.save('sq3d_600K.npz')
    """

    def __init__(self, trajectory: BaseTrajectory, config: Sq3DConfig):
        self.traj = trajectory
        self.cfg = config

        # Validate species
        unique_species = sorted(set(trajectory.species.tolist()))
        if config.species is None:
            self.species = unique_species
        else:
            for sp in config.species:
                if sp not in unique_species:
                    raise ValueError(
                        f'Requested species {sp!r} not in trajectory '
                        f'(found {unique_species}).'
                    )
            self.species = list(config.species)

        # Default cross-pairs: all (a, b) with a < b
        if config.cross_pairs is None:
            sp = self.species
            self.cross_pairs = [(sp[i], sp[j])
                                for i in range(len(sp))
                                for j in range(i + 1, len(sp))]
        else:
            self.cross_pairs = list(config.cross_pairs)

        # Sub-region setup
        if config.sub_regions > 1:
            if config.sub_region_cells is None:
                raise ValueError(
                    'sub_region_cells must be set when sub_regions > 1.'
                )
            if config.sub_region_cells > config.n_cells:
                raise ValueError(
                    f'sub_region_cells ({config.sub_region_cells}) must be '
                    f'<= n_cells ({config.n_cells}).'
                )

    def _build_region_masks(self):
        """Compute integer cell indices for atoms in frame 0 and pick origins."""
        if self.cfg.sub_regions <= 1:
            return None, None

        a_cub = self.traj.L_box / self.cfg.n_cells
        cell_idx = (np.floor(self.traj.ref_positions / a_cub)
                    .astype(np.int32) % self.cfg.n_cells)
        rng = np.random.default_rng(self.cfg.rng_seed)
        origins, masks = [], []
        N_sub = self.cfg.sub_region_cells
        for _ in range(self.cfg.sub_regions):
            origin = rng.integers(0, self.cfg.n_cells - N_sub + 1, size=3)
            mask = np.all(
                (cell_idx >= origin) & (cell_idx < origin + N_sub), axis=1
            )
            origins.append(tuple(int(x) for x in origin))
            masks.append(mask)
        return np.asarray(origins), masks

    def run(self, verbose: bool = True) -> Sq3DResult:
        """Run the full computation. Returns an :class:`Sq3DResult`."""
        try:
            import cupy as cp
        except ImportError as e:
            raise ImportError(
                'CuPy is required for GPU computation. '
                'Install via `pip install cupy-cuda12x` (or matching CUDA).'
            ) from e

        cfg = self.cfg
        traj = self.traj
        a_cub = traj.L_box / cfg.n_cells
        N = cfg.n_total
        dx = traj.L_box / N
        N_half = N // 2 + 1
        shape_q = (N, N, N_half)

        if verbose:
            dev = cp.cuda.runtime.getDeviceProperties(0)['name']
            if isinstance(dev, bytes):
                dev = dev.decode()
            print(f'[gpuscatter.Sq3D] device: {dev}')
            print(f'[gpuscatter.Sq3D] L_box = {traj.L_box:.3f} A, '
                  f'a_cub = {a_cub:.3f} A')
            print(f'[gpuscatter.Sq3D] grid {N}^3, dq = 1/{cfg.n_cells} r.l.u., '
                  f'q_max = {N // 2 / cfg.n_cells:.2f} r.l.u.')
            print(f'[gpuscatter.Sq3D] species = {self.species}')
            print(f'[gpuscatter.Sq3D] sub-regions = {cfg.sub_regions}')

        # ---- region masks (or single full-box) ----
        region_origins, region_masks = self._build_region_masks()

        if cfg.sub_regions <= 1:
            spec_indices_per_region = [
                {sp: cp.asarray(np.where(traj.species == sp)[0])
                 for sp in self.species}
            ]
        else:
            spec_indices_per_region = []
            for r_mask in region_masks:
                d = {}
                for sp in self.species:
                    sub_atoms = np.where(r_mask & (traj.species == sp))[0]
                    d[sp] = cp.asarray(sub_atoms)
                spec_indices_per_region.append(d)

        n_R = max(cfg.sub_regions, 1)

        # ---- accumulators (per region per species) ----
        F_avg = {(r, sp): cp.zeros(shape_q, dtype=cp.complex64)
                 for r in range(n_R) for sp in self.species}
        FF_avg = {(r, sp): cp.zeros(shape_q, dtype=cp.float32)
                  for r in range(n_R) for sp in self.species}
        Cross_avg = {(r, key): cp.zeros(shape_q, dtype=cp.float32)
                     for r in range(n_R) for key in self.cross_pairs}

        # ---- frame loop ----
        F_per = {sp: None for sp in self.species}
        t0 = time.time()
        global_idx = 0
        NF = traj.n_frames

        for fi, p_frame in traj.iter_frames():
            p_frame = (p_frame.astype(np.float32) % traj.L_box)
            p_gpu_full = cp.asarray(p_frame)

            for r in range(n_R):
                spec_idx = spec_indices_per_region[r]
                for sp in self.species:
                    p_sp = p_gpu_full[spec_idx[sp]]
                    rho = _cic_bin_gpu(p_sp, N, dx)
                    F = cp.fft.rfftn(rho)
                    del rho
                    FF_avg[(r, sp)] += cp.abs(F).astype(cp.float32) ** 2
                    F_avg[(r, sp)] += F
                    F_per[sp] = F
                for (a, b) in self.cross_pairs:
                    Cross_avg[(r, (a, b))] += cp.real(
                        F_per[a] * cp.conj(F_per[b])
                    ).astype(cp.float32)
                for sp in self.species:
                    F_per[sp] = None
            del p_gpu_full

            global_idx += 1
            if verbose and (global_idx % 200 == 0 or global_idx == NF):
                cp.cuda.runtime.deviceSynchronize()
                elapsed = time.time() - t0
                rate = global_idx / max(elapsed, 1e-9)
                eta = (NF - global_idx) / max(rate, 1e-9)
                print(f'[gpuscatter.Sq3D]   frame {global_idx}/{NF}, '
                      f'{elapsed:.0f}s, eta {eta:.0f}s', flush=True)

        loop_t = time.time() - t0

        # ---- normalize ----
        for k in F_avg:
            F_avg[k] /= NF
            FF_avg[k] /= NF
        for k in Cross_avg:
            Cross_avg[k] /= NF

        # ---- region-averaged diffuse partials ----
        S_diffuse = {}
        for sp in self.species:
            S = cp.zeros(shape_q, dtype=cp.float32)
            for r in range(n_R):
                S += FF_avg[(r, sp)] - cp.abs(F_avg[(r, sp)]) ** 2
            S_diffuse[(sp, sp)] = S / n_R
        for (a, b) in self.cross_pairs:
            S = cp.zeros(shape_q, dtype=cp.float32)
            for r in range(n_R):
                S += (Cross_avg[(r, (a, b))]
                      - cp.real(F_avg[(r, a)] * cp.conj(F_avg[(r, b)])))
            S_diffuse[(a, b)] = S / n_R

        del F_avg, FF_avg, Cross_avg

        # ---- CIC kernel deconvolution ----
        kx = cp.fft.fftfreq(N, d=dx) * 2 * cp.pi
        ky = kx.copy()
        kz = cp.fft.rfftfreq(N, d=dx) * 2 * cp.pi
        KX, KY, KZ = cp.meshgrid(kx, ky, kz, indexing='ij')
        sx = cp.sinc(KX * dx / (2 * np.pi))
        sy = cp.sinc(KY * dx / (2 * np.pi))
        sz = cp.sinc(KZ * dx / (2 * np.pi))
        W2 = (sx ** 4 * sy ** 4 * sz ** 4).astype(cp.float32)
        W2 = cp.maximum(W2, 1e-6)
        del sx, sy, sz

        # ---- form factors ----
        if cfg.weighting == 'xray':
            q_mag = cp.sqrt(KX ** 2 + KY ** 2 + KZ ** 2)
            q_mag_cpu = cp.asnumpy(q_mag).astype(np.float64)
            f_per = {
                sp: cp.asarray(
                    f_xray(q_mag_cpu.ravel(), sp).reshape(shape_q).astype(np.float32)
                )
                for sp in self.species
            }
            del q_mag, q_mag_cpu
        elif cfg.weighting == 'unit':
            f_per = {sp: cp.asarray(
                np.ones(shape_q, dtype=np.float32))
                     for sp in self.species}
        else:
            raise ValueError(f'Unknown weighting: {cfg.weighting!r}')
        del KX, KY, KZ

        # ---- assemble partials with deconvolution + form factors ----
        pair_weights: dict[tuple[str, str], float] = {
            (sp, sp): 1.0 for sp in self.species
        }
        for (a, b) in self.cross_pairs:
            pair_weights[(a, b)] = 2.0

        S_partials = {}
        for key, S in S_diffuse.items():
            a, b = key
            S = S / W2 * f_per[a] * f_per[b] * pair_weights[key]
            S_partials[key] = S

        S_total = cp.zeros(shape_q, dtype=cp.float32)
        for S in S_partials.values():
            S_total += S

        # ---- shift & move to host ----
        def shift_hk(arr_gpu):
            return cp.asnumpy(cp.fft.fftshift(arr_gpu, axes=(0, 1)))

        h_arr = (np.arange(-N // 2, N // 2) / cfg.n_cells).astype(np.float32)
        L_arr = (np.arange(0, N // 2 + 1) / cfg.n_cells).astype(np.float32)

        partials_cpu = {key: shift_hk(S) for key, S in S_partials.items()}
        total_cpu = shift_hk(S_total)

        method = (
            f'3D density binning (CIC, {cfg.n_voxels_per_cell} voxels/cell) '
            f'+ rFFT, {cfg.weighting} weighting, '
            f'{"sub-region averaged ({} regions)".format(cfg.sub_regions) if cfg.sub_regions > 1 else "full box"}, '
            f'exact Bragg subtraction (<F>_t per region), CIC kernel deconvolved'
        )

        return Sq3DResult(
            h_arr=h_arr, k_arr=h_arr.copy(), L_arr=L_arr,
            a_cub=a_cub, L_box=traj.L_box,
            n_frames=NF, n_cells=cfg.n_cells,
            n_voxels_per_cell=cfg.n_voxels_per_cell,
            n_regions=n_R,
            method=method,
            partials=partials_cpu, total=total_cpu,
            elapsed_s=loop_t,
            region_origins=region_origins,
        )

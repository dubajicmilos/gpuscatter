"""Dynamic structure factor S(q, omega) on a user-specified q-set.

Algorithm (the same as PSF / dynasor v2, GPU-accelerated):

1. Compute ``F_a(q, t) = sum_{n in a} exp(-i q . r_n(t))`` per species,
   per frame. (PBC-unwrapped to the first frame.)
2. Subtract the time-mean: ``dF_a = F_a - <F_a>_t`` (exact Bragg subtraction).
3. Time-FFT along the frame axis: ``F_omega_a = FFT_t[dF_a]``.
4. Form X-ray-weighted partials:
       ``S_ab(q, omega) = (1/N_t) f_a(q) f_b(q) Re[F_omega_a F_omega_b*]``.

The q-set is fully user-controlled: pass any ``(n_q, 3)`` array of
q-vectors in 1/A. Two helper functions construct typical q-grids:

* :func:`make_qgrid_HK_plane` for an HK plane at fixed L (e.g. HK1.5).
* :func:`make_qgrid_BZ` for the full first BZ (BZ-folded dispersion).

Memory model: F(q, t) is held on the GPU as ``complex64``,
size ``n_q * n_frames * 8 bytes`` per species. For the 600 K demo
(25 921 q × 5001 frames × 3 species = 3.1 GB) this fits on a GTX 1070
with 5 GB headroom. For larger problems, use chunking over q-points.
"""
from __future__ import annotations
from dataclasses import dataclass, field
from pathlib import Path
import time
from typing import Sequence
import numpy as np

from .form_factors import f_xray
from .trajectory import BaseTrajectory, unwrap_positions


H_PLANCK_meVps = 4.13567        # h in meV * ps


@dataclass
class SqwConfig:
    """Configuration for :class:`Sqw`."""
    q_vecs: np.ndarray              # (n_q, 3) Cartesian q-vectors in 1/A
    dt_fs: float                    # frame spacing in fs
    species: Sequence[str] | None = None
    cross_pairs: Sequence[tuple[str, str]] | None = None
    weighting: str = 'xray'
    atom_chunk: int = 8192          # chunking over atoms in matmul
    sign_convention: str = 'minus'  # 'minus' for exp(-iq.r) (PSF/dynasor), 'plus' for exp(+iq.r)


@dataclass
class SqwResult:
    """Container for dynamic S(q, omega) output."""
    q_vecs: np.ndarray
    E_axis_meV: np.ndarray
    n_frames: int
    dt_fs: float
    method: str
    partials: dict[tuple[str, str], np.ndarray] = field(default_factory=dict)
    total: np.ndarray | None = None
    elapsed_s: float = 0.0

    def save(self, path: str | Path):
        out = {
            'q_vecs': self.q_vecs,
            'E_axis_meV': self.E_axis_meV,
            'n_frames': self.n_frames,
            'dt_fs': self.dt_fs,
            'method': self.method,
        }
        if self.total is not None:
            out['S_total'] = self.total
        for (a, b), S in self.partials.items():
            out[f'S_{a}{b}'] = S
        np.savez(Path(path), **out)


def make_qgrid_HK_plane(L_value: float, a_cub: float,
                        h_max: float = 4.0,
                        n_grid: int = 161
                        ) -> tuple[np.ndarray, np.ndarray, tuple[int, int]]:
    """Build a Cartesian q-grid for an HK plane at fixed L (in r.l.u.).

    Returns
    -------
    h_arr
        (n_grid,) reduced H values (= K values).
    q_vecs
        (n_grid * n_grid, 3) Cartesian q-vectors in 1/A.
    grid_shape
        (n_grid, n_grid) for reshaping the q-axis later.
    """
    h_arr = np.linspace(-h_max, h_max, n_grid, dtype=np.float32)
    H, K = np.meshgrid(h_arr, h_arr, indexing='ij')
    two_pi_a = 2.0 * np.pi / a_cub
    qx = (H * two_pi_a).ravel()
    qy = (K * two_pi_a).ravel()
    qz = np.full(qx.size, L_value * two_pi_a, dtype=np.float32)
    q_vecs = np.stack([qx, qy, qz], axis=1).astype(np.float32)
    return h_arr, q_vecs, (n_grid, n_grid)


def make_qgrid_BZ(n_cells: int, a_cub: float
                  ) -> tuple[np.ndarray, np.ndarray, np.ndarray, tuple[int, int, int]]:
    """Build a 3D q-grid spanning the first BZ at the supercell-RLV step.

    Returns
    -------
    q_red_1d
        (n_cells,) reduced 1D q values in [-1/2, 1/2).
    q_red
        (3, n_q) reduced q in r.l.u.
    q_cart
        (n_q, 3) Cartesian q in 1/A.
    grid_shape
        (n_cells, n_cells, n_cells).
    """
    q_red_1d = (np.arange(n_cells, dtype=np.float32) - n_cells // 2) / n_cells
    QX, QY, QZ = np.meshgrid(q_red_1d, q_red_1d, q_red_1d, indexing='ij')
    q_red = np.stack([QX.ravel(), QY.ravel(), QZ.ravel()], axis=0)
    two_pi_a = 2.0 * np.pi / a_cub
    q_cart = (q_red.T * two_pi_a).astype(np.float32)
    return q_red_1d, q_red, q_cart, (n_cells, n_cells, n_cells)


def _amplitude_inline(positions, q_vecs, sign='minus', chunk=8192):
    """A(q) = sum_n exp(±i q.r_n) with chunking over atoms (CuPy)."""
    import cupy as cp
    n_q = q_vecs.shape[0]
    A_re = cp.zeros(n_q, dtype=cp.float32)
    A_im = cp.zeros(n_q, dtype=cp.float32)
    N = positions.shape[0]
    s = -1.0 if sign == 'minus' else +1.0
    for i in range(0, N, chunk):
        p = positions[i:i + chunk]
        phase = q_vecs @ p.T
        A_re += cp.cos(phase).sum(axis=1)
        A_im += s * cp.sin(phase).sum(axis=1)
    return A_re + 1j * A_im


class Sqw:
    """Dynamic S(q, omega) computation.

    Parameters
    ----------
    trajectory
        :class:`BaseTrajectory`; positions are unwrapped to ``ref_positions``.
    config
        :class:`SqwConfig`.

    Examples
    --------
    HK1.5 plane:

    >>> from gpuscatter.sqw import Sqw, SqwConfig, make_qgrid_HK_plane
    >>> h, q_vecs, gshape = make_qgrid_HK_plane(L_value=1.5, a_cub=6.34)
    >>> cfg = SqwConfig(q_vecs=q_vecs, dt_fs=200.0)
    >>> result = Sqw(traj, cfg).run()

    Full BZ (for phonon dispersion):

    >>> from gpuscatter.sqw import make_qgrid_BZ
    >>> q_red_1d, q_red, q_cart, gshape = make_qgrid_BZ(24, 6.34)
    >>> cfg = SqwConfig(q_vecs=q_cart, dt_fs=200.0)
    """

    def __init__(self, trajectory: BaseTrajectory, config: SqwConfig):
        self.traj = trajectory
        self.cfg = config

        unique_species = sorted(set(trajectory.species.tolist()))
        if config.species is None:
            self.species = unique_species
        else:
            self.species = list(config.species)
        self.species_idx = {sp: np.where(trajectory.species == sp)[0]
                            for sp in self.species}

        if config.cross_pairs is None:
            sp = self.species
            self.cross_pairs = [(sp[i], sp[j])
                                for i in range(len(sp))
                                for j in range(i + 1, len(sp))]
        else:
            self.cross_pairs = list(config.cross_pairs)

    def _energy_axis(self, NF: int) -> np.ndarray:
        """Positive-frequency axis of the time-FFT, in meV."""
        n_omega_pos = NF // 2 + 1
        k_indices = np.arange(n_omega_pos)
        return (k_indices / (NF * self.cfg.dt_fs / 1000.0)
                * H_PLANCK_meVps).astype(np.float32)

    def run(self, verbose: bool = True) -> SqwResult:
        try:
            import cupy as cp
        except ImportError as e:
            raise ImportError(
                'CuPy is required for GPU computation.'
            ) from e

        cfg = self.cfg
        traj = self.traj
        NF = traj.n_frames
        n_q = cfg.q_vecs.shape[0]

        if verbose:
            dev = cp.cuda.runtime.getDeviceProperties(0)['name']
            if isinstance(dev, bytes):
                dev = dev.decode()
            E_max = H_PLANCK_meVps / (2 * cfg.dt_fs / 1000.0)
            dE = H_PLANCK_meVps / (NF * cfg.dt_fs / 1000.0)
            print(f'[gpuscatter.Sqw] device: {dev}')
            print(f'[gpuscatter.Sqw] frames: {NF}, dt = {cfg.dt_fs} fs, '
                  f'n_q = {n_q}, species = {self.species}')
            print(f'[gpuscatter.Sqw] E_max (Nyquist) = {E_max:.3f} meV, '
                  f'dE = {dE * 1000:.3f} ueV')
            mem_gb = n_q * NF * 8 / 1e9 * len(self.species)
            print(f'[gpuscatter.Sqw] F(q,t) GPU storage: {mem_gb:.2f} GB '
                  f'(complex64, {len(self.species)} species)')

        # ---- form factors ----
        q_norm = np.linalg.norm(cfg.q_vecs, axis=1).astype(np.float64)
        if cfg.weighting == 'xray':
            f_q = {sp: f_xray(q_norm, sp).astype(np.float32)
                   for sp in self.species}
        elif cfg.weighting == 'unit':
            f_q = {sp: np.ones(n_q, dtype=np.float32) for sp in self.species}
        else:
            raise ValueError(f'Unknown weighting: {cfg.weighting!r}')

        # ---- F(q, t) GPU storage ----
        F = {sp: cp.zeros((n_q, NF), dtype=cp.complex64)
             for sp in self.species}
        qv_gpu = cp.asarray(cfg.q_vecs, dtype=cp.float32)

        # ---- frame loop ----
        t0 = time.time()
        for global_idx, p in traj.iter_frames():
            p_unwrapped = unwrap_positions(p, traj.ref_positions, traj.L_box)
            for sp in self.species:
                p_sp_gpu = cp.asarray(p_unwrapped[self.species_idx[sp]],
                                       dtype=cp.float32)
                A = _amplitude_inline(p_sp_gpu, qv_gpu,
                                       sign=cfg.sign_convention,
                                       chunk=cfg.atom_chunk)
                F[sp][:, global_idx] = A
            if verbose and ((global_idx + 1) % 200 == 0
                            or global_idx + 1 == NF):
                cp.cuda.runtime.deviceSynchronize()
                elapsed = time.time() - t0
                rate = (global_idx + 1) / max(elapsed, 1e-9)
                eta = (NF - (global_idx + 1)) / max(rate, 1e-9)
                print(f'[gpuscatter.Sqw]   frame {global_idx+1}/{NF}, '
                      f'{elapsed:.0f}s, eta {eta:.0f}s', flush=True)

        loop_t = time.time() - t0
        if verbose:
            print(f'[gpuscatter.Sqw] amplitudes: {loop_t:.0f}s')

        # ---- exact Bragg subtraction ----
        for sp in self.species:
            F[sp] -= cp.mean(F[sp], axis=1, keepdims=True)

        # ---- time-FFT (cuFFT batched 1D) ----
        t_fft = time.time()
        F_omega = {sp: cp.fft.fft(F[sp], axis=1) for sp in self.species}
        for sp in self.species:
            del F[sp]
        cp.get_default_memory_pool().free_all_blocks()
        if verbose:
            print(f'[gpuscatter.Sqw] time-FFT: {time.time() - t_fft:.1f}s')

        n_omega_pos = NF // 2 + 1
        E_axis = self._energy_axis(NF)

        # ---- partials ----
        cp_fq = {sp: cp.asarray(f_q[sp]) for sp in self.species}
        partials = {}

        all_pairs = [(sp, sp) for sp in self.species] + list(self.cross_pairs)
        for (a, b) in all_pairs:
            w = 1.0 if a == b else 2.0
            S = cp.real(F_omega[a] * cp.conj(F_omega[b])) / NF
            S = S * cp_fq[a][:, None] * cp_fq[b][:, None] * w
            S = S[:, :n_omega_pos]
            partials[(a, b)] = cp.asnumpy(S).astype(np.float32)
            del S

        total = np.zeros_like(next(iter(partials.values())))
        for S in partials.values():
            total += S

        method = (
            f'GPU direct atomic Fourier sum (CuPy), '
            f'{cfg.weighting} weighting, exact Bragg subtraction, '
            f'sign exp({cfg.sign_convention} i q.r)'
        )

        return SqwResult(
            q_vecs=cfg.q_vecs,
            E_axis_meV=E_axis,
            n_frames=NF, dt_fs=cfg.dt_fs,
            method=method,
            partials=partials, total=total,
            elapsed_s=loop_t,
        )

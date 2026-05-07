"""3D delta-PDF (partial diffuse Patterson) per partial S_ab(q).

For each Bragg-subtracted partial ``S_ab(q)`` on the supercell-RLV grid
(produced by :class:`gpuscatter.sq3d.Sq3D`), compute the inverse 3D FFT
to get the real-space partial diffuse Patterson:

    Delta_PDF_ab(r) = iFT[ S_ab^diffuse(q) ]

This is the local pair-correlation function for atom species ``(a, b)``,
with the average-crystal periodic part subtracted. The delta-PDF carries
the local-correlation information that the full PDF buries under sharp
Bragg lattice peaks.

References
----------
Welberry, *Diffuse X-ray Scattering and Models of Disorder* (OUP, 2010).

Weber & Simonov, *Z. Krist.* **227**, 238 (2012) — 3D-DeltaPDF formalism.

Simonov et al., *J. Appl. Cryst.* **47**, 1146 (2014) — Yell.

Simonov & Goodwin, *Nat. Rev. Chem.* **4**, 657 (2020) — review.
"""
from __future__ import annotations
from dataclasses import dataclass, field
from pathlib import Path
import numpy as np

from .sq3d import Sq3DResult


@dataclass
class DeltaPDFResult:
    """Container for the 3D delta-PDF per partial."""
    r_arr: np.ndarray              # 1D axis (A), shared by x, y, z
    a_cub: float
    L_box: float
    dx: float
    n_total: int
    method: str
    pdfs: dict[tuple[str, str], np.ndarray] = field(default_factory=dict)
    total: np.ndarray | None = None

    def save(self, path: str | Path):
        out = {
            'r_arr': self.r_arr,
            'a_cub': self.a_cub, 'L_box': self.L_box,
            'dx': self.dx, 'n_total': self.n_total,
            'method': self.method,
        }
        if self.total is not None:
            out['P_total'] = self.total
        for (a, b), P in self.pdfs.items():
            out[f'P_{a}{b}'] = P
        np.savez(Path(path), **out)


def compute_delta_pdf(sq_result: Sq3DResult,
                      use_gpu: bool = False) -> DeltaPDFResult:
    """Compute the 3D delta-PDF from an :class:`Sq3DResult`.

    Parameters
    ----------
    sq_result
        Output of :meth:`gpuscatter.sq3d.Sq3D.run`. The Bragg-subtracted,
        CIC-deconvolved, X-ray-weighted partials are inverse-Fourier
        transformed.
    use_gpu
        If True, use CuPy's irfftn. Otherwise NumPy. The arrays are
        small enough (192^3 = 28 MB / partial) that CPU is fine.

    Returns
    -------
    DeltaPDFResult
        Per-partial real-space delta-PDF on a centered (n_total^3) grid.
    """
    N = sq_result.n_cells * sq_result.n_voxels_per_cell
    L_box = sq_result.L_box
    dx = L_box / N
    r_arr = (np.arange(-N // 2, N // 2) * dx).astype(np.float32)

    if use_gpu:
        try:
            import cupy as xp
        except ImportError:
            xp = np
    else:
        xp = np

    pdfs: dict[tuple[str, str], np.ndarray] = {}
    for key, S_q_shifted in sq_result.partials.items():
        # Saved S is fftshift'd along (0, 1) but L axis is the rfft half.
        S_q_unshifted = np.fft.ifftshift(S_q_shifted, axes=(0, 1))
        S_complex = S_q_unshifted.astype(np.complex64)
        if xp is np:
            PDF = np.fft.irfftn(S_complex, s=(N, N, N))
        else:
            PDF = xp.asnumpy(
                xp.fft.irfftn(xp.asarray(S_complex), s=(N, N, N))
            )
        pdfs[key] = np.fft.fftshift(PDF).astype(np.float32)

    total_pdf = None
    if sq_result.total is not None:
        S_t_unshifted = np.fft.ifftshift(sq_result.total, axes=(0, 1))
        if xp is np:
            T = np.fft.irfftn(S_t_unshifted.astype(np.complex64),
                              s=(N, N, N))
        else:
            T = xp.asnumpy(xp.fft.irfftn(
                xp.asarray(S_t_unshifted.astype(np.complex64)),
                s=(N, N, N)
            ))
        total_pdf = np.fft.fftshift(T).astype(np.float32)

    method = (
        'iFT of Bragg-subtracted partial S_ab(q); '
        'CIC-deconvolved S used; full-box rfftn cube zero-padded '
        'on negative-L axis via Hermitian symmetry'
    )

    return DeltaPDFResult(
        r_arr=r_arr, a_cub=sq_result.a_cub, L_box=L_box,
        dx=dx, n_total=N, method=method,
        pdfs=pdfs, total=total_pdf,
    )

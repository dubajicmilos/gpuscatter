"""Example 2 — Compute the 3D Delta-PDF from the static S(q) cube.

For each Bragg-subtracted partial ``S_ab(q)`` produced by Example 1,
inverse-Fourier transform to a real-space partial Patterson:

    Delta_PDF_ab(r) = iFT[S_ab^diffuse(q)]

The Delta-PDF carries the local-correlation information that the full
PDF buries under sharp Bragg lattice peaks. Each peak in the partial
PDF sits at the average pair vector, with amplitude proportional to
the displacement correlation strength.

Expected wall time: a few seconds on CPU (the heavy lifting was done
by the forward S(q) calculation).
"""
from __future__ import annotations
from pathlib import Path
import sys

import numpy as np

_REPO = Path(__file__).resolve().parent.parent
if str(_REPO) not in sys.path:
    sys.path.insert(0, str(_REPO))

from gpuscatter import compute_delta_pdf      # noqa: E402
from gpuscatter.sq3d import Sq3DResult        # noqa: E402


def load_sq3d(npz_path: Path) -> Sq3DResult:
    """Reconstruct an Sq3DResult from a saved NPZ."""
    d = np.load(npz_path, allow_pickle=True)
    species = sorted(set(
        ''.join(filter(str.isupper, k[2:]))[:1]  # crude — replaced below
        for k in d.files if k.startswith('S_') and k != 'S_total'
    ))
    # Re-parse partial keys precisely
    partials = {}
    for k in d.files:
        if not k.startswith('S_') or k == 'S_total':
            continue
        body = k[2:]
        # Split body into two species. Assume capitalization: species
        # symbols are 1- or 2-letter and start with an uppercase.
        if len(body) >= 2 and body[1].isupper():
            a = body[:1]
        elif len(body) >= 3 and body[2].isupper():
            a = body[:2]
        else:
            a = body[:2] if len(body) >= 4 else body[:1]
        b = body[len(a):]
        partials[(a, b)] = d[k]

    return Sq3DResult(
        h_arr=d['h_arr'], k_arr=d['k_arr'], L_arr=d['L_arr'],
        a_cub=float(d['a_cub']), L_box=float(d['L_box']),
        n_frames=int(d['n_frames']),
        n_cells=int(d['n_cells']),
        n_voxels_per_cell=int(d['n_voxels_per_cell']),
        n_regions=int(d['n_regions']),
        method=str(d['method']),
        partials=partials,
        total=d['S_total'] if 'S_total' in d.files else None,
    )


def main(in_path: Path, out_path: Path | None = None):
    print(f'Loading S(q) cube from {in_path.name}...')
    sq = load_sq3d(in_path)
    print(f'  shape per partial: {next(iter(sq.partials.values())).shape}')
    print(f'  partials: {[f"{a}{b}" for a, b in sq.partials]}')

    print('Computing inverse 3D rFFT per partial...')
    pdf = compute_delta_pdf(sq, use_gpu=False)
    print(f'  Delta-PDF shape: {next(iter(pdf.pdfs.values())).shape}')
    print(f'  voxel: dx = {pdf.dx:.3f} A')
    print(f'  span: r in [{pdf.r_arr[0]:.1f}, {pdf.r_arr[-1]:.1f}] A')

    if out_path is None:
        out_path = in_path.with_name(
            in_path.stem.replace('sq3d', 'delta_pdf') + '.npz'
        )
    pdf.save(out_path)
    print(f'Saved {out_path.name}'
          f' ({out_path.stat().st_size / 1e6:.1f} MB)')


if __name__ == '__main__':
    import argparse
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument('--in-npz', type=Path,
                        default=Path(__file__).parent
                        / 'sq3d_CsPbI3_600K_8reg.npz')
    parser.add_argument('--out', type=Path, default=None)
    args = parser.parse_args()
    main(args.in_npz, args.out)

# Examples

Five runnable examples covering the full feature set of `gpuscatter`,
all using the **CsPbI₃ 600 K demo trajectory** (Baldwin et al. 2024,
1 ns at 200 fs spacing, 24³ supercell, 5001 frames).

| # | File | Computes | Wall time on GTX 1070 |
|---|---|---|---:|
| 1 | `01_static_sq3d.py` | Full 3D S(q) cube on 192³ q-grid (all 97 unique L-planes at once) | **1.7 min (full box) / ~14 min (8-region averaged)** |
| 2 | `02_delta_pdf.py` | 3D ΔPDF per partial via inverse FFT | < 5 s |
| 3 | `03_dynamic_sqw_HK_plane.py` | S(q, ω) on a chosen HK plane (default L = 1.5) | **21 min** |
| 4 | `04_dynamic_sqw_BZ.py` | Full first-BZ S(q, ω) on 24³ q-grid for phonon dispersion | **11 min** |
| 5 | `05_dispersion_paths.py` | Project the BZ S(q, ω) along Γ–X–M–R–Γ | < 5 s |

For comparison: the equivalent 1 plane S(q, ω) on a single CPU using
dynasor v2 takes ~5 hours; the equivalent 3D S(q) cube via direct
atomic Fourier sum takes ~34 hours. **gpuscatter is 14–75× faster.**

## Quick start

```bash
# Edit DEFAULT_TRAJ_DIR in each script to point at your data, then:
python 01_static_sq3d.py --n-regions 8         # 14 min: ripple-suppressed
python 02_delta_pdf.py                          # ~5 s
python 03_dynamic_sqw_HK_plane.py --L 1.5       # 21 min
python 04_dynamic_sqw_BZ.py                     # 11 min
python 05_dispersion_paths.py                   # ~5 s
```

For a **fast smoke test** (one trajectory file = 501 frames instead of
5001), the wall time drops to ~12 s for the static 3D S(q) and
~2 min for S(q, ω) — try this first to verify the GPU pipeline works.

## Memory budgets

| Example | GPU peak | Disk output |
|---|---:|---:|
| 1, full box | 1.5 GB | 100 MB |
| 1, 8 regions | 1.8 GB | 100 MB |
| 2 | 0 (CPU) | 200 MB |
| 3, HK plane (3 species, 5001 frames) | 3.1 GB | 1.8 GB |
| 4, full BZ (3 species, 5001 frames) | 1.7 GB | 970 MB |
| 5 | 0 | 200 KB PNG |

All comfortably fit on an 8 GB GTX 1070. Larger problems (more L planes
in one shot, finer ω resolution, larger supercells) just chunk over the
q-axis without changing wall time meaningfully.

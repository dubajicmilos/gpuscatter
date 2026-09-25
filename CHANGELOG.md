# Changelog

## Unreleased

### Elements

* **Periodic-table-wide coverage** for X-ray and neutron weighting.
  `CROMER_MANN` now carries 65 neutral elements (Cromer-Mann
  coefficients from International Tables Vol. C, Table 6.1.1.4) and
  `B_NEUTRON` carries 62 elements (natural-abundance-weighted
  coherent scattering lengths from the NIST neutron data tables).
  `ATOMIC_NUMBER_TO_SYMBOL` extended to match. Coverage now spans
  Z = 1-57 contiguously plus the heavy elements common in materials
  research (Hf, Ta, W, Pt, Au, Hg, Tl, Pb, Bi). Earlier limited set
  (~29 elements) only covered halide perovskites.
* New regression test `test_cromer_mann_has_full_periodic_table_coverage`
  and the existing `test_f_xray_at_zero_returns_atomic_number` test
  extended to validate `f(q=0) = Z` for every new entry.
* Enables Sq3D / Sqw on metallic systems (NiAl, Fe-Cr, Cu-Zn),
  chalcogenides, oxides, and almost any other condensed-matter target.

### Documentation

* **q_Nyquist edge artifact** documented in the `Sq3D` and `Sq3DConfig`
  docstrings and in a new "Caveat" subsection of the README. The outer
  ~15 % of the FFT q-grid (`|q| > 0.85 * n_voxels_per_cell / 2 r.l.u.`)
  is contaminated by aliasing across `q_Nyquist` and the `1/sinc^4` CIC
  deconvolution overshoot. This is a property of every density-binning
  + FFT pipeline; users should pick `n_voxels_per_cell >= 2.4 * q_max`
  for the highest q they need and trim the displayed range. The 2D
  Butler-Welberry direct-sum partials do not have this artifact.

### Features

* `Sq3DResult.q_max_clean` property — recommended upper q (in r.l.u.)
  for trusted signal, set to `0.85 * n_voxels_per_cell / 2`.
* `Sq3DResult.trim(q_max=None)` method — return a copy with `h_arr`,
  `k_arr`, `L_arr`, `partials`, and `total` restricted to
  `|q| <= q_max` (default = `q_max_clean`). Drops the contaminated
  outer band so downstream code that does not check `q_max_clean` is
  also protected.
* `Sq3D.run` verbose output now prints `q_max_clean` and points users
  at `result.trim()`.

### Symmetry

* **Laue-group orbit averaging** of 3D reciprocal-space volumes, ported
  from rspace3d. `symmetrize_volume(h_arr, k_arr, L_arr, intensity,
  laue_group)` gathers, for every voxel, the full set of
  symmetry-equivalent positions (the *orbit* `{S(R q)}`) and writes
  their mean. Supports the seven signed-permutation Laue groups
  (`-1, 2/m, mmm, 4/m, 4/mmm, m-3, m-3m`) — every cubic / tetragonal /
  orthorhombic perovskite phase; trigonal and hexagonal are not
  included. Exposed at top level alongside `get_symmetry_operations`
  and `expand_rfft_L`.
* **Why it is useful:** folding the `N_ops` crystallographically
  equivalent copies of a `<|F|^2>`-type diffuse map cuts the per-voxel
  statistical noise by up to `sqrt(N)` (e.g. up to `sqrt(8)` for
  `4/mmm` on an HKx plane, where the L-flipped orbit members are exact
  Friedel duplicates of the in-plane ones) *without changing the
  q-grid*. This is signal-to-noise that cannot be recovered by
  computing on a finer grid, which leaves per-pixel noise intact. On an
  8001-frame MAPbBr3 X-ray cube the per-voxel scatter drops from ~2 %
  to ~1 % and individual noise spikes collapse onto their orbit
  consensus, while the diffuse structure is preserved.
* **Exact on the grid, no interpolation:** each Laue operation is a
  signed permutation of `(h, k, l)`, so it maps grid nodes onto grid
  nodes exactly, provided the grid spans both `+q` and `-q` for every
  axis the group flips.
* `expand_rfft_L(h_arr, k_arr, L_arr, vol)` Friedel-expands a `Sq3D`
  rfft half-spectrum (`L >= 0`) into the full symmetric `+-L` volume
  that the L-flipping operations require, via `S(H,K,-L)=S(-H,-K,L)`.
  The in-plane axes must be symmetric about 0 (trim to `|q| <= q_max`
  first).
* Optional per-orbit MAD outlier rejection (`sigma=`) for
  measurement-style data; the default `sigma=None` does a pure orbit
  mean, appropriate for computed S(q).

### I/O

* `Sq3DResult.save_rspace3d(path, channel='total', *, full_l=True,
  wavelength=1.0, compression='gzip', compression_level=4)` writes the
  total or one partial as an HDF5 volume in the layout of rspace3d's
  `save_volume_h5`, which rspace3d's `load_volume_h5` and
  xrays-on-detector's `SqVolume` read: `data` (float32, axis order
  H, K, L), float64 axes `H`, `K`, `L`, `UB = wavelength * I / a_cub`,
  and the attributes `wavelength`, `grid_kind = 'hkl_regular'`,
  `plane_type = 'HK'`, the cubic cell and the provenance keys
  `gpuscatter_channel`, `gpuscatter_method` and `gpuscatter_n_frames`.
  The wavelength only scales UB and has no physical meaning for a
  simulation.
* With `full_l=True` (the default) it rebuilds the full `+-L` volume
  from the rfft half-spectrum with `expand_rfft_L`, which needs H and K
  axes symmetric about 0, so call it on `result.trim()`.
* The axes are written in float64 as exact multiples of `1 / n_cells`.
  A plain float64 cast of Sq3D's float32 axes makes the step vary by
  more than 1e-6 (relative) for `n_cells = 24`, and xrays-on-detector
  rejects such an axis as non-uniform.

### Tests

* `tests/test_sq3d_config.py` covers the new property, the `n_total`
  field, and 9 cases for `trim()` (default vs custom q_max, value
  preservation, shape consistency, metadata propagation, error paths,
  idempotence). No GPU required.
* `tests/test_symmetry.py` — 7 tests for Laue group orders and
  orthogonality, the synthetic `mmm` orbit-mean hand check, the Friedel
  `expand_rfft_L` round-trip, and `4/mmm` idempotence + 4-fold/L-mirror
  invariance. No GPU required.
* `tests/test_rspace3d_io.py`: 4 tests for `save_rspace3d`, covering
  every dataset and attribute with its dtype and value, the
  Friedel-expanded full-L volume, channel selection and the error
  paths, and the exact float64 axes. No GPU required; needs h5py.

## 0.1.0 — 2026-05-07

Initial public release.

### Features

* **Sq3D**: GPU 3D static partial S(q) cube via density binning
  (cloud-in-cell) + 3D rFFT. The user picks the simulation supercell
  size and the voxels-per-cell, which together fix the q step and
  q_max; all reciprocal-space planes come out of one calculation.
  Demo benchmark (5001 frames, 24³ supercell, 8 voxels per cell, so
  192 × 192 × 97 rFFT half-grid): **1.7 min on a GTX 1070**.
* **Sqw**: GPU dynamic structure factor S(q, ω) on any user q-set, via
  direct atomic Fourier amplitude + cuFFT batched 1D time-FFT. **20 min
  for the full HK1.5 plane on a GTX 1070**, vs ~5 h on single-CPU dynasor
  v2.
* **compute_delta_pdf**: 3D delta-PDF (partial diffuse Patterson) per
  X-ray partial via inverse 3D rFFT. **< 5 s** total; the heavy lifting
  was done by Sq3D.
* **DispersionProjection**: project a BZ-folded `S(q, ω)` cube onto
  cubic high-symmetry paths Γ-X-M-R-Γ.
* X-ray Cromer-Mann form factors for 22 elements; neutron coherent
  scattering lengths for the same.
* Trajectory readers: NpzTrajectory (Baldwin et al. format),
  SingleNpzTrajectory, plus a BaseTrajectory abstraction for custom
  formats.

### Examples

* `01_static_sq3d.py` — full 3D S(q) cube on CsPbI3 600 K.
* `02_delta_pdf.py` — 3D ΔPDF from the same cube.
* `03_dynamic_sqw_HK_plane.py` — S(q, ω) on HK1.5.
* `04_dynamic_sqw_BZ.py` — S(q, ω) on the full first BZ.
* `05_dispersion_paths.py` — Γ-X-M-R-Γ phonon-dispersion projection.

### Benchmarks

* `benchmark_sq3d.py` — 3D static S(q) GPU vs CPU.
* `benchmark_sqw.py` — Dynamic S(q, ω) GPU vs CPU.

### Tests

* 15 unit tests covering form factors, q-grid construction, and
  dispersion path projection.

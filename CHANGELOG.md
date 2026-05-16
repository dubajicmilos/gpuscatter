# Changelog

## Unreleased

### Elements

* Added 3d transition-metal coverage: **Cr, Mn, Fe, Co, Ni, Cu, Zn**.
  Cromer-Mann X-ray coefficients (International Tables Vol. C, Table
  6.1.1.4) and NIST neutron coherent scattering lengths added to
  `gpuscatter/form_factors.py`. Corresponding atomic-number entries
  (Z = 24-30) added to `ATOMIC_NUMBER_TO_SYMBOL` in
  `gpuscatter/trajectory.py`. Validated by the existing
  `test_f_xray_at_zero_returns_atomic_number` test (extended to cover
  the new elements and Al). Enables Sq3D on metallic systems such as
  NiAl, Fe-Cr, Cu-Zn alloys etc.

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

### Tests

* `tests/test_sq3d_config.py` covers the new property, the `n_total`
  field, and 9 cases for `trim()` (default vs custom q_max, value
  preservation, shape consistency, metadata propagation, error paths,
  idempotence). No GPU required.

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

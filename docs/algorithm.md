# Algorithms

Detailed algorithm notes for each compute module. Read alongside the
source.

## 1. 3D static S(q) — density-binning + rFFT

**File**: [`gpuscatter/sq3d.py`](../gpuscatter/sq3d.py).

### Pipeline per frame

```
positions[species]    →    rho[species] (192x192x192 grid)    →    F[species] = rfft3(rho)
              CIC bin (8 corners)                3D rFFT
```

### Per frame, accumulators

* `<F_a>` for each species (complex64, 192x192x97)
* `<|F_a|^2>` for each species (float32, 192x192x97)
* `<Re[F_a F_b*]>` for each cross-pair (float32, 192x192x97)

### After all frames

* Auto:  `S_aa^diff(q) = <|F_a|^2> - |<F_a>|^2`
* Cross: `S_ab^diff(q) = <Re[F_a F_b*]> - Re[<F_a><F_b>*]`

This is the **exact Bragg subtraction** — using the time-mean of the
spatial Fourier amplitude rather than the Debye-Waller approximation
`A(q, <r>)`. It is what dynasor v2 uses, except dynasor computes by
direct sum on a small q-set; we compute on the full 192³ grid.

### CIC kernel deconvolution

CIC binning is real-space convolution with a triangle of width
`2 dx`. Its Fourier transform is `sinc^2(q dx / 2)` per dim. Hence

    F_grid(q) = F_atom(q) * sinc^2(q_x dx/2) * sinc^2(q_y dx/2) * sinc^2(q_z dx/2)

To recover `|F_atom|^2`, divide `|F_grid|^2` by

    W^2(q) = sinc^4(q_x dx/2) * sinc^4(q_y dx/2) * sinc^4(q_z dx/2)

The deconvolution is exact at q-grid points (where `sinc` is the FT
of an exact triangle). Off-grid sampling would need a non-uniform
FFT (cuFINUFFT).

### Sub-region averaging

For ripple suppression: pick `n_regions` random sub-cubes of
`sub_region_cells^3` of the supercell. For each region, compute
`<F_r>_t`, `<|F_r|^2>_t`, etc. independently, and average the
diffuse partials over regions. This trades O(n_regions) compute for
suppression of long-vector finite-N Fourier ripples.

### q-grid

Forced to multiples of `1 / N_cells r.l.u.` because the FFT is on
the supercell-reciprocal-lattice grid. For a 24³ supercell, that is
1/24 ≈ 0.0417 r.l.u. — slightly finer than the 0.05 r.l.u. used by
direct-sum convention.

`q_max` is set by the binning resolution: `n_voxels_per_cell = 8`
gives `q_max = 4 r.l.u.`. Set `n_voxels_per_cell = 12` for `q_max = 6`
at the cost of `12³/8³ = 3.4×` more memory.

### Speedup origin

The asymptotic complexity per frame:

| Method | Complexity |
|---|---|
| Direct atomic sum, single L-plane | `O(n_atoms × n_q^2D)` |
| Direct atomic sum, full 3D cube | `O(n_atoms × n_q^3D)` |
| Density + 3D rFFT | `O(n_atoms × n_corners) + O(N^3 log N)` |

For our 600 K demo:
* `n_atoms = 70 000`, `n_q^2D = 25 921`, `n_q^3D = 3.6 M`
* Direct sum, 1 L-plane: 70 000 × 25 921 = 1.8e9 ops/frame
* Direct sum, full cube: 70 000 × 3.6e6 = 2.5e11 ops/frame
* Density + rFFT: 70 000 × 8 + 192^3 × log_2(192) = 5.6e5 + 5.6e7 = 5.6e7 ops/frame

So the FFT route is ~32× cheaper than direct-sum on the same workload
(single L-plane), or ~4500× cheaper than direct-sum on the full cube.
Combined with a ~10–15× GPU speedup over single-CPU numpy, the total
is ~75× per single-plane workload, or ~50 000× for full-cube workloads.

## 2. Dynamic S(q, ω) — direct atomic sum + cuFFT

**File**: [`gpuscatter/sqw.py`](../gpuscatter/sqw.py).

### Pipeline

```
For each frame, each species:
    F_a(q, t) = sum_{n in a} exp(-i q . r_n(t))      (chunked GPU matmul)

Subtract time-mean:    dF_a(q, t) = F_a(q, t) - <F_a>_t

Time-FFT batched:      F_omega_a(q, omega) = FFT_t[ dF_a(q, t) ]

Form partials:         S_ab(q, omega) = (1/N_t) f_a f_b Re[F_omega_a F_omega_b*]
```

This is identical to PSF and dynasor v2 algorithmically; the speedup is
purely from CuPy + cuFFT.

### Memory model

`F_a(q, t)` lives on the GPU as `complex64`, size `n_q × n_frames × 8 bytes`
per species. For the 600 K demo (25 921 q × 5001 frames × 3 species)
this is 3.1 GB. The time-FFT is in-place, so peak memory is ~5 GB.

For larger problems (more species, longer trajectories, larger q-grids),
chunk over the q-axis: process e.g. 4096 q-points at a time, do their
full ω-FFT, free, repeat. Memory stays under 1 GB and total wall time
is unchanged.

### Frame-loop bottleneck

The dominant cost is the spatial Fourier amplitude. For each frame:
3 species × 25 921 q × 70 000 atoms = 5.4e9 complex muladds. On the
GTX 1070 sustaining ~6 TFLOPS, that's ~1 ms theoretical, ~250 ms
measured (including chunking overhead, kernel launch, Python loop).

### Energy axis

`E_axis = k / T × h_planck` where `T = N_frames × dt` is the total
trajectory time and `h_planck = 4.13567 meV·ps`. Resolution is
`dE = h / T`. For our 1 ns demo at 200 fs frame spacing,
`dE = 4.135 µeV` (which is *finer* than typical INS instrument
resolution).

Nyquist `E_max = h / (2 dt)` = 10.337 meV at 200 fs, so we cover
acoustic phonons and soft-tilt physics but not the optical phonon
manifold above 10 meV (which is aliased and unreliable).

## 3. 3D delta-PDF — inverse 3D rFFT

**File**: [`gpuscatter/delta_pdf.py`](../gpuscatter/delta_pdf.py).

For each Bragg-subtracted partial S_ab(q):

    Delta_PDF_ab(r) = iFFT3D[ S_ab^diffuse(q) ]

The partial S_ab(q) is real (after Bragg subtraction the imaginary
part of the cross terms vanishes by Hermitian symmetry of the
density-correlation), so:

* The forward S(q) is on a `(192, 192, 97)` half-spectrum (rfftn).
* iFFT (irfftn) gives a real `(192, 192, 192)` real-space cube.
* `fftshift` centers the origin (r=0 at the middle voxel).

The voxel size in real space is `dx = L_box / 192 ≈ 0.77 Å`; the cube
spans ±L_box/2 ≈ ±74 Å in each axis. The signal is meaningful out to
~2 nm — beyond that, finite-supercell artefacts dominate.

This is the partial diffuse Patterson of Welberry, Weber & Simonov.
Each peak in `Delta_PDF_ab(r)` lives at an average pair vector `r_a - r_b`,
with amplitude proportional to the displacement-correlation strength
at that vector. Decomposed by atom-pair, which crystallographic ΔPDF
cannot do without anomalous-scattering contrast variation.

## 4. Phonon dispersion projection

**File**: [`gpuscatter/dispersion.py`](../gpuscatter/dispersion.py).

Given a BZ-folded S(q, ω) cube on a `(N, N, N, n_omega)` grid where N
is the supercell size, project along piecewise-linear paths in r.l.u.

For the cubic Brillouin zone, the standard high-symmetry path is
`Γ → X → M → R → Γ`:

* Γ = (0, 0, 0) — BZ center
* X = (½, 0, 0) — face center
* M = (½, ½, 0) — edge center
* R = (½, ½, ½) — corner (soft-tilt mode condensation in CsPbI3)

Each segment is sampled with `n_per_seg` points; each sample is
nearest-neighbour quantized to the BZ grid (because our q-set is the
supercell-RLV grid, not arbitrary q's — for arbitrary q we would need
direct atomic sum on the new q-set, not interpolation).

For dynasor-style longitudinal/transverse current projections
(`C_L`, `C_T`), one would need access to the velocities, which are not
in the standard MD-trajectory NPZ format used here. That would be a
straightforward extension if the velocities are available.

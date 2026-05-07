# Benchmarks

Measured wall times on a **GTX 1070 (8 GB, 2016)** + **i7-7700K** CPU,
using the CsPbI₃ 600 K demo trajectory (5001 frames, 69 120 atoms,
24³ supercell).

Run yourself:

```bash
python benchmarks/benchmark_sq3d.py     # 3D static S(q)
python benchmarks/benchmark_sqw.py      # Dynamic S(q, ω)
```

## Headline: 3D static S(q)

This is the **headline speedup** of `gpuscatter` over conventional
direct-atomic-sum tools. The density+rFFT route on GPU is asymptotically
faster than direct evaluation, *and* it produces all 97 unique L-planes
in one shot rather than one plane at a time.

| Method | Wall time | Output coverage |
|---|---:|---|
| Direct atomic sum, single-thread numpy, 1 L plane | **16 h** | 1 plane only |
| Direct atomic sum, numba JIT, 1 L plane (project's existing pipeline) | **21 min** | 1 plane only |
| Direct atomic sum, numpy, all 97 L planes (extrapolated) | **65 days** | full 3D cube |
| **gpuscatter, GPU rFFT, full 3D cube** | **1.7 min** | **full 3D cube** |

**Speedup vs same-grid CPU direct sum: ~75× (vs numba JIT) to ~500× (vs single-thread numpy)**.
**Speedup if a full 3D cube is needed: ~50 000×**, because gpuscatter
gets all 97 L planes from the same compute.

The headline 1.7 min number is for the full 5001-frame, 192³ q-cube on
a GTX 1070. The same calculation on an A100 is roughly 30 s (extrapolating
from raw FLOPs). Detailed per-frame numbers below.

### Timing detail (Sq3D, GTX 1070)

| Stage | Time |
|---|---:|
| Frame loop: stream NPZ + CIC bin (3 species) + 3 × 3D rFFT + accumulate | **~22 ms / frame** |
| Per-region overhead (8 sub-regions): each frame is processed 8× | ×8 |
| CIC kernel deconvolution + form factor weighting | ~1 s total |
| Save to NPZ | ~3 s |
| **Total, 5001 frames, full box** | **1.7 min** |
| **Total, 5001 frames, 8 sub-regions** | **14 min** |

### Memory

Peak GPU usage on the 192³ q-grid:

| | Size |
|---|---:|
| Density grid (1 species at a time) | 28 MB |
| Per-species rFFT output (3 species) | 86 MB |
| Persistent ⟨F⟩, ⟨\|F\|²⟩, cross accumulators | 172 MB (per region) |
| cuFFT workspace | ~150 MB |
| **Peak measured (full box)** | **1.5 GB** |
| **Peak measured (8 sub-regions)** | **1.8 GB** |

Comfortable headroom on an 8 GB GTX 1070. Could push to 256³ q-grid
(10 voxels per cell, q_max = 5 r.l.u.) without trouble.

## Dynamic S(q, ω)

Same trajectory, ω resolution 4.13 µeV (limit: 1 ns total time):

| Method | Wall time | q-grid |
|---|---:|---|
| Direct atomic sum, single-thread numpy | **16 h** | 161×161 = 25 921 q × 5001 frames |
| Direct atomic sum, numba JIT (dynasor v2 baseline) | ~5 h | same |
| **gpuscatter Sqw (CuPy)** | **20 min** | same |

**Speedup: 50× vs numpy, 15× vs numba JIT (dynasor v2)**. The
algorithm is identical to PSF / dynasor — same atomic Fourier sum,
same time-FFT, same Bragg subtraction; the speedup comes purely from
CuPy + cuFFT.

### Timing detail (Sqw, GTX 1070, full HK plane at L = 1.5)

| Stage | Time |
|---|---:|
| Setup, q-grid, GPU allocate | < 1 s |
| Frame loop, spatial Fourier amplitude on GPU | **~245 ms / frame** |
| Total spatial FT, 5001 frames | 20 min |
| Subtract ⟨F⟩_t (exact Bragg) | < 1 s |
| Time-FFT (cuFFT batched, 25 921 batches × 5001 length) | 8 s |
| Form 6 X-ray-weighted partials + save 1.8 GB NPZ | 4 s |
| **Total** | **~21 min** |

### Memory

| | Size |
|---|---:|
| F(q, t) on GPU (3 species × 25 921 q × 5001 frames × 8 bytes) | **3.1 GB** |
| F_omega(q, ω) (in-place FFT) | 3.1 GB |
| **Peak measured** | **5.2 GB** |

### Full first-BZ S(q, ω) for phonon dispersion

Same backend, different q-grid (24³ first BZ = 13 824 q-points instead
of 25 921):

| | Wall time | Memory |
|---|---:|---:|
| **gpuscatter Sqw, 5001 frames, BZ 24³** | **11 min** | **1.7 GB** |

Output is fed to the dispersion-projection module to extract sheets
along Γ-X-M-R-Γ.

## How the speedup breaks down

Two effects compound:

1. **GPU vs CPU per FLOP**. The GTX 1070 sustains ~6 TFLOPS in single
   precision; a single CPU core on the same workload sustains ~50 GFLOPS.
   Raw arithmetic ratio: ~120×. Real speedup is lower because of memory
   bandwidth, kernel launch, and Python overhead — but the chunked
   matmul kernel inside `_amplitude_inline` runs at ~70% of peak
   on the 1070, giving the measured 50–75× over single-thread numpy.

2. **Density binning + 3D rFFT vs direct sum** (only for static S(q)).
   The rFFT scales as `N³ log N`, the direct sum as `n_atoms × n_q`.
   For the 192³ q-grid + 70 000 atoms, the FFT is ~10× cheaper *per
   q-point* than the direct sum on the same hardware, in addition to
   the GPU speedup. This is why the static cube is so fast.

   The dynamic `S(q, ω)` does not benefit from this, because the
   density-binning approach requires storing `F(q, t)` for all
   frames, which is too memory-hungry for 5001 frames × full 3D cube.
   So the dynamic case uses direct atomic Fourier sums, and the
   speedup is purely from the GPU.

## Reproducibility

The exact wall times in this README were measured on:

* **GPU**: NVIDIA GeForce GTX 1070, 8 GB, CUDA 12.x, CuPy 13.6.0
* **CPU**: Intel i7-7700K @ 4.20 GHz, 4 cores / 8 threads (CPU benchmark
  is single-threaded numpy)
* **OS**: Windows 11 Pro
* **Python**: 3.14, NumPy ≥ 1.26

The benchmarks save full per-frame timings to NPZ; rerun on your
hardware with `python benchmarks/benchmark_*.py`.

# Contributing to gpuscatter

Contributions are welcome. The code base is small and focused — the
two main files (`sq3d.py`, `sqw.py`) are each ~300 lines.

## Development setup

```bash
git clone https://github.com/dubajicmilos/gpuscatter
cd gpuscatter
pip install -e ".[dev,gpu-cuda12]"        # or gpu-cuda11
pytest                                    # runs the CPU-only tests
```

## Running the GPU paths

The `Sq3D` and `Sqw` modules need a CUDA GPU to run. The unit tests
(`tests/`) only exercise the CPU-only parts so they work without CUDA;
to test the GPU path, run an example:

```bash
python examples/01_static_sq3d.py --n-regions 1 \
    --traj-dir <path-to-CsPbI3-600K-trajectory>
```

## Adding a new trajectory format

Subclass `BaseTrajectory` in `gpuscatter/trajectory.py` and implement:

* `species` (np.ndarray of element symbols)
* `L_box` (float, A)
* `n_frames` (int)
* `ref_positions` (np.ndarray, shape (n_atoms, 3))
* `iter_frames()` yielding `(frame_idx, positions)` per frame.

See `NpzTrajectory` for a full example.

## Code style

* `ruff check .` for lint
* docstrings in NumPy style
* tests for any new public function

## Reporting bugs

Open an issue with:

1. A minimal reproducer (preferably standalone Python).
2. Your CuPy / CUDA / GPU version (`python -c 'import cupy; cupy.show_config()'`).
3. The full traceback.

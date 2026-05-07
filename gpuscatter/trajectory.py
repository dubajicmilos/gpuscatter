"""Trajectory loaders that yield position frames one at a time.

A trajectory is anything that exposes:

* ``species`` : ``np.ndarray`` of element symbols, shape ``(n_atoms,)``.
* ``L_box`` : float, simulation box length (cubic) in angstroms.
* ``n_frames`` : total number of frames.
* ``iter_frames()`` : generator yielding ``(frame_idx, positions)`` where
  ``positions`` is a ``(n_atoms, 3)`` ``float32`` array in angstroms.
* ``ref_positions`` : ``(n_atoms, 3)`` reference positions, used for
  PBC unwrapping in dynamic computations.

Two readers are provided:

* :class:`NpzTrajectory` — reads the multi-file NPZ format used by the
  Baldwin et al. CsPbI3 trajectories and other ``ase``-style dumps.
* :class:`SingleNpzTrajectory` — single-NPZ convenience for testing.

You can subclass :class:`BaseTrajectory` to support other formats.
"""
from __future__ import annotations
from pathlib import Path
from typing import Iterable, Iterator
import numpy as np


# Atomic-number → symbol map (extend as needed).
ATOMIC_NUMBER_TO_SYMBOL: dict[int, str] = {
    1: 'H',  6: 'C', 7: 'N', 8: 'O', 9: 'F',
    11: 'Na', 12: 'Mg', 13: 'Al', 14: 'Si', 16: 'S', 17: 'Cl',
    19: 'K', 20: 'Ca', 22: 'Ti', 35: 'Br', 37: 'Rb',
    38: 'Sr', 53: 'I', 55: 'Cs', 56: 'Ba', 82: 'Pb', 83: 'Bi',
}


class BaseTrajectory:
    """Abstract trajectory reader. Subclass and implement ``iter_frames``."""

    species: np.ndarray
    L_box: float
    n_frames: int
    ref_positions: np.ndarray

    def iter_frames(self) -> Iterator[tuple[int, np.ndarray]]:
        raise NotImplementedError

    @property
    def n_atoms(self) -> int:
        return self.species.shape[0]

    @property
    def species_indices(self) -> dict[str, np.ndarray]:
        """Indices into ``species`` for each unique element."""
        return {sp: np.where(self.species == sp)[0]
                for sp in sorted(set(self.species.tolist()))}


class NpzTrajectory(BaseTrajectory):
    """Reader for trajectories split across multiple NPZ files.

    Each file is expected to contain:

    * ``positions`` : ``(F_i, n_atoms, 3)`` float, angstroms.
    * ``numbers`` : ``(F_i, n_atoms)`` int (atomic numbers).
    * ``cells`` : ``(F_i, 3, 3)`` float, angstroms.

    Frames at file boundaries are deduplicated: the last frame of file i
    is assumed to equal the first frame of file i+1, so file 0 yields
    frames ``[0, F_0)``, file 1 yields ``[1, F_1)``, etc. This matches
    the layout of the Baldwin et al. (2024) CsPbI3 trajectories.

    Parameters
    ----------
    file_paths
        Ordered list of NPZ file paths to read.
    skip_first_in_continuation
        If True (default), skip the first frame of every file *after*
        the first to deduplicate boundary frames.
    """

    def __init__(self, file_paths: Iterable[Path],
                 skip_first_in_continuation: bool = True):
        self.file_paths = [Path(p) for p in file_paths]
        if not self.file_paths:
            raise ValueError('No files supplied.')
        self.skip_first = skip_first_in_continuation

        d0 = np.load(self.file_paths[0])
        numbers = np.asarray(d0['numbers'])
        if numbers.ndim == 2:
            numbers = numbers[0]
        self.species = np.array(
            [ATOMIC_NUMBER_TO_SYMBOL[int(z)] for z in numbers]
        )

        cells = np.asarray(d0['cells'])
        if cells.ndim == 3:
            self.L_box = float(cells[0, 0, 0])
        else:
            self.L_box = float(cells[0, 0])

        self.ref_positions = np.asarray(d0['positions'][0],
                                        dtype=np.float32)

        # count frames per file (subtract 1 for continuation files)
        self._frame_counts = []
        for i, p in enumerate(self.file_paths):
            d = np.load(p, mmap_mode='r')
            f_i = d['positions'].shape[0]
            if i > 0 and self.skip_first:
                f_i -= 1
            self._frame_counts.append(f_i)
        self.n_frames = sum(self._frame_counts)

    def iter_frames(self) -> Iterator[tuple[int, np.ndarray]]:
        global_idx = 0
        for i, p in enumerate(self.file_paths):
            d = np.load(p)
            positions = d['positions']
            start = 1 if (i > 0 and self.skip_first) else 0
            stop = positions.shape[0]
            for fi in range(start, stop):
                yield global_idx, positions[fi].astype(np.float32)
                global_idx += 1
            del positions, d


class SingleNpzTrajectory(BaseTrajectory):
    """Reader for a single NPZ file with the Baldwin layout."""

    def __init__(self, file_path: Path):
        d = np.load(file_path)
        numbers = np.asarray(d['numbers'])
        if numbers.ndim == 2:
            numbers = numbers[0]
        self.species = np.array(
            [ATOMIC_NUMBER_TO_SYMBOL[int(z)] for z in numbers]
        )
        cells = np.asarray(d['cells'])
        self.L_box = float(cells[0, 0, 0]) if cells.ndim == 3 else float(cells[0, 0])
        self._positions = np.asarray(d['positions'], dtype=np.float32)
        self.ref_positions = self._positions[0]
        self.n_frames = self._positions.shape[0]

    def iter_frames(self) -> Iterator[tuple[int, np.ndarray]]:
        for i, p in enumerate(self._positions):
            yield i, p


def unwrap_positions(positions: np.ndarray, ref: np.ndarray,
                     L_box: float) -> np.ndarray:
    """Unwrap a frame's positions to a reference using PBC.

    Atoms that have wrapped through the periodic boundary are mapped back
    to the side closest to the reference. For nearly-orthorhombic boxes.
    """
    diff = positions - ref
    diff -= np.round(diff / L_box) * L_box
    return ref + diff

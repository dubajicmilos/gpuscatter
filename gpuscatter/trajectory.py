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
import warnings
import numpy as np


# Atomic-number → symbol map (extend as needed).
ATOMIC_NUMBER_TO_SYMBOL: dict[int, str] = {
    1: 'H',   2: 'He',  3: 'Li',  4: 'Be',  5: 'B',   6: 'C',   7: 'N',
    8: 'O',   9: 'F',  10: 'Ne', 11: 'Na', 12: 'Mg', 13: 'Al', 14: 'Si',
    15: 'P', 16: 'S',  17: 'Cl', 18: 'Ar', 19: 'K',  20: 'Ca', 21: 'Sc',
    22: 'Ti',23: 'V',  24: 'Cr', 25: 'Mn', 26: 'Fe', 27: 'Co', 28: 'Ni',
    29: 'Cu',30: 'Zn', 31: 'Ga', 32: 'Ge', 33: 'As', 34: 'Se', 35: 'Br',
    36: 'Kr',37: 'Rb', 38: 'Sr', 39: 'Y',  40: 'Zr', 41: 'Nb', 42: 'Mo',
    44: 'Ru',45: 'Rh', 46: 'Pd', 47: 'Ag', 48: 'Cd', 49: 'In', 50: 'Sn',
    51: 'Sb',52: 'Te', 53: 'I',  54: 'Xe', 55: 'Cs', 56: 'Ba', 57: 'La',
    72: 'Hf',73: 'Ta', 74: 'W',  78: 'Pt', 79: 'Au', 80: 'Hg', 81: 'Tl',
    82: 'Pb',83: 'Bi',
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

class LammpsDumpTrajectory(BaseTrajectory):
    """Reader for LAMMPS custom dump files with 'element x y z' columns.

    Expects dump sections in the format::

        ITEM: TIMESTEP
        <step>
        ITEM: NUMBER OF ATOMS
        <n>
        ITEM: BOX BOUNDS [xy xz yz] pp pp pp
        <xlo_bound> <xhi_bound> [<xy>]
        <ylo_bound> <yhi_bound> [<xz>]
        <zlo_bound> <zhi_bound> [<yz>]
        ITEM: ATOMS id element x y z
        <id> <el> <x> <y> <z>
        ...

    Orthogonal and triclinic (xy xz yz) boxes are both supported.
    ``L_box`` is set to the x-span of the first frame (``xhi - xlo``),
    consistent with the cubic assumption used by ``NpzTrajectory``.
    Positions are stored in angstroms and assumed to be unwrapped
    (or wrapped — ``ref_positions`` is taken from frame 0).

    Parameters
    ----------
    file_path
        Path to a single LAMMPS dump file (may contain multiple timesteps).
    """

    TILT_WARN  = 0.05  # scalar L_box approximation becoming inaccurate
    TILT_ERROR = 0.5   # LAMMPS minimum-image convention broken

    def __init__(self, file_path: Path):
        self.file_path = Path(file_path)
        self._frames: list[np.ndarray] = []
        self._parse()

    def _check_tilt(self, row0: list[str], row1: list[str], row2: list[str]):
        """Warn or raise if triclinic tilt factors are too large.

        Tilt ratios are normalised to box length (e.g. |xy| / lx).
        Ratios above TILT_WARN indicate the scalar L_box PBC unwrapping
        is becoming inaccurate. Ratios above TILT_ERROR mean the LAMMPS
        minimum-image convention itself is violated and the trajectory is
        likely corrupt.
        """
        xlo, xhi = float(row0[0]), float(row0[1])
        ylo, yhi = float(row1[0]), float(row1[1])
        zlo, zhi = float(row2[0]), float(row2[1])

        if len(row0) < 3:
            return  # orthogonal box, nothing to check

        xy = float(row0[2])
        xz = float(row1[2])
        yz = float(row2[2])

        lx = xhi - xlo
        ly = yhi - ylo
        lz = zhi - zlo

        checks = [
            (abs(xy), lx, 'xy', 'lx'),
            (abs(xz), lx, 'xz', 'lx'),
            (abs(yz), ly, 'yz', 'ly'),
        ]

        violations_warn  = []
        violations_error = []

        for tilt, box_len, tilt_name, box_name in checks:
            ratio = tilt / box_len
            if ratio > self.TILT_ERROR:
                violations_error.append(
                    f'{tilt_name}/{box_name}={ratio:.3f} > {self.TILT_ERROR}'
                )
            elif ratio > self.TILT_WARN:
                violations_warn.append(
                    f'{tilt_name}/{box_name}={ratio:.3f} > {self.TILT_WARN}'
                )

        if violations_error:
            raise ValueError(
                f'Triclinic tilt factors exceed {self.TILT_ERROR} — LAMMPS '
                f'minimum-image convention is violated, trajectory is likely '
                f'corrupt.\n'
                f'Violations: {", ".join(violations_error)}'
            )
        if violations_warn:
            warnings.warn(
                f'Triclinic tilt factors exceed {self.TILT_WARN} of box '
                f'length — scalar L_box PBC unwrapping may be inaccurate.\n'
                f'Violations: {", ".join(violations_warn)}\n'
                f'Subclass BaseTrajectory with full 3×3 cell support to fix '
                f'this.',
                UserWarning,
                stacklevel=3,
            )

    def _parse(self):
        frames = []
        species_set = False
        n_atoms = 0

        with open(self.file_path) as f:
            lines = f.readlines()

        i = 0
        while i < len(lines):
            line = lines[i].strip()

            if line == 'ITEM: TIMESTEP':
                i += 2  # skip timestep value

            elif line == 'ITEM: NUMBER OF ATOMS':
                n_atoms = int(lines[i + 1].strip())
                i += 2

            elif line.startswith('ITEM: BOX BOUNDS'):
                is_triclinic = 'xy' in line
                row0 = lines[i + 1].split()
                row1 = lines[i + 2].split()
                row2 = lines[i + 3].split()

                xlo, xhi = float(row0[0]), float(row0[1])
                if not hasattr(self, 'L_box'):
                    self.L_box = xhi - xlo
                    if is_triclinic:
                        self._check_tilt(row0, row1, row2)
                i += 4

            elif line.startswith('ITEM: ATOMS'):
                cols   = line.split()[2:]
                el_col = cols.index('element')
                x_col  = cols.index('x')
                y_col  = cols.index('y')
                z_col  = cols.index('z')
                i += 1

                atom_lines = lines[i: i + n_atoms]
                positions = np.empty((n_atoms, 3), dtype=np.float32)
                if not species_set:
                    species = []

                for j, al in enumerate(atom_lines):
                    tok = al.split()
                    positions[j, 0] = float(tok[x_col])
                    positions[j, 1] = float(tok[y_col])
                    positions[j, 2] = float(tok[z_col])
                    if not species_set:
                        species.append(tok[el_col])

                if not species_set:
                    self.species = np.array(species)
                    species_set = True

                frames.append(positions)
                i += n_atoms

            else:
                i += 1

        self._frames = frames
        self.n_frames = len(frames)
        self.ref_positions = self._frames[0].copy()

    def iter_frames(self) -> Iterator[tuple[int, np.ndarray]]:
        for i, pos in enumerate(self._frames):
            yield i, pos


class BinaryTrajectory(BaseTrajectory):
    """Fast reader for pre-converted binary trajectory files.

    Loads a directory produced by
    :meth:`StreamingLammpsDumpTrajectory.to_binary`. Positions are
    memory-mapped so frame reads are near-instant with no text parsing.

    Parameters
    ----------
    dir_path
        Directory containing ``meta.npz`` and ``positions.npy``.
    """

    def __init__(self, dir_path: Path):
        dir_path = Path(dir_path)
        meta = np.load(dir_path / 'meta.npz', allow_pickle=True)
        self.species = meta['species']
        self.L_box = float(meta['L_box'])
        self.ref_positions = meta['ref_positions'].astype(np.float32)
        self._positions = np.load(
            dir_path / 'positions.npy', mmap_mode='r'
        )
        self.n_frames = self._positions.shape[0]

    def iter_frames(self) -> Iterator[tuple[int, np.ndarray]]:
        for i in range(self.n_frames):
            yield i, np.asarray(self._positions[i], dtype=np.float32)


class StreamingLammpsDumpTrajectory(BaseTrajectory):
    """Memory-efficient reader for large LAMMPS dump files.

    Unlike :class:`LammpsDumpTrajectory`, this never loads the whole file
    into memory. It reads one frame at a time via ``iter_frames()``,
    keeping RAM usage constant regardless of file size.

    For repeated computations on the same trajectory, call
    :meth:`to_binary` once to convert to a memory-mapped binary format,
    then use :class:`BinaryTrajectory` for ~10x faster frame reads.

    Parameters
    ----------
    file_path
        Path to a LAMMPS dump file (may contain many timesteps).
    skip_frames
        Number of leading frames to skip (e.g. 1 to discard pre-equilibrium).
    """

    def __init__(self, file_path: Path, skip_frames: int = 0):
        self.file_path = Path(file_path)
        self._skip = skip_frames
        self._init_metadata()

    def _count_lines_fast(self) -> int:
        count = 0
        with open(self.file_path, 'rb') as f:
            for chunk in iter(lambda: f.read(1 << 20), b''):
                count += chunk.count(b'\n')
        return count

    def _init_metadata(self):
        with open(self.file_path) as f:
            f.readline()  # ITEM: TIMESTEP
            f.readline()  # timestep value
            f.readline()  # ITEM: NUMBER OF ATOMS
            n_atoms = int(f.readline().strip())

            header_line = f.readline()  # ITEM: BOX BOUNDS ...
            is_triclinic = 'xy' in header_line
            row0 = f.readline().split()
            row1 = f.readline().split()
            row2 = f.readline().split()
            xlo, xhi = float(row0[0]), float(row0[1])
            self.L_box = xhi - xlo

            if is_triclinic:
                LammpsDumpTrajectory._check_tilt(
                    LammpsDumpTrajectory, row0, row1, row2
                )

            atoms_header = f.readline()  # ITEM: ATOMS ...
            cols = atoms_header.split()[2:]
            el_col = cols.index('element')
            x_col = cols.index('x')
            y_col = cols.index('y')
            z_col = cols.index('z')
            self._col_indices = (el_col, x_col, y_col, z_col)

            species = []
            ref_pos = np.empty((n_atoms, 3), dtype=np.float32)
            for j in range(n_atoms):
                tok = f.readline().split()
                species.append(tok[el_col])
                ref_pos[j, 0] = float(tok[x_col])
                ref_pos[j, 1] = float(tok[y_col])
                ref_pos[j, 2] = float(tok[z_col])

            self.species = np.array(species)
            self.ref_positions = ref_pos
            self._n_atoms = n_atoms

        total_lines = self._count_lines_fast()
        lines_per_frame = 9 + n_atoms
        total_frames = total_lines // lines_per_frame
        self.n_frames = total_frames - self._skip

    def iter_frames(self) -> Iterator[tuple[int, np.ndarray]]:
        el_col, x_col, y_col, z_col = self._col_indices
        n_atoms = self._n_atoms
        global_idx = 0
        frame_in_file = 0

        with open(self.file_path) as f:
            while True:
                line = f.readline()
                if not line:
                    break
                if not line.startswith('ITEM: TIMESTEP'):
                    continue
                f.readline()  # timestep
                f.readline()  # ITEM: NUMBER OF ATOMS
                f.readline()  # n_atoms
                f.readline()  # ITEM: BOX BOUNDS
                f.readline()  # xlo xhi
                f.readline()  # ylo yhi
                f.readline()  # zlo zhi
                f.readline()  # ITEM: ATOMS header

                if frame_in_file < self._skip:
                    for _ in range(n_atoms):
                        f.readline()
                    frame_in_file += 1
                    continue

                positions = np.empty((n_atoms, 3), dtype=np.float32)
                for j in range(n_atoms):
                    tok = f.readline().split()
                    positions[j, 0] = float(tok[x_col])
                    positions[j, 1] = float(tok[y_col])
                    positions[j, 2] = float(tok[z_col])

                yield global_idx, positions
                global_idx += 1
                frame_in_file += 1

    def to_binary(self, out_dir: Path | None = None,
                  verbose: bool = True) -> Path:
        """Convert this LAMMPS dump to a fast binary directory.

        Writes ``meta.npz`` (species, L_box, ref_positions) and
        ``positions.npy`` (float32, shape ``(n_frames, n_atoms, 3)``)
        to *out_dir*. The result can be loaded instantly via
        :class:`BinaryTrajectory`.

        Parameters
        ----------
        out_dir
            Output directory. Defaults to ``<dump_stem>_binary/`` next
            to the dump file.
        verbose
            Print progress every 500 frames.

        Returns
        -------
        Path
            The output directory.
        """
        import time as _time

        if out_dir is None:
            out_dir = self.file_path.with_name(
                self.file_path.stem + '_binary'
            )
        out_dir = Path(out_dir)
        out_dir.mkdir(parents=True, exist_ok=True)

        np.savez(
            out_dir / 'meta.npz',
            species=self.species,
            L_box=np.float64(self.L_box),
            ref_positions=self.ref_positions,
        )

        pos_path = out_dir / 'positions.npy'
        shape = (self.n_frames, self._n_atoms, 3)
        mmap = np.lib.format.open_memmap(
            str(pos_path), mode='w+', dtype=np.float32, shape=shape,
        )

        t0 = _time.time()
        for idx, pos in self.iter_frames():
            mmap[idx] = pos
            if verbose and ((idx + 1) % 500 == 0
                            or idx + 1 == self.n_frames):
                elapsed = _time.time() - t0
                rate = (idx + 1) / max(elapsed, 1e-9)
                eta = (self.n_frames - idx - 1) / max(rate, 1e-9)
                size_gb = pos_path.stat().st_size / 1e9
                print(f'[to_binary] frame {idx+1}/{self.n_frames}, '
                      f'{elapsed:.0f}s, eta {eta:.0f}s, '
                      f'{size_gb:.1f} GB written', flush=True)

        mmap.flush()
        del mmap

        if verbose:
            total = _time.time() - t0
            size_gb = pos_path.stat().st_size / 1e9
            print(f'[to_binary] done in {total:.0f}s, '
                  f'{size_gb:.1f} GB -> {out_dir}')

        return out_dir


def unwrap_positions(positions: np.ndarray, ref: np.ndarray,
                     L_box: float) -> np.ndarray:
    """Unwrap a frame's positions to a reference using PBC.

    Atoms that have wrapped through the periodic boundary are mapped back
    to the side closest to the reference. For nearly-orthorhombic boxes.
    """
    diff = positions - ref
    diff -= np.round(diff / L_box) * L_box
    return ref + diff

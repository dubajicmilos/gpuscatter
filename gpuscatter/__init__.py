"""gpuscatter — GPU-accelerated diffuse scattering, S(q, omega), and
3D-DeltaPDF from MD trajectories.

Top-level imports for the most common workflows:

>>> from gpuscatter import (
...     NpzTrajectory,
...     Sq3D, Sq3DConfig,
...     Sqw, SqwConfig, make_qgrid_HK_plane, make_qgrid_BZ,
...     compute_delta_pdf,
...     DispersionProjection, HIGH_SYMMETRY_POINTS_CUBIC,
... )

See ``examples/`` for runnable scripts using the CsPbI3 600 K demo data.
"""
from .form_factors import f_xray, f_neutron, CROMER_MANN, B_NEUTRON
from .trajectory import (
    BaseTrajectory, NpzTrajectory, SingleNpzTrajectory,
    unwrap_positions, ATOMIC_NUMBER_TO_SYMBOL, LammpsDumpTrajectory
)
from .sq3d import Sq3D, Sq3DConfig, Sq3DResult
from .sqw import (
    Sqw, SqwConfig, SqwResult,
    make_qgrid_HK_plane, make_qgrid_BZ,
)
from .delta_pdf import compute_delta_pdf, DeltaPDFResult
from .dispersion import (
    DispersionProjection, project_dispersion,
    HIGH_SYMMETRY_POINTS_CUBIC, make_path_indices,
)

__version__ = '0.1.0'

__all__ = [
    'f_xray', 'f_neutron', 'CROMER_MANN', 'B_NEUTRON',
    'BaseTrajectory', 'NpzTrajectory', 'SingleNpzTrajectory',
    'unwrap_positions', 'ATOMIC_NUMBER_TO_SYMBOL',
    'Sq3D', 'Sq3DConfig', 'Sq3DResult',
    'Sqw', 'SqwConfig', 'SqwResult',
    'make_qgrid_HK_plane', 'make_qgrid_BZ',
    'compute_delta_pdf', 'DeltaPDFResult',
    'DispersionProjection', 'project_dispersion',
    'HIGH_SYMMETRY_POINTS_CUBIC', 'make_path_indices',
    '__version__',
]

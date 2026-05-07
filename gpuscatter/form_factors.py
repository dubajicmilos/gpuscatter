"""X-ray and neutron atomic form factors.

Cromer-Mann parameterisation of the X-ray form factor f0(q) for neutral
atoms; coefficients from International Tables for Crystallography Vol. C,
Table 6.1.1.4. Neutron coherent scattering lengths from the NIST table.

For X-rays:

    f0(q) = sum_{i=1..4} a_i exp(-b_i s^2) + c       with s = q / (4 pi)

    q is the magnitude of the momentum transfer in 1/A.
"""
from __future__ import annotations
import numpy as np


# (a1, b1, a2, b2, a3, b3, a4, b4, c) for each species.
CROMER_MANN: dict[str, tuple] = {
    'H':  (0.493002, 10.5109,  0.322912, 26.1257,  0.140191,  3.14236, 0.040810, 57.7997,  0.003038),
    'C':  (2.31000,  20.8439,  1.02000,  10.2075,  1.58860,   0.568700, 0.865000, 51.6512,  0.215600),
    'N':  (12.2126,  0.005700, 3.13220,  9.89330,  2.01250,  28.9975,  1.16630,  0.582600, -11.5290),
    'O':  (3.04850,  13.2771,  2.28680,  5.70110,  1.54630,   0.323900, 0.867000, 32.9089,  0.250800),
    'F':  (3.53920,  10.2825,  2.64120,  4.29440,  1.51700,   0.261500, 1.02430,  26.1476,  0.277600),
    'Na': (4.76260,  3.28500,  3.17360,  8.84220,  1.26740,   0.313600, 1.11280,  129.424,  0.676000),
    'Mg': (5.42040,  2.82750,  2.17350,  79.2611,  1.22690,   0.380800, 2.30730,  7.19370,  0.858400),
    'Al': (6.42020,  3.03870,  1.90020,  0.742600, 1.59360,  31.5472,  1.96460,  85.0886,  1.11510),
    'Si': (6.29150,  2.43860,  3.03530,  32.3337,  1.98910,   0.678500, 1.54100,  81.6937,  1.14070),
    'S':  (6.90530,  1.46790,  5.20340,  22.2151,  1.43790,   0.253600, 1.58630,  56.1720,  0.866900),
    'Cl': (11.4604,  0.0104,   7.1964,   1.1662,   6.2556,   18.5194,  1.6455,   47.7784,  -9.5574),
    'K':  (8.21860,  12.7949,  7.43980,  0.774800, 1.05190, 213.187,   0.865900, 41.6841,  1.42280),
    'Ca': (8.62660,  10.4421,  7.38730,  0.659900, 1.58990,  85.7484,  1.02110, 178.437,   1.37510),
    'Ti': (9.75950,  7.85080,  7.35580,  0.500000, 1.69910,  35.6338,  1.90210, 116.105,   1.28070),
    'Sr': (17.5663,  1.55640, 9.81840,  14.0988,   5.42200,   0.166400, 2.66940, 132.376,   2.50640),
    'Br': (17.1789,  2.1723,   5.2358,  16.5796,   5.6377,    0.2609,  3.9851,   41.4328,   2.9557),
    'Rb': (17.1784,  1.78880,  9.64350, 17.3151,   5.13990,   0.274800, 1.52920, 164.934,   3.48730),
    'I':  (20.1472,  4.3470,  18.9949,  0.3814,   7.5138,   27.7660,  2.2735,   66.8776,   4.0712),
    'Cs': (20.3892,  3.5690,  19.1062,  0.3107,  10.6620,   24.3879,  1.4953,  213.9040,   3.3352),
    'Ba': (20.3361,  3.21600, 19.2970,  0.275600, 10.8880,  20.2073,  2.69590, 167.202,    2.77310),
    'Pb': (31.0617,  0.6902,  13.0637,  2.3576,  18.4420,    8.6180,  5.9696,  47.2579,   13.4118),
    'Bi': (33.3689,  0.704000, 12.9510,  2.92380, 16.5877,    8.79370, 6.46920, 48.0093,   13.5782),
}


# Neutron coherent scattering lengths (fm) — for completeness.
B_NEUTRON: dict[str, float] = {
    'H':  -3.7406, 'D':   6.671, 'C':   6.6460, 'N':   9.36,
    'O':   5.803,  'F':   5.654, 'Na':  3.63,   'Mg':  5.375,
    'Al':  3.449,  'Si':  4.1491, 'S':   2.847,  'Cl':  9.577,
    'K':   3.67,   'Ca':  4.70,  'Ti': -3.438,  'Sr':  7.02,
    'Br':  6.795,  'Rb':  7.09,  'I':   5.28,   'Cs':  5.42,
    'Ba':  5.07,   'Pb':  9.405,
}


def f_xray(q_norm: np.ndarray, species: str) -> np.ndarray:
    """X-ray atomic form factor f0(q) (in units of electrons).

    Parameters
    ----------
    q_norm
        Magnitude of q-vectors, in inverse angstroms (shape arbitrary).
    species
        Element symbol, e.g. 'Pb', 'Cs', 'I'.

    Returns
    -------
    np.ndarray
        f0(q), same shape as ``q_norm``.

    Raises
    ------
    KeyError
        If ``species`` is not in :data:`CROMER_MANN`.
    """
    if species not in CROMER_MANN:
        raise KeyError(
            f'No Cromer-Mann coefficients for {species!r}. '
            f'Add them to CROMER_MANN in form_factors.py.'
        )
    s2 = (np.asarray(q_norm) / (4.0 * np.pi)) ** 2
    a1, b1, a2, b2, a3, b3, a4, b4, c = CROMER_MANN[species]
    return (a1 * np.exp(-b1 * s2) +
            a2 * np.exp(-b2 * s2) +
            a3 * np.exp(-b3 * s2) +
            a4 * np.exp(-b4 * s2) + c)


def f_neutron(species: str) -> float:
    """Neutron coherent scattering length b in fm (q-independent)."""
    if species not in B_NEUTRON:
        raise KeyError(f'No neutron length for {species!r}.')
    return B_NEUTRON[species]

"""Tests for form_factors.py."""
import numpy as np
import pytest

from gpuscatter.form_factors import f_xray, f_neutron, CROMER_MANN


def test_f_xray_at_zero_returns_atomic_number():
    """At q=0, f0 should equal Z (the number of electrons).

    Exercised across the periodic table so any wrong constant in any
    new entry is caught immediately.
    """
    z_lookup = {
        'H': 1, 'He': 2, 'Li': 3, 'Be': 4, 'B': 5, 'C': 6, 'N': 7, 'O': 8,
        'F': 9, 'Ne': 10, 'Na': 11, 'Mg': 12, 'Al': 13, 'Si': 14, 'P': 15,
        'S': 16, 'Cl': 17, 'Ar': 18, 'K': 19, 'Ca': 20, 'Sc': 21, 'Ti': 22,
        'V': 23, 'Cr': 24, 'Mn': 25, 'Fe': 26, 'Co': 27, 'Ni': 28, 'Cu': 29,
        'Zn': 30, 'Ga': 31, 'Ge': 32, 'As': 33, 'Se': 34, 'Br': 35, 'Kr': 36,
        'Rb': 37, 'Sr': 38, 'Y': 39, 'Zr': 40, 'Nb': 41, 'Mo': 42, 'Ru': 44,
        'Rh': 45, 'Pd': 46, 'Ag': 47, 'Cd': 48, 'In': 49, 'Sn': 50, 'Sb': 51,
        'Te': 52, 'I': 53, 'Xe': 54, 'Cs': 55, 'Ba': 56, 'La': 57, 'Hf': 72,
        'Ta': 73, 'W': 74, 'Pt': 78, 'Au': 79, 'Hg': 80, 'Tl': 81, 'Pb': 82,
        'Bi': 83,
    }
    for sp, z in z_lookup.items():
        if sp not in CROMER_MANN:
            continue
        f0 = f_xray(np.array([0.0]), sp)
        assert abs(float(f0[0]) - z) < 0.5, (
            f'{sp}: f(0) = {float(f0[0]):.2f} should be ~{z}'
        )


def test_f_xray_decays_with_q():
    """f0(q) should monotonically decrease with q."""
    q = np.linspace(0, 10, 50)
    for sp in ['H', 'C', 'O', 'Al', 'Ni', 'Cu', 'Mo', 'Cs', 'Pb', 'Au']:
        f = f_xray(q, sp)
        diffs = np.diff(f)
        assert np.all(diffs < 0), f'{sp} f0 not monotonically decreasing'


def test_cromer_mann_has_full_periodic_table_coverage():
    """Spot-check that the table now covers the common neutral elements."""
    must_have = ['H', 'C', 'N', 'O', 'Al', 'Si', 'P', 'S', 'Cl', 'Ti', 'Cr',
                 'Fe', 'Ni', 'Cu', 'Zn', 'Br', 'I', 'Cs', 'Pb', 'Au']
    missing = [sp for sp in must_have if sp not in CROMER_MANN]
    assert not missing, f'Missing X-ray entries: {missing}'


def test_f_xray_unknown_species_raises():
    with pytest.raises(KeyError):
        f_xray(np.array([1.0]), 'Xx')


def test_f_neutron_returns_float():
    assert isinstance(f_neutron('Pb'), float)


def test_f_neutron_known_values():
    """Spot-check a few neutron lengths against the NIST reference table.

    Negative b for H is a key feature exploited by isotope substitution
    (H<->D contrast).
    """
    # https://www.ncnr.nist.gov/resources/n-lengths/
    expected_fm = {
        'H': -3.7406,
        'D':  6.671,
        'C':  6.6460,
        'N':  9.36,
        'Pb': 9.405,
        'Cs': 5.42,
    }
    for sp, b_expected in expected_fm.items():
        assert abs(f_neutron(sp) - b_expected) < 0.01, (
            f'{sp}: f_neutron = {f_neutron(sp)} vs expected {b_expected}'
        )


def test_f_neutron_unknown_species_raises():
    import pytest
    with pytest.raises(KeyError):
        f_neutron('Xx')


def test_f_xray_shape_preserved():
    q = np.random.rand(7, 5).astype(np.float32)
    f = f_xray(q, 'Pb')
    assert f.shape == q.shape

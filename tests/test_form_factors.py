"""Tests for form_factors.py."""
import numpy as np
import pytest

from gpuscatter.form_factors import f_xray, f_neutron, CROMER_MANN


def test_f_xray_at_zero_returns_atomic_number():
    """At q=0, f0 should equal Z (the number of electrons)."""
    z_lookup = {'H': 1, 'C': 6, 'N': 7, 'O': 8, 'Cl': 17,
                'Br': 35, 'I': 53, 'Cs': 55, 'Pb': 82}
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
    for sp in ['Pb', 'I', 'Cs']:
        f = f_xray(q, sp)
        diffs = np.diff(f)
        assert np.all(diffs < 0), f'{sp} f0 not monotonically decreasing'


def test_f_xray_unknown_species_raises():
    with pytest.raises(KeyError):
        f_xray(np.array([1.0]), 'Xx')


def test_f_neutron_returns_float():
    assert isinstance(f_neutron('Pb'), float)


def test_f_xray_shape_preserved():
    q = np.random.rand(7, 5).astype(np.float32)
    f = f_xray(q, 'Pb')
    assert f.shape == q.shape

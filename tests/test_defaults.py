"""Invariants over the chemistry tables in defaults.py."""
from defaults import (
    BINDING_ATOMS,
    ION_CHARGE,
    ION_SPIN,
    MIN_COORD,
    USES_BACKBONE_O,
)


def test_ion_keys_consistent_across_tables():
    ions = set(BINDING_ATOMS)
    assert ions == set(MIN_COORD)
    assert ions == set(ION_CHARGE)
    assert ions == set(ION_SPIN)


def test_ion_codes_uppercase():
    for ion in BINDING_ATOMS:
        assert ion.isupper()
        assert 1 <= len(ion) <= 2


def test_uses_backbone_o_subset_of_ions():
    assert USES_BACKBONE_O <= set(BINDING_ATOMS)


def test_binding_atoms_are_nonempty_str_lists():
    for ion, residues in BINDING_ATOMS.items():
        assert residues, ion
        for resname, atoms in residues.items():
            assert resname.isupper() and len(resname) == 3
            assert atoms, (ion, resname)
            assert all(isinstance(a, str) and a for a in atoms)


def test_charges_and_spins_are_positive_ints():
    for ion, charge in ION_CHARGE.items():
        assert isinstance(charge, int) and charge > 0, ion
    for ion, spin in ION_SPIN.items():
        assert isinstance(spin, int) and spin >= 1, ion


def test_min_coord_is_positive():
    for ion, n in MIN_COORD.items():
        assert isinstance(n, int) and n > 0, ion


def test_nd_is_lanthanide_with_backbone_o():
    # ND is the only ion the pipeline cares about that's a lanthanide;
    # geomopt and ion placement both rely on these specific values.
    assert 'ND' in USES_BACKBONE_O
    assert ION_CHARGE['ND'] == 3

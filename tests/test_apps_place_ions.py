"""End-to-end test of apps.place_ions on a synthetic mini-PDB.

We construct PDBs in-memory and let Bio.PDB parse them so atom masses and
elements get populated naturally — no mocking of Bio.PDB or sklearn.
"""
from pathlib import Path

import pytest

pytest.importorskip('Bio')
pytest.importorskip('sklearn')
try:
    import apps
except Exception as exc:  # pragma: no cover - environment-dependent
    pytest.skip(
        f'apps import failed ({type(exc).__name__}: {exc})',
        allow_module_level=True,
    )

from Bio.PDB import PDBParser  # noqa: F401  (used implicitly by apps.place_ions)


def _atom_line(serial, name, resname, chain, resid, xyz, element):
    """Emit a fixed-width PDB ATOM line."""
    # cols 13-16 are the atom name; 1-3 char names get a leading space
    if len(name) < 4:
        atom_field = ' ' + name.ljust(3)
    else:
        atom_field = name
    x, y, z = xyz
    return (
        f'ATOM  {serial:5d} {atom_field} {resname:>3s} {chain:1s}{resid:4d}    '
        f'{x:8.3f}{y:8.3f}{z:8.3f}'
        f'  1.00  0.00          {element:>2s}'
    )


def _write_pdb(path, residues):
    """residues = [(resname, resid, [(atom_name, (x,y,z), element), ...]), ...]"""
    serial = 1
    lines = []
    for resname, resid, atoms in residues:
        for name, xyz, element in atoms:
            lines.append(_atom_line(serial, name, resname, 'A', resid, xyz, element))
            serial += 1
    lines.append('END')
    Path(path).write_text('\n'.join(lines) + '\n')


def _asp(resid, com):
    """Asp residue with side-chain OD1/OD2 placed near `com`."""
    cx, cy, cz = com
    return ('ASP', resid, [
        ('N',   (cx - 4.0, cy, cz),       'N'),
        ('CA',  (cx - 3.0, cy, cz),       'C'),
        ('C',   (cx - 2.5, cy + 1.0, cz), 'C'),
        ('O',   (cx - 2.5, cy + 2.0, cz), 'O'),
        ('CB',  (cx - 2.0, cy, cz),       'C'),
        ('CG',  (cx - 1.0, cy, cz),       'C'),
        ('OD1', (cx - 0.3, cy - 0.5, cz), 'O'),
        ('OD2', (cx - 0.3, cy + 0.5, cz), 'O'),
    ])


def test_place_ions_zn_seats_ion_when_min_coord_met(tmp_path):
    pdb = tmp_path / 'in.pdb'
    out_pdb = tmp_path / 'out.pdb'

    # Two ASP residues with side chains both centered near (10, 10, 10);
    # ZN MIN_COORD = 3, ASP contributes 2 atoms each -> 4 total, satisfied.
    _write_pdb(pdb, [
        _asp(1, (10.0, 10.0, 10.0)),
        _asp(2, (11.0, 10.0, 10.0)),
    ])

    placements = apps.place_ions(
        pdb_file=pdb,
        output_pdb=out_pdb,
        binding_residues=[1, 2],
        ion='ZN',
    )

    assert len(placements) == 1
    new_resid, contributing = next(iter(placements.items()))
    assert contributing == [1, 2]
    # next_resid is max(existing) + 1; existing residues are 1 and 2
    assert new_resid == 3

    # The output PDB should contain a ZN HETATM
    text = out_pdb.read_text()
    assert 'ZN' in text
    assert 'HETATM' in text


def test_place_ions_returns_empty_when_below_min_coord(tmp_path):
    pdb = tmp_path / 'in.pdb'
    out_pdb = tmp_path / 'out.pdb'

    # Single ASP -> 2 atoms, below ZN MIN_COORD=3.
    _write_pdb(pdb, [_asp(1, (5.0, 5.0, 5.0))])

    placements = apps.place_ions(
        pdb_file=pdb, output_pdb=out_pdb,
        binding_residues=[1], ion='ZN',
    )
    assert placements == {}


def test_place_ions_skips_residues_not_in_predictions(tmp_path):
    pdb = tmp_path / 'in.pdb'
    out_pdb = tmp_path / 'out.pdb'

    _write_pdb(pdb, [
        _asp(1, (10.0, 10.0, 10.0)),
        _asp(2, (11.0, 10.0, 10.0)),
        _asp(3, (10.5, 10.5, 10.0)),
    ])

    # binding_residues only flags residue 99 (does not exist) -> nothing seated
    placements = apps.place_ions(
        pdb_file=pdb, output_pdb=out_pdb,
        binding_residues=[99], ion='ZN',
    )
    assert placements == {}


def test_place_ions_clusters_by_distance_threshold(tmp_path):
    pdb = tmp_path / 'in.pdb'
    out_pdb = tmp_path / 'out.pdb'

    # Two pairs of ASPs, 50 Å apart -> two distinct clusters of 4 atoms each.
    _write_pdb(pdb, [
        _asp(1, (0.0, 0.0, 0.0)),
        _asp(2, (1.0, 0.0, 0.0)),
        _asp(3, (50.0, 0.0, 0.0)),
        _asp(4, (51.0, 0.0, 0.0)),
    ])

    placements = apps.place_ions(
        pdb_file=pdb, output_pdb=out_pdb,
        binding_residues=[1, 2, 3, 4], ion='ZN',
        cluster_threshold=7.0,
    )
    assert len(placements) == 2

    # The two clusters should partition residues {1,2} and {3,4}
    groups = sorted(sorted(g) for g in placements.values())
    assert groups == [[1, 2], [3, 4]]


def test_place_ions_for_ca_includes_backbone_o(tmp_path):
    """CA is in USES_BACKBONE_O — backbone O contributes to coordination.

    With three ASN residues (which contribute OD1 only for CA) plus their
    backbone Os, we should reach CA's MIN_COORD=5 (three OD1 + three O = 6).
    Without the backbone O, three OD1 would not meet the threshold... wait,
    3 < 5 only without backbone O; 6 >= 5 with it.
    """
    def _asn(resid, com):
        cx, cy, cz = com
        return ('ASN', resid, [
            ('N',   (cx - 4.0, cy, cz),       'N'),
            ('CA',  (cx - 3.0, cy, cz),       'C'),
            ('C',   (cx - 2.5, cy + 1.0, cz), 'C'),
            ('O',   (cx - 2.5, cy + 2.0, cz), 'O'),
            ('CB',  (cx - 2.0, cy, cz),       'C'),
            ('CG',  (cx - 1.0, cy, cz),       'C'),
            ('OD1', (cx - 0.3, cy, cz),       'O'),
            ('ND2', (cx - 0.3, cy + 1.0, cz), 'N'),
        ])

    pdb = tmp_path / 'in.pdb'
    out_pdb = tmp_path / 'out.pdb'
    _write_pdb(pdb, [
        _asn(1, (10.0, 10.0, 10.0)),
        _asn(2, (11.0, 10.0, 10.0)),
        _asn(3, (10.5, 10.5, 10.0)),
    ])

    placements = apps.place_ions(
        pdb_file=pdb, output_pdb=out_pdb,
        binding_residues=[1, 2, 3], ion='CA',
    )
    assert len(placements) == 1

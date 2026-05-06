"""Test extract_clusters using a tiny stub for the OpenMM Simulator surface.

The Simulator object only needs to expose `simulation.context.getState(...)` and
`simulation.topology.atoms()`, so we build a minimal SimpleNamespace stub of
exactly that shape rather than spinning up real OpenMM.
"""
import json
from types import SimpleNamespace

import numpy as np
import pytest

try:
    import apps
except Exception as exc:  # pragma: no cover - environment-dependent
    pytest.skip(
        f'apps import failed ({type(exc).__name__}: {exc})',
        allow_module_level=True,
    )


class StubResidue:
    def __init__(self, name, resid):
        self.name = name
        self.id = str(resid)


class StubAtom:
    def __init__(self, index, name, element_symbol, residue):
        self.index = index
        self.name = name
        self.element = (
            SimpleNamespace(symbol=element_symbol)
            if element_symbol is not None else None
        )
        self.residue = residue


def _make_sim(positions, atoms):
    state = SimpleNamespace(getPositions=lambda asNumpy=False: positions)
    context = SimpleNamespace(getState=lambda getPositions=False: state)
    topology = SimpleNamespace(atoms=lambda: iter(atoms))
    return SimpleNamespace(
        simulation=SimpleNamespace(context=context, topology=topology),
    )


def _build_system():
    """One Nd ion at origin, one ASP nearby, one OPC water close, one OPC far."""
    nd_res = StubResidue('Nd', 1)
    asp_res = StubResidue('ASP', 2)
    water_close = StubResidue('OPC', 3)
    water_far = StubResidue('OPC', 4)

    atoms = []
    positions = []

    # Nd ion (index 0)
    atoms.append(StubAtom(0, 'Nd', 'Nd', nd_res))
    positions.append([0.0, 0.0, 0.0])

    # ASP residue: 8 atoms at indices 1..8
    asp_atoms = [('N','N'),('CA','C'),('C','C'),('O','O'),
                 ('CB','C'),('CG','C'),('OD1','O'),('OD2','O')]
    for i, (name, el) in enumerate(asp_atoms):
        atoms.append(StubAtom(1 + i, name, el, asp_res))
        positions.append([2.0 + i * 0.1, 0.0, 0.0])

    # OPC water (close): 3 atoms at indices 9..11, O at distance 3.0 < 4.0 cutoff
    for i, (name, el) in enumerate([('O','O'),('H1','H'),('H2','H')]):
        atoms.append(StubAtom(9 + i, name, el, water_close))
        positions.append([3.0 + i * 0.1, 0.0, 0.0])

    # OPC water (far): 3 atoms at indices 12..14, distance 20.0 > 4.0 cutoff
    for i, (name, el) in enumerate([('O','O'),('H1','H'),('H2','H')]):
        atoms.append(StubAtom(12 + i, name, el, water_far))
        positions.append([20.0 + i * 0.1, 0.0, 0.0])

    return atoms, np.array(positions, dtype=float)


def test_extract_clusters_includes_close_water_and_skips_far(tmp_path):
    atoms, positions = _build_system()
    sim = _make_sim(positions, atoms)

    written = apps.extract_clusters(
        sim=sim, ion='Nd',
        ion_to_group={1: [2]},
        output_dir=tmp_path / 'clusters',
        water_cutoff=4.0,
    )

    assert len(written) == 1
    xyz = written[0]
    assert xyz.exists()
    assert xyz.name == 'cluster_1.xyz'

    js = xyz.with_suffix('.json')
    assert js.exists()

    meta = json.loads(js.read_text())
    assert meta['ion'] == 'Nd'
    assert meta['ion_resid'] == 1
    assert meta['binding_residues'] == [2]
    # 1 (Nd) + 8 (ASP) + 3 (close water) = 12
    assert meta['n_atoms'] == 12
    assert meta['ion_formal_charge'] == apps.ION_CHARGE['ND']
    assert meta['spin_multiplicity_guess'] == apps.ION_SPIN['ND']

    lines = xyz.read_text().splitlines()
    assert int(lines[0]) == 12
    # First atom emitted is the Nd ion
    first_atom = lines[2].split()
    assert first_atom[0] == 'Nd'


def test_extract_clusters_skips_waters_when_cutoff_is_zero(tmp_path):
    atoms, positions = _build_system()
    sim = _make_sim(positions, atoms)

    written = apps.extract_clusters(
        sim=sim, ion='Nd',
        ion_to_group={1: [2]},
        output_dir=tmp_path / 'clusters',
        water_cutoff=0.0,
    )

    assert len(written) == 1
    meta = json.loads(written[0].with_suffix('.json').read_text())
    # 1 (Nd) + 8 (ASP) only — no waters
    assert meta['n_atoms'] == 9


def test_extract_clusters_skips_ions_missing_from_topology(tmp_path):
    atoms, positions = _build_system()
    sim = _make_sim(positions, atoms)

    # ion_to_group references resid 99 which doesn't exist -> silently skipped
    written = apps.extract_clusters(
        sim=sim, ion='Nd',
        ion_to_group={99: [2]},
        output_dir=tmp_path / 'clusters',
        water_cutoff=4.0,
    )
    assert written == []

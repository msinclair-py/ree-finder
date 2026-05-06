"""Real PySCF DFT through apps.geomopt on a tiny H–F molecule.

We use the cheapest sensible setup — sto-3g basis, LDA functional, no
dispersion correction, grid level 1 — so the test runs in a couple of
seconds on CPU. PySCF and geometric are required; if either is missing the
test is skipped.
"""
import json

import numpy as np
import pytest

pytest.importorskip('pyscf')
pytest.importorskip('geometric')
try:
    import apps
except Exception as exc:  # pragma: no cover - environment-dependent
    pytest.skip(
        f'apps import failed ({type(exc).__name__}: {exc})',
        allow_module_level=True,
    )


def _write_hf_inputs(tmp_path):
    xyz = tmp_path / 'hf.xyz'
    xyz.write_text(
        '2\n'
        'HF molecule\n'
        'H  0.000000  0.000000  0.000000\n'
        'F  0.000000  0.000000  0.917000\n'
    )
    js = tmp_path / 'hf.json'
    js.write_text(json.dumps({
        'ion': 'F',           # not really an ion — geomopt only reads charge/spin
        'ion_resid': 0,
        'ion_formal_charge': 0,
        'spin_multiplicity_guess': 1,  # closed-shell singlet -> pyscf spin = 0
        'binding_residues': [],
        'n_atoms': 2,
        'water_cutoff_A': 0.0,
    }))
    return xyz, js


def test_geomopt_optimizes_hf(tmp_path):
    xyz, js = _write_hf_inputs(tmp_path)

    coords, e_final = apps.geomopt(
        xyz_path=xyz,
        json_path=js,
        basis='sto-3g',
        functional='svwn',     # plain LDA, no libxc surprises
        dispersion='',         # empty string -> pyscf skips dispersion
        max_steps=20,
        num_threads=1,
        verbose=0,
        max_memory=2000,
        grid_level=1,
    )

    # Two atoms in 3D
    assert coords.shape == (2, 3)

    # SCF energy should be sensibly negative for HF
    assert np.isfinite(e_final)
    assert e_final < 0.0

    # Bond length should remain near the starting 0.917 Å (LDA/sto-3g optimum
    # is ~0.96 Å — well within a generous bound)
    bond_length = float(np.linalg.norm(coords[1] - coords[0]))
    assert 0.7 < bond_length < 1.3

    out_xyz = xyz.with_name('hf_opt.xyz')
    out_json = xyz.with_name('hf_opt.json')
    assert out_xyz.exists()
    assert out_json.exists()

    # Optimized XYZ has the same atom count and element symbols
    lines = out_xyz.read_text().splitlines()
    assert int(lines[0]) == 2
    elements = [line.split()[0] for line in lines[2:4]]
    assert elements == ['H', 'F']

    meta = json.loads(out_json.read_text())
    assert meta['basis'] == 'sto-3g'
    assert meta['functional'] == 'svwn'
    assert meta['grid_level'] == 1
    assert 'energy_hartree' in meta
    assert isinstance(meta['converged'], bool)
    # Original sidecar fields should pass through
    assert meta['ion_formal_charge'] == 0
    assert meta['spin_multiplicity_guess'] == 1

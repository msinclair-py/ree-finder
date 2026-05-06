"""Test build_amber_system without a real Amber install.

We monkeypatch subprocess.run since invoking tleap/parmed requires the actual
Amber binaries — that's the only thing we mock. The script-generation logic,
box-sizing math, and ion-mask formatting are exercised for real.
"""
import subprocess

import pytest

try:
    import apps
except Exception as exc:  # pragma: no cover - environment-dependent
    pytest.skip(
        f'apps import failed ({type(exc).__name__}: {exc})',
        allow_module_level=True,
    )


def _write_minimal_pdb(path):
    # Two atoms 5 Å apart along each axis -> extent = 5
    path.write_text(
        'ATOM      1  CA  ALA A   1       0.000   0.000   0.000  1.00  0.00           C\n'
        'HETATM    2  Nd  Nd  A   2       5.000   5.000   5.000  1.00  0.00          ND\n'
        'END\n'
    )


def test_build_amber_system_emits_correct_scripts(tmp_path, monkeypatch):
    out = tmp_path / 'out'; out.mkdir()
    build = tmp_path / 'build'; build.mkdir()
    pdb = build / 'protein_with_metal.pdb'
    _write_minimal_pdb(pdb)

    fake_amber = tmp_path / 'amber'
    (fake_amber / 'bin').mkdir(parents=True)

    calls = []

    def fake_run(cmd, **kw):
        calls.append({'cmd': list(cmd), 'cwd': kw.get('cwd'), 'env': kw.get('env')})
        return subprocess.CompletedProcess(cmd, 0, stdout='', stderr='')

    monkeypatch.setattr(apps.subprocess, 'run', fake_run)

    prmtop, inpcrd, pdb_out = apps.build_amber_system(
        metal_pdb=pdb, out=out, build=build, ion='ND',
        amberhome=fake_amber, padding=10.0, salt_conc_M=0.15,
    )

    assert prmtop == out / 'system.prmtop'
    assert inpcrd == out / 'system.inpcrd'
    assert pdb_out == out / 'system.pdb'

    tleap_in = (build / 'tleap.in').read_text()
    assert 'source leaprc.protein.ff19SB' in tleap_in
    assert 'source leaprc.water.opc' in tleap_in
    assert 'loadamberparams frcmod.ionslm_1264_opc' in tleap_in
    assert f'PROT = loadpdb {pdb}' in tleap_in
    assert 'solvatebox PROT OPCBOX' in tleap_in
    # extent = 5, padding = 10 -> dim = int(5 + 20) = 25
    assert '{25 25 25}' in tleap_in
    assert f'savepdb PROT {pdb_out}' in tleap_in
    assert f'saveamberparm PROT {prmtop} {inpcrd}' in tleap_in

    parmed_in = (build / 'parmed.in').read_text()
    # Li/Merz mask uses capitalized + charge: ND -> Nd3+
    assert 'add12_6_4 @%Nd3+ watermodel OPC' in parmed_in
    assert f'loadRestrt {inpcrd}' in parmed_in

    # Two subprocess calls: tleap then parmed
    assert len(calls) == 2
    assert calls[0]['cmd'][0].endswith('tleap')
    assert calls[1]['cmd'][0].endswith('parmed')
    # Both should run with AMBERHOME pointed at the supplied install
    for call in calls:
        assert call['env']['AMBERHOME'] == str(fake_amber)


def test_build_amber_system_scales_salt_with_concentration(tmp_path, monkeypatch):
    """The number of salt ions tracks the requested concentration."""
    out = tmp_path / 'out'; out.mkdir()
    build = tmp_path / 'build'; build.mkdir()
    pdb = build / 'p.pdb'
    _write_minimal_pdb(pdb)

    monkeypatch.setattr(
        apps.subprocess, 'run',
        lambda cmd, **kw: subprocess.CompletedProcess(cmd, 0, stdout='', stderr=''),
    )

    fake_amber = tmp_path / 'amber'
    (fake_amber / 'bin').mkdir(parents=True)

    apps.build_amber_system(
        metal_pdb=pdb, out=out, build=build, ion='ND',
        amberhome=fake_amber, padding=10.0, salt_conc_M=0.30,  # 2x reference
    )

    tleap_in = (build / 'tleap.in').read_text()
    # dim=25, factor 2x at 0.15: round(25**3 * 1e-5 * 2 * 9.03) = 3
    assert 'addIonsRand PROT Na+ 3 Cl- 3' in tleap_in


def test_build_amber_system_ion_mask_for_calcium(tmp_path, monkeypatch):
    """Ion mask uses ION_CHARGE to format the parmed selector."""
    out = tmp_path / 'out'; out.mkdir()
    build = tmp_path / 'build'; build.mkdir()
    pdb = build / 'p.pdb'
    _write_minimal_pdb(pdb)

    monkeypatch.setattr(
        apps.subprocess, 'run',
        lambda cmd, **kw: subprocess.CompletedProcess(cmd, 0, stdout='', stderr=''),
    )

    fake_amber = tmp_path / 'amber'
    (fake_amber / 'bin').mkdir(parents=True)

    apps.build_amber_system(
        metal_pdb=pdb, out=out, build=build, ion='CA',
        amberhome=fake_amber, padding=10.0, salt_conc_M=0.15,
    )

    parmed_in = (build / 'parmed.in').read_text()
    assert 'add12_6_4 @%Ca2+ watermodel OPC' in parmed_in

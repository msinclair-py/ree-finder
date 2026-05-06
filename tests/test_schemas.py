"""Tests for the cluster sidecar JSON schema contract.

These run in CI's slim env: schemas.py only depends on pydantic, so unlike
the apps-level tests these don't get skipped on import.
"""
import json
from pathlib import Path

import pytest
import yaml
from pydantic import ValidationError
from schemas import (
    SUPPORTED_IONS,
    ClusterMetadata,
    ESMBindMultiModalConfig,
    ESMBindSingleConfig,
    OptimizedClusterMetadata,
    PipelineConfig,
)


def _valid_cluster_payload():
    return dict(
        ion='Nd',
        ion_resid=1,
        ion_formal_charge=3,
        spin_multiplicity_guess=4,
        binding_residues=[2, 5, 7],
        n_atoms=12,
        water_cutoff_A=4.0,
    )


def test_cluster_metadata_round_trips_through_json():
    payload = _valid_cluster_payload()
    m = ClusterMetadata(**payload)
    restored = ClusterMetadata.model_validate_json(m.model_dump_json())
    assert restored == m


def test_cluster_metadata_accepts_legacy_handwritten_json():
    """The geomopt test writes raw json; pydantic must read it back."""
    raw = json.dumps({
        'ion': 'F',
        'ion_resid': 0,
        'ion_formal_charge': 0,
        'spin_multiplicity_guess': 1,
        'binding_residues': [],
        'n_atoms': 2,
        'water_cutoff_A': 0.0,
    })
    m = ClusterMetadata.model_validate_json(raw)
    assert m.ion_formal_charge == 0
    assert m.spin_multiplicity_guess == 1
    assert m.binding_residues == []


def test_cluster_metadata_rejects_extra_fields():
    payload = _valid_cluster_payload() | {'unknown_field': 1}
    with pytest.raises(ValidationError):
        ClusterMetadata(**payload)


@pytest.mark.parametrize('field,bad_value', [
    ('spin_multiplicity_guess', 0),  # ge=1
    ('n_atoms', 0),                  # gt=0
    ('water_cutoff_A', -0.5),        # ge=0
])
def test_cluster_metadata_rejects_out_of_range(field, bad_value):
    payload = _valid_cluster_payload() | {field: bad_value}
    with pytest.raises(ValidationError):
        ClusterMetadata(**payload)


def test_optimized_metadata_extends_cluster_metadata():
    base = ClusterMetadata(**_valid_cluster_payload())
    out = OptimizedClusterMetadata(
        **base.model_dump(),
        energy_hartree=-1234.5,
        basis='def2-TZVP',
        functional='B3LYP',
        dispersion='d3bj',
        grid_level=3,
        converged=True,
    )
    data = json.loads(out.model_dump_json())
    # Input fields propagate
    for k, v in _valid_cluster_payload().items():
        assert data[k] == v
    # New fields present
    assert data['energy_hartree'] == -1234.5
    assert data['converged'] is True
    assert data['grid_level'] == 3


def test_optimized_metadata_requires_dft_fields():
    base = ClusterMetadata(**_valid_cluster_payload())
    with pytest.raises(ValidationError):
        OptimizedClusterMetadata(**base.model_dump())  # missing DFT fields


def _multimodal_kwargs(**overrides):
    base = dict(
        feature_dim_1=8, feature_dim_2=4,
        hidden_dim_1=6, hidden_dim_2=6, hidden_dim=4,
        noise_level=0.0, dropout=0.0,
    )
    return base | overrides


def test_multimodal_config_attribute_access_matches_simplenamespace():
    """model.py reads `conf.x` directly — pydantic must satisfy the same shape."""
    cfg = ESMBindMultiModalConfig(**_multimodal_kwargs())
    assert cfg.feature_dim_1 == 8
    assert cfg.feature_dim_2 == 4
    assert cfg.hidden_dim == 4
    # ESMBindBase falls back to 1 when feature_dim isn't declared
    assert getattr(cfg, 'feature_dim', 1) == 1


def test_multimodal_config_is_frozen():
    cfg = ESMBindMultiModalConfig(**_multimodal_kwargs())
    with pytest.raises(ValidationError):
        cfg.dropout = 0.5  # type: ignore[misc]


def test_multimodal_config_rejects_typo():
    with pytest.raises(ValidationError):
        ESMBindMultiModalConfig(**_multimodal_kwargs(droput=0.5))


@pytest.mark.parametrize('override', [
    {'feature_dim_1': 0},     # gt=0
    {'hidden_dim': 0},        # gt=0
    {'dropout': 1.5},         # le=1
    {'dropout': -0.1},        # ge=0
    {'noise_level': -0.5},    # ge=0
])
def test_multimodal_config_rejects_out_of_range(override):
    with pytest.raises(ValidationError):
        ESMBindMultiModalConfig(**_multimodal_kwargs(**override))


def test_single_config_carries_independent_feature_dim():
    cfg = ESMBindSingleConfig(
        feature_dim=8, hidden_dim_1=6, hidden_dim=4,
        noise_level=0.0, dropout=0.0,
    )
    assert cfg.feature_dim == 8
    assert cfg.hidden_dim_1 == 6


def test_model_classes_consume_pydantic_configs():
    """ESMBindBase / Single / MultiModal must accept the typed configs."""
    torch = pytest.importorskip('torch')
    from model import ESMBindMultiModal, ESMBindSingle

    mm_cfg = ESMBindMultiModalConfig(**_multimodal_kwargs())
    mm = ESMBindMultiModal(mm_cfg).eval()
    with torch.no_grad():
        out = mm(torch.randn(2, 5, 8), torch.randn(2, 5, 4), 'CA', training=False)
    assert out.shape == (2, 5)

    s_cfg = ESMBindSingleConfig(
        feature_dim=8, hidden_dim_1=6, hidden_dim=4,
        noise_level=0.0, dropout=0.0,
    )
    s = ESMBindSingle(s_cfg).eval()
    with torch.no_grad():
        out = s(torch.randn(2, 5, 8), 'CA', training=False)
    assert out.shape == (2, 5)


def test_pipeline_config_defaults_only_require_amberhome():
    cfg = PipelineConfig(amberhome=Path('/opt/amber'))
    assert cfg.amberhome == Path('/opt/amber')
    assert cfg.ensemble_path == Path('esmbind_weights')
    assert cfg.basis == 'def2-TZVP'
    assert cfg.functional == 'B3LYP'
    assert cfg.dispersion == 'd3bj'
    assert cfg.water_cutoff == 4.0
    assert cfg.ions == ['CA', 'MG', 'FE', 'ND']
    assert cfg.device == 'cuda'


def test_pipeline_config_run_dir_is_timestamped_under_outputs():
    cfg = PipelineConfig(amberhome=Path('/opt/amber'))
    assert cfg.run_dir.parent == Path('outputs')
    # Pattern: YYYYMMDD_HHMMSS — 15 chars including underscore
    stem = cfg.run_dir.name
    assert len(stem) == 15 and stem[8] == '_'
    assert stem.replace('_', '').isdigit()


def test_pipeline_config_run_dir_factory_runs_per_instance():
    """Default factory must not share state across instances."""
    a = PipelineConfig(amberhome=Path('/opt/amber'))
    b = PipelineConfig(
        amberhome=Path('/opt/amber'),
        run_dir=Path('/tmp/explicit'),
    )
    assert b.run_dir == Path('/tmp/explicit')
    # `a` got the default, not `b`'s explicit path
    assert a.run_dir != b.run_dir


def test_pipeline_config_is_frozen():
    cfg = PipelineConfig(amberhome=Path('/opt/amber'))
    with pytest.raises(ValidationError):
        cfg.basis = 'cc-pVDZ'  # type: ignore[misc]


def test_pipeline_config_rejects_unknown_ion():
    with pytest.raises(ValidationError, match='unsupported ions'):
        PipelineConfig(amberhome=Path('/opt/amber'), ions=['CA', 'PB'])


def test_pipeline_config_rejects_empty_ions():
    with pytest.raises(ValidationError, match='non-empty'):
        PipelineConfig(amberhome=Path('/opt/amber'), ions=[])


def test_pipeline_config_supported_ions_covers_all_defaults():
    """The default ion list must itself satisfy the validator."""
    defaults = PipelineConfig(amberhome=Path('/opt/amber')).ions
    assert set(defaults) <= SUPPORTED_IONS


def test_pipeline_config_water_cutoff_must_be_positive():
    with pytest.raises(ValidationError):
        PipelineConfig(amberhome=Path('/opt/amber'), water_cutoff=0.0)


def test_pipeline_config_rejects_typo_in_kwarg():
    with pytest.raises(ValidationError):
        PipelineConfig(
            amberhome=Path('/opt/amber'),
            funcitonal='B3LYP',  # typo
        )


def test_pipeline_config_coerces_string_paths():
    cfg = PipelineConfig(amberhome='/opt/amber', ensemble_path='/tmp/weights')
    assert isinstance(cfg.amberhome, Path)
    assert cfg.amberhome == Path('/opt/amber')
    assert cfg.ensemble_path == Path('/tmp/weights')


def test_pipeline_config_from_yaml_round_trip(tmp_path):
    src = PipelineConfig(
        amberhome=Path('/opt/amber'),
        basis='cc-pVDZ',
        ions=['CA', 'ND'],
        run_dir=Path('/tmp/run42'),
    )
    yaml_path = tmp_path / 'pipeline.yaml'
    yaml_path.write_text(yaml.safe_dump(json.loads(src.model_dump_json())))

    loaded = PipelineConfig.from_yaml(yaml_path)
    assert loaded.basis == 'cc-pVDZ'
    assert loaded.ions == ['CA', 'ND']
    assert loaded.run_dir == Path('/tmp/run42')
    assert loaded.amberhome == Path('/opt/amber')


def test_pipeline_config_from_yaml_validates(tmp_path):
    yaml_path = tmp_path / 'bad.yaml'
    yaml_path.write_text(yaml.safe_dump({
        'amberhome': '/opt/amber',
        'ions': ['CA', 'PB'],  # PB not supported
    }))
    with pytest.raises(ValidationError):
        PipelineConfig.from_yaml(yaml_path)

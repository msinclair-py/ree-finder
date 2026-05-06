"""Pydantic schemas for sidecar files exchanged between pipeline stages.

Stages run in separate Parsl workers, so anything serialized to disk is a
contract — these models pin it down with explicit types and validation.
"""
from datetime import datetime
from pathlib import Path

from pydantic import BaseModel, ConfigDict, Field, field_validator

# Mirror of pipeline.ESMBIND_CHANNEL keys. Duplicated here because pipeline.py
# imports parsl/torch and we want schemas.py to stay heavy-dep-free.
SUPPORTED_IONS: frozenset[str] = frozenset({'CA', 'MG', 'FE', 'ND'})


def _default_run_dir() -> Path:
    """Timestamped output root, evaluated per :class:`PipelineConfig` instance."""
    return Path('outputs') / datetime.now().strftime('%Y%m%d_%H%M%S')


class ClusterMetadata(BaseModel):
    """Sidecar JSON written by :func:`apps.extract_clusters`.

    Records what's in the matching ``cluster_{resid}.xyz`` so the downstream
    QM stage can build a PySCF molecule (charge, spin) and report the
    coordination shell that produced it.
    """

    model_config = ConfigDict(extra='forbid')

    ion: str
    ion_resid: int
    ion_formal_charge: int
    spin_multiplicity_guess: int = Field(ge=1)
    binding_residues: list[int]
    n_atoms: int = Field(gt=0)
    water_cutoff_A: float = Field(ge=0)


class OptimizedClusterMetadata(ClusterMetadata):
    """Sidecar JSON written by :func:`apps.geomopt` after DFT optimization.

    Carries every field of the input :class:`ClusterMetadata` so the file is
    self-contained, plus the converged energy and the DFT settings used.
    """

    energy_hartree: float
    basis: str
    functional: str
    dispersion: str
    grid_level: int
    converged: bool


class ESMBindBaseConfig(BaseModel):
    """Architecture knobs shared by all ESMBind classifier variants.

    Frozen so a misconfigured model can't be silently mutated mid-run.
    Consumed by :class:`model.ESMBindBase` via attribute access, so any
    ``conf.x`` / ``getattr(conf, "x", default)`` call site keeps working.
    """

    model_config = ConfigDict(frozen=True, extra='forbid')

    hidden_dim: int = Field(gt=0)
    dropout: float = Field(ge=0, le=1)
    noise_level: float = Field(ge=0)


class ESMBindSingleConfig(ESMBindBaseConfig):
    """Config for :class:`model.ESMBindSingle` (one feature stream)."""

    feature_dim: int = Field(gt=0)
    hidden_dim_1: int = Field(gt=0)


class ESMBindMultiModalConfig(ESMBindBaseConfig):
    """Config for :class:`model.ESMBindMultiModal` (sequence + structure)."""

    feature_dim_1: int = Field(gt=0)
    feature_dim_2: int = Field(gt=0)
    hidden_dim_1: int = Field(gt=0)
    hidden_dim_2: int = Field(gt=0)


class PipelineConfig(BaseModel):
    """End-to-end run configuration for :class:`pipeline.Pipeline`.

    Frozen so a long-running pipeline can't have its settings mutated mid-flight.
    Load from a YAML file with :meth:`from_yaml` for reproducible runs.
    """

    model_config = ConfigDict(frozen=True, extra='forbid')

    amberhome: Path
    ensemble_path: Path = Path('esmbind_weights')
    basis: str = 'def2-TZVP'
    functional: str = 'B3LYP'
    dispersion: str = 'd3bj'
    water_cutoff: float = Field(default=4.0, gt=0)
    ions: list[str] = Field(default_factory=lambda: ['CA', 'MG', 'FE', 'ND'])
    device: str = 'cuda'
    run_dir: Path = Field(default_factory=_default_run_dir)

    @field_validator('ions')
    @classmethod
    def _validate_ions(cls, v: list[str]) -> list[str]:
        if not v:
            raise ValueError('ions must be non-empty')
        unknown = set(v) - SUPPORTED_IONS
        if unknown:
            raise ValueError(
                f'unsupported ions {sorted(unknown)}; '
                f'supported: {sorted(SUPPORTED_IONS)}'
            )
        return v

    @classmethod
    def from_yaml(cls, path: Path) -> 'PipelineConfig':
        """Load and validate a config from a YAML file."""
        import yaml

        return cls.model_validate(yaml.safe_load(Path(path).read_text()))

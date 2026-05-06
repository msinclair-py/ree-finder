"""Tests for pipeline-level helpers.

Importing pipeline transitively pulls in all of apps (boltz, esm, parsl,
pyscf, …), so we guard with importorskip and only exercise the pure-Python
helper that doesn't need any of that to actually run.
"""
import numpy as np
import pytest

# importorskip only catches ImportError; pipeline pulls in gpu4pyscf which can
# raise CUDARuntimeError on a GPU-less host, so we widen the net.
try:
    import pipeline
except Exception as exc:  # pragma: no cover - environment-dependent
    pytest.skip(
        f'pipeline import failed ({type(exc).__name__}: {exc})',
        allow_module_level=True,
    )


def test_binding_residues_thresholds_with_lower_factor():
    predictions = {
        'threshold_f1': {'CA': 0.4},
        'CA': {'p1': np.array([0.1, 0.5, 0.9, 0.3])},
    }
    # threshold = 0.4 * 0.5 = 0.2 -> indices 1, 2 (0-indexed) -> [2, 3]
    assert pipeline.binding_residues(predictions, 'p1', 'CA') == [2, 3]


def test_binding_residues_lower_factor_widens_recall():
    predictions = {
        'threshold_f1': {'CA': 0.4},
        'CA': {'p1': np.array([0.21, 0.5, 0.9])},
    }
    strict = pipeline.binding_residues(predictions, 'p1', 'CA', lower_factor=1.0)
    loose = pipeline.binding_residues(predictions, 'p1', 'CA', lower_factor=0.5)
    assert set(strict) <= set(loose)
    assert 1 in loose and 1 not in strict


def test_binding_residues_nd_proxies_through_ca_channel():
    predictions = {
        'threshold_f1': {'CA': 0.4},
        'CA': {'p1': np.array([0.5, 0.05])},
    }
    # ND has no own channel — pipeline.ESMBIND_CHANNEL routes ND -> CA
    assert pipeline.binding_residues(predictions, 'p1', 'ND') == [1]


def test_binding_residues_returns_empty_for_unknown_protein():
    predictions = {'threshold_f1': {'CA': 0.4}, 'CA': {}}
    assert pipeline.binding_residues(predictions, 'missing', 'CA') == []


def test_binding_residues_returns_empty_when_nothing_crosses_threshold():
    predictions = {
        'threshold_f1': {'MG': 0.9},
        'MG': {'p1': np.array([0.1, 0.2, 0.3])},
    }
    assert pipeline.binding_residues(predictions, 'p1', 'MG') == []


def test_esmbind_channel_covers_pipeline_default_ions():
    """Every ion in the Pipeline default list must map to a real ESMBind channel."""
    default_ions = ['CA', 'MG', 'FE', 'ND']
    for ion in default_ions:
        assert ion in pipeline.ESMBIND_CHANNEL
        # Channel must be one of the 7 ESMBind training ligands
        assert pipeline.ESMBIND_CHANNEL[ion] in {
            'MG', 'FE', 'CU', 'CO', 'CA', 'MN', 'ZN'
        }

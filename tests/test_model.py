"""Forward-pass shape checks for the ESMBind model variants on CPU."""
from types import SimpleNamespace

import pytest

torch = pytest.importorskip('torch')
from model import ESMBindBase, ESMBindMultiModal, ESMBindSingle  # noqa: E402

LIGANDS = ['ZN', 'CA', 'MG', 'MN', 'FE', 'CO', 'CU']


def _eval(model):
    model.eval()
    return model


def test_base_forward_shape():
    conf = SimpleNamespace(
        feature_dim=8, hidden_dim=4, dropout=0.0, noise_level=0.0
    )
    model = _eval(ESMBindBase(conf))
    x = torch.randn(2, 5, 8)
    with torch.no_grad():
        out = model(x, 'ZN', training=False)
    assert out.shape == (2, 5)


def test_base_supports_all_ligand_heads():
    conf = SimpleNamespace(
        feature_dim=4, hidden_dim=4, dropout=0.0, noise_level=0.0
    )
    model = _eval(ESMBindBase(conf))
    x = torch.randn(1, 3, 4)
    with torch.no_grad():
        for ligand in LIGANDS:
            assert model(x, ligand, training=False).shape == (1, 3)


def test_single_forward_shape():
    conf = SimpleNamespace(
        feature_dim=8, hidden_dim_1=6, hidden_dim=4,
        dropout=0.0, noise_level=0.0,
    )
    model = _eval(ESMBindSingle(conf))
    x = torch.randn(2, 5, 8)
    with torch.no_grad():
        out = model(x, 'CA', training=False)
    assert out.shape == (2, 5)
    # The wrapper exposes encoder + classifier under params for SWA loading.
    assert {'encoder', 'classifier'} <= set(model.params)


def test_multimodal_forward_shape():
    conf = SimpleNamespace(
        feature_dim_1=8, feature_dim_2=4,
        hidden_dim_1=6, hidden_dim_2=6, hidden_dim=4,
        dropout=0.0, noise_level=0.0,
    )
    model = _eval(ESMBindMultiModal(conf))
    a = torch.randn(2, 5, 8)
    b = torch.randn(2, 5, 4)
    with torch.no_grad():
        out = model(a, b, 'CA', training=False)
    assert out.shape == (2, 5)
    assert {'encoder', 'classifier'} <= set(model.params)


def test_multimodal_deterministic_in_eval_mode():
    conf = SimpleNamespace(
        feature_dim_1=4, feature_dim_2=4,
        hidden_dim_1=4, hidden_dim_2=4, hidden_dim=4,
        dropout=0.0, noise_level=0.0,
    )
    model = _eval(ESMBindMultiModal(conf))
    a = torch.randn(1, 3, 4)
    b = torch.randn(1, 3, 4)
    with torch.no_grad():
        out1 = model(a, b, 'ZN', training=False)
        out2 = model(a, b, 'ZN', training=False)
    assert torch.equal(out1, out2)


def test_add_noise_returns_input_when_zero_level():
    conf = SimpleNamespace(
        feature_dim=4, hidden_dim=4, dropout=0.0, noise_level=0.0
    )
    model = ESMBindBase(conf)
    x = torch.randn(2, 3, 4)
    assert torch.equal(model.add_noise(x), x)

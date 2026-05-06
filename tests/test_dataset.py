"""MultimodalDataset round-trip with real .npy files on disk."""
import numpy as np
import pytest

torch = pytest.importorskip('torch')
from dataset import MultimodalDataset  # noqa: E402


@pytest.fixture
def tiny_dataset(tmp_path):
    seq_dir = tmp_path / 'seq'
    str_dir = tmp_path / 'str'
    seq_dir.mkdir()
    str_dir.mkdir()

    # Two proteins of different lengths so we exercise padding.
    rng = np.random.default_rng(0)
    np.save(seq_dir / 'p1.npy', rng.standard_normal((7, 8)).astype(np.float32))
    np.save(str_dir / 'p1.npy', rng.standard_normal((7, 4)).astype(np.float32))
    np.save(seq_dir / 'p2.npy', rng.standard_normal((3, 8)).astype(np.float32))
    np.save(str_dir / 'p2.npy', rng.standard_normal((3, 4)).astype(np.float32))

    return MultimodalDataset(['p1', 'p2'], seq_dir, str_dir)


def test_dim_inference_from_first_id(tiny_dataset):
    assert tiny_dataset.dim_1 == 8
    assert tiny_dataset.dim_2 == 4


def test_len_and_getitem(tiny_dataset):
    assert len(tiny_dataset) == 2
    _id, seq, struct = tiny_dataset[0]
    assert _id == 'p1'
    assert seq.shape == (7, 8)
    assert struct.shape == (7, 4)


def test_collate_pads_to_longest_and_emits_mask(tiny_dataset):
    batch = [tiny_dataset[0], tiny_dataset[1]]  # lengths 7 and 3
    ids, feat_1, feat_2, mask = tiny_dataset.collate_fn(batch)

    assert ids == ['p1', 'p2']
    assert feat_1.shape == (2, 7, 8)
    assert feat_2.shape == (2, 7, 4)
    assert mask.shape == (2, 7)

    # mask is 1 over real tokens, 0 over padding
    assert mask[0].tolist() == [1] * 7
    assert mask[1].tolist() == [1] * 3 + [0] * 4

    # padded rows are exactly zero
    assert torch.all(feat_1[1, 3:] == 0)
    assert torch.all(feat_2[1, 3:] == 0)

    # real rows match the original arrays
    _, seq_p2, _ = tiny_dataset[1]
    assert np.allclose(feat_1[1, :3].numpy(), seq_p2)

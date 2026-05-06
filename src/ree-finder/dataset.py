"""Dataset wrapping per-protein ESM-2 and ESM-IF embeddings for ESMBind."""
import numpy as np
import torch
from pathlib import Path
from torch.utils.data import Dataset

try:
    import intel_extension_for_pytorch
except ImportError:
    pass

class MultimodalDataset(Dataset):
    """Pair sequence and structure embeddings stored as one ``.npy`` per protein.

    Both ``seq_dir`` and ``str_dir`` are expected to contain ``{id}.npy`` files
    of shape ``(seq_len, dim)``. Per-protein lengths may vary; the
    :meth:`collate_fn` zero-pads each batch to its longest member and emits a
    boolean-style mask. The feature dimensions ``dim_1`` and ``dim_2`` are
    inferred from the first id at construction time via mmap (no full read).

    Attributes:
        ids: Protein identifiers used to look up files in both directories.
        seq_dir: Directory of sequence-embedding ``.npy`` files (e.g. ESM-2).
        str_dir: Directory of structure-embedding ``.npy`` files (e.g. ESM-IF).
        dim_1: Feature dimension of the sequence embeddings.
        dim_2: Feature dimension of the structure embeddings.
    """

    def __init__(
        self,
        ids: list[str],
        seq_dir: Path,
        str_dir: Path,
    ):
        """Initialize the dataset and infer feature dimensions from the first id.

        Args:
            ids: Protein identifiers; each must have a corresponding ``.npy``
                in both ``seq_dir`` and ``str_dir``.
            seq_dir: Directory of sequence-embedding files.
            str_dir: Directory of structure-embedding files.
        """
        self.ids = ids
        self.seq_dir = Path(seq_dir)
        self.str_dir = Path(str_dir)

        sample_seq = np.load(self.seq_dir / f'{ids[0]}.npy', mmap_mode='r')
        sample_str = np.load(self.str_dir / f'{ids[0]}.npy', mmap_mode='r')
        self.dim_1 = sample_seq.shape[1]
        self.dim_2 = sample_str.shape[1]

    def __len__(self) -> int:
        """Return the number of proteins in the dataset."""
        return len(self.ids)

    def __getitem__(self, idx: int) -> tuple[str,
                                             np.ndarray,
                                             np.ndarray]:
        """Load and return ``(id, seq_feat, struct_feat)`` for a single protein."""
        _id = self.ids[idx]
        feat_1 = np.load(self.seq_dir / f'{_id}.npy')
        feat_2 = np.load(self.str_dir / f'{_id}.npy')
        return _id, feat_1, feat_2

    def padding(self, batch, maxlen: int) -> tuple[list[int],
                                                   torch.Tensor,
                                                   torch.Tensor,
                                                   torch.Tensor]:
        """Zero-pad a batch of variable-length embeddings to ``maxlen``.

        Args:
            batch: Sequence of ``(id, seq_feat, struct_feat)`` tuples as
                produced by :meth:`__getitem__`.
            maxlen: Target sequence length to pad each tensor to.

        Returns:
            Tuple ``(ids, feat_1, feat_2, mask)`` where the three tensors
            have shape ``(batch_size, maxlen, dim)`` (or ``(batch_size,
            maxlen)`` for ``mask``) and ``mask`` is ``1`` over the original
            tokens and ``0`` over padding.
        """
        batch_feat_1 = []
        batch_feat_2 = []
        batch_mask = []
        batch_id = []

        for _id, feat_1, feat_2 in batch:
            batch_id.append(_id)
            padded_feat_1 = np.zeros((maxlen, self.dim_1))
            padded_feat_1[:feat_1.shape[0], :] = feat_1
            padded_feat_1 = torch.tensor(padded_feat_1, dtype=torch.float)
            batch_feat_1.append(padded_feat_1)

            padded_feat_2 = np.zeros((maxlen, self.dim_2))
            padded_feat_2[:feat_2.shape[0], :] = feat_2
            padded_feat_2 = torch.tensor(padded_feat_2, dtype=torch.float)
            batch_feat_2.append(padded_feat_2)

            mask = np.zeros(maxlen)
            mask[:feat_1.shape[0]] = 1
            mask = torch.tensor(mask, dtype=torch.long)
            batch_mask.append(mask)

        return (
            batch_id,
            torch.stack(batch_feat_1),
            torch.stack(batch_feat_2),
            torch.stack(batch_mask),
        )

    def collate_fn(self, batch) -> tuple[list[int],
                                         torch.Tensor,
                                         torch.Tensor,
                                         torch.Tensor]:
        """DataLoader collate hook that pads each batch to its longest member."""
        maxlen = max([feat_1.shape[0] for _, feat_1, _ in batch])
        return self.padding(batch, maxlen)

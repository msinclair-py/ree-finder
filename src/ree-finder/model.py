"""ESMBind classifier modules: shared base, single-modal, and multimodal heads."""
import torch
import torch.nn as nn
import torch.nn.functional as F

try:
    import intel_extension_for_pytorch
except ImportError:
    pass

class ESMBindBase(nn.Module):
    """Shared base: input projection plus per-ligand binary classification heads.

    Subclasses override ``input_block`` to define how raw protein features are
    encoded; the base provides the per-residue per-ligand logits via
    ``classifier_heads`` (one ``Linear`` per ion type). Optional Gaussian
    noise injection is applied to the inputs during training as light data
    augmentation.

    Args:
        conf: Namespace with ``feature_dim``, ``hidden_dim``, ``dropout``,
            and ``noise_level`` attributes.
    """

    def __init__(self, conf):
        super(ESMBindBase, self).__init__()
        self.noise_level = conf.noise_level
        feature_dim = getattr(conf, "feature_dim", 1)
        self.ligand_types = [
            "ZN",
            "CA",
            "MG",
            "MN",
            "FE",
            "CO",
            "CU",
        ]
        self.input_block = nn.Sequential(
            nn.LayerNorm(feature_dim, eps=1e-6),
            nn.Dropout(conf.dropout),
            nn.Linear(feature_dim, conf.hidden_dim),
            nn.LeakyReLU(),
        )

        # Ligand-specific layers
        self.classifier_heads = nn.ModuleDict(
            {
                ligand: nn.Sequential(
                    nn.LayerNorm(conf.hidden_dim, eps=1e-6),
                    nn.Dropout(conf.dropout),
                    nn.Linear(conf.hidden_dim, 1, bias=True),
                )
                for ligand in self.ligand_types
            }
        )

        # Initialization
        for p in self.parameters():
            if p.dim() > 1:
                nn.init.xavier_uniform_(p)

    def add_noise(self, input):
        """Add zero-mean Gaussian noise scaled by ``self.noise_level``."""
        return input + self.noise_level * torch.randn_like(input)

    def get_logits(self, input, ligand):
        """Run the ligand-specific classification head and squeeze the trailing dim."""
        return self.classifier_heads[ligand](input).squeeze(-1)

    def forward(self, protein_feat, ligand, training=True):
        """Project features and emit per-residue logits for ``ligand``.

        Args:
            protein_feat: Per-residue features, shape ``(batch, seq_len,
                feature_dim)``.
            ligand: Ion code keying into ``classifier_heads``.
            training: If ``True``, apply Gaussian input noise before encoding.

        Returns:
            Logits tensor of shape ``(batch, seq_len)``.
        """
        if training:
            protein_feat = self.add_noise(protein_feat)
        output = self.input_block(protein_feat)
        logits = self.get_logits(output, ligand)
        return logits


class ESMBindSingle(ESMBindBase):
    """Single-modality variant: a two-layer MLP encoder over one feature stream.

    Args:
        conf: Namespace with ``feature_dim``, ``hidden_dim_1``, ``hidden_dim``,
            ``dropout``, and ``noise_level`` attributes. The
            ``feature_dim`` → ``hidden_dim_1`` → ``hidden_dim`` projection
            replaces the single-layer encoder from :class:`ESMBindBase`.
    """

    def __init__(self, conf):
        super(ESMBindSingle, self).__init__(conf)
        modules = [
            nn.LayerNorm(conf.feature_dim, eps=1e-6),
            nn.Dropout(conf.dropout),
            nn.Linear(conf.feature_dim, conf.hidden_dim_1),
            nn.LeakyReLU(),
            nn.LayerNorm(conf.hidden_dim_1, eps=1e-6),
            nn.Dropout(conf.dropout),
            nn.Linear(conf.hidden_dim_1, conf.hidden_dim),
            nn.LeakyReLU(),
        ]
        self.input_block = nn.Sequential(*modules)
        self.params = nn.ModuleDict(
            {"encoder": self.input_block, "classifier": self.classifier_heads}
        )


class ESMBindMultiModal(ESMBindBase):
    """Two-stream encoder fusing sequence (ESM-2) and structure (ESM-IF) features.

    Each stream is independently projected and L2-normalized, the results are
    concatenated, and a third block reduces to ``hidden_dim`` before the
    per-ligand classification heads from :class:`ESMBindBase` produce logits.

    Args:
        conf: Namespace with ``feature_dim_1``, ``feature_dim_2``,
            ``hidden_dim_1``, ``hidden_dim_2``, ``hidden_dim``, ``dropout``,
            and ``noise_level`` attributes.
    """

    def __init__(self, conf):
        super(ESMBindMultiModal, self).__init__(conf)
        # Define feature and hidden dimensions
        self.feature_dims = [conf.feature_dim_1, conf.feature_dim_2]
        self.hidden_dims = [conf.hidden_dim_1, conf.hidden_dim_2, conf.hidden_dim]

        # Create input blocks dynamically
        self.input_blocks = nn.ModuleList(
            [
                self.create_block(in_dim, out_dim, conf.dropout)
                for in_dim, out_dim in zip(
                    self.feature_dims + [sum(self.hidden_dims[:2])], self.hidden_dims
                )
            ]
        )
        self.params = nn.ModuleDict(
            {"encoder": self.input_blocks, "classifier": self.classifier_heads}
        )

    def create_block(self, input_dim, output_dim, dropout_rate):
        """Return a LayerNorm → Dropout → Linear → LeakyReLU encoder block."""
        return nn.Sequential(
            nn.LayerNorm(input_dim, eps=1e-6),
            nn.Dropout(dropout_rate),
            nn.Linear(input_dim, output_dim),
            nn.LeakyReLU(),
        )

    def forward(self, feat_a, feat_b, ligand, training=True):
        """Fuse sequence and structure features and emit per-residue logits.

        Args:
            feat_a: Sequence-side features, shape
                ``(batch, seq_len, feature_dim_1)``.
            feat_b: Structure-side features, shape
                ``(batch, seq_len, feature_dim_2)``.
            ligand: Ion code keying into ``classifier_heads``.
            training: If ``True``, apply Gaussian noise to both inputs.

        Returns:
            Logits tensor of shape ``(batch, seq_len)``.
        """
        if training:
            feat_a = self.add_noise(feat_a)
            feat_b = self.add_noise(feat_b)

        # Process the features through the respective blocks
        output_1 = F.normalize(self.input_blocks[0](feat_a), dim=-1)
        output_2 = F.normalize(self.input_blocks[1](feat_b), dim=-1)

        # Concatenate and further process the features
        concatenated_features = torch.cat((output_1, output_2), dim=-1)
        combined_output = self.input_blocks[2](concatenated_features)

        logits = self.get_logits(combined_output, ligand)
        return logits

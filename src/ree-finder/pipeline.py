"""Parsl-orchestrated REE binding-site discovery pipeline.

Wires the per-stage kernels in :mod:`apps` into a fan-out / fan-back-out
graph: sequences → folded structures → embeddings → ESMBind predictions →
per-(protein, ion) MD relaxation → per-cluster DFT geometry optimization.
"""
from pathlib import Path

import numpy as np

# Import torch before gpu4pyscf so the system CUDA stack (linked against
# system torch) gets loaded first; otherwise gpu4pyscf's bundled cuda-12.8
# nvjitlink shadows it and breaks system torch's cusparse.
from parsl import python_app

try:
    import intel_extension_for_pytorch  # noqa: F401
except ImportError:
    pass

try:
    from gpu4pyscf import dft  # noqa: F401
    qm_exec = 'gpu'
except ImportError:
    qm_exec = 'cpu'

from apps import (
    esmbind_inference,
    fold,
    geomopt,
    relax,
    sequence_embeddings,
    structural_embeddings,
)
from dataset import MultimodalDataset
from schemas import PipelineConfig

# Which esmbind output channel to use for each target ion.
# ESMBind was trained on {MG, FE, CU, CO, CA, MN, ZN}; Nd is proxied through Ca.
ESMBIND_CHANNEL = {
    'CA': 'CA',
    'MG': 'MG',
    'FE': 'FE',
    'ND': 'CA',
}

def binding_residues(
    predictions: dict,
    protein_id: str,
    ion: str,
    lower_factor: float = 0.5,
) -> list[int]:
    """Threshold ESMBind probabilities into a list of 1-indexed binding residues.

    Looks up the appropriate ESMBind output channel for ``ion`` (Nd is proxied
    through the Ca channel, since ESMBind was not trained on lanthanides),
    multiplies the channel's stored F1-optimal threshold by ``lower_factor``
    to widen recall, and returns the residues whose averaged sigmoid score
    exceeds it.

    Args:
        predictions: Output of :func:`apps.esmbind_inference`.
        protein_id: Identifier whose probabilities to threshold.
        ion: Target ion code (must be a key of ``ESMBIND_CHANNEL``).
        lower_factor: Multiplier applied to the stored F1 threshold;
            ``< 1`` trades precision for recall.

    Returns:
        Sorted 1-indexed residue numbers crossing the threshold, or
        ``[]`` if the protein has no predictions for the chosen channel.
    """
    channel = ESMBIND_CHANNEL[ion]
    threshold = predictions['threshold_f1'][channel] * lower_factor
    probs = predictions[channel].get(protein_id)
    if probs is None:
        return []
    return (np.where(probs > threshold)[0] + 1).tolist()


class Pipeline:
    """End-to-end orchestrator for the REE binding-site workflow.

    Takes a validated :class:`schemas.PipelineConfig` and registers each
    :mod:`apps` kernel as a Parsl ``python_app`` targeting an appropriate
    executor. ``run`` consumes a multi-FASTA and drives the fan-out → reduce
    → fan-out execution graph.

    Attributes:
        config: The :class:`PipelineConfig` driving this run.
        ensemble_path: Mirrored from ``config`` for convenience inside :meth:`run`.
        amberhome: Mirrored from ``config``.
        basis: Mirrored from ``config``.
        functional: Mirrored from ``config``.
        dispersion: Mirrored from ``config``.
        water_cutoff: Mirrored from ``config``.
        ions: Mirrored from ``config``.
        device: Mirrored from ``config``.
        run_dir: Mirrored from ``config``.
    """

    def __init__(self, config: PipelineConfig):
        """Capture run config and wrap each kernel as a Parsl app.

        Args:
            config: Validated :class:`PipelineConfig`. Construct directly,
                or load from YAML via :meth:`PipelineConfig.from_yaml`.
        """
        self.config = config
        self.ensemble_path = config.ensemble_path
        self.amberhome = config.amberhome
        self.basis = config.basis
        self.functional = config.functional
        self.dispersion = config.dispersion
        self.water_cutoff = config.water_cutoff
        self.ions = config.ions
        self.device = config.device
        self.run_dir = config.run_dir
        self._register_apps()

    def _register_apps(self) -> None:
        """Wrap each :mod:`apps` kernel as a Parsl app on the right executor.

        All ML and MD stages target the ``'gpu'`` executor; the DFT stage
        targets the ``'gpu'`` executor when ``gpu4pyscf`` imports cleanly,
        otherwise the CPU executor.
        """
        self.fold = python_app(fold, executors=['gpu'])
        self.sequence_embeddings = python_app(sequence_embeddings, executors=['gpu'])
        self.structural_embeddings = python_app(structural_embeddings, executors=['gpu'])
        self.predict_metal_binding = python_app(esmbind_inference, executors=['gpu'])
        self.relax = python_app(relax, executors=['gpu'])
        self.geomopt = python_app(geomopt, executors=[qm_exec])

    def run(self, fasta: Path) -> list:
        """Execute the full pipeline on a multi-FASTA of input proteins.

        Parses the FASTA (header lines must use ``>name|id|...`` format;
        the second pipe-delimited field is taken as the protein id), folds
        every sequence in parallel, computes sequence and structure
        embeddings, runs ESMBind, and for each ``(protein, ion)`` with a
        non-empty binding set runs MD relaxation followed by per-cluster
        DFT geometry optimization.

        Args:
            fasta: Path to the multi-FASTA file.

        Returns:
            List of ``(opt_coords, e_final)`` tuples — one per QM cluster
            successfully optimized — as produced by :func:`apps.geomopt`.
        """
        lines = fasta.read_text().splitlines()
        headers = [line.split('|')[1].strip() for line in lines[::2]]
        sequences = [line.strip() for line in lines[1::2]]

        folds_dir = self.run_dir / 'folds'
        seq_dir = self.run_dir / 'embeddings' / 'sequence'
        str_dir = self.run_dir / 'embeddings' / 'structure'
        sims_root = self.run_dir / 'sims'

        # fold all sequences in parallel
        fold_futures = [
            self.fold(h, s, folds_dir)
            for h, s in zip(headers, sequences, strict=True)
        ]

        # sequence embedding can run while folds are still in flight
        seq_future = self.sequence_embeddings(
            headers, sequences, seq_dir, self.device
        )

        # parsl only auto-unwraps top-level future args, so resolve the list
        # of fold futures to concrete paths before passing them in
        pdbs = [f.result() for f in fold_futures]
        seq_future.result()
        self.structural_embeddings(headers, pdbs, str_dir, self.device).result()

        # one esmbind pass over the full dataset
        dataset = MultimodalDataset(headers, seq_dir, str_dir)
        predictions = self.predict_metal_binding(
            dataset, self.ensemble_path, self.device
        ).result()

        # fan back out: one relax per (protein, ion) with predicted binding residues
        relaxes = []
        for header, pdb in zip(headers, pdbs, strict=True):
            for ion in self.ions:
                residues = binding_residues(predictions, header, ion)
                if not residues:
                    continue
                relaxes.append(self.relax(
                    pdb_file=pdb,
                    binding_residues=residues,
                    ion=ion,
                    output_dir=sims_root / header / ion,
                    water_cutoff=self.water_cutoff,
                    amberhome=self.amberhome,
                ))

        # fan further: one geomopt per cluster returned by each relax
        optima = [
            self.geomopt(
                xyz_path=xyz,
                json_path=xyz.with_suffix('.json'),
                basis=self.basis,
                functional=self.functional,
                dispersion=self.dispersion,
            )
            for relax_future in relaxes
            for xyz in relax_future.result()
        ]

        return [f.result() for f in optima]

if __name__ == '__main__':
    pass

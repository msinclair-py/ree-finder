# ree-finder

A Parsl-orchestrated pipeline for predicting and refining metal binding sites
in proteins, with a particular focus on rare-earth elements (REEs).

Given a multi-FASTA of input proteins, ree-finder folds each sequence,
predicts per-residue metal-binding probabilities, builds and relaxes an
explicit-solvent Amber system around each predicted site, extracts the
coordination cluster, and runs a DFT geometry optimization on it.

## Pipeline overview

```
sequences (FASTA)
     │
     ▼
  fold (Boltz)  ────►  PDB structures
     │
     ├──►  sequence embeddings   (ESM-2)
     └──►  structure embeddings  (ESM-IF)
                       │
                       ▼
              ESMBind ensemble  ─►  per-residue binding probabilities
                       │
                       ▼   (one fan-out per (protein, ion) with hits)
              Amber MD relaxation  ─►  coordination clusters (.xyz + .json)
                       │
                       ▼   (one fan-out per cluster)
              DFT geometry optimization (PySCF / gpu4pyscf)
                       │
                       ▼
              optimized coords + energies
```

Stages are wired together in [`pipeline.py`](src/ree-finder/pipeline.py); the
underlying compute kernels live in [`apps.py`](src/ree-finder/apps.py).

### Supported ions

`CA`, `MG`, `FE`, `ND`. Nd is proxied through the Ca channel of ESMBind, since
ESMBind was not trained on lanthanides; per-ion chemistry (coordination
preferences, formal charge, spin) is encoded in
[`defaults.py`](src/ree-finder/defaults.py).

## Installation

```bash
pip install -e .
```

GPU acceleration for the DFT stage:

```bash
pip install -e ".[gpu]"
```

Intel-extension PyTorch (optional):

```bash
pip install -e ".[intel]"
```

External tools the pipeline shells out to:

- **AmberTools** (`tleap`, `pdb4amber`) — point `amberhome` at your install root.
- **Boltz** weights are cached under `src/ree-finder/boltz_cache/` on first run.
- **ESMBind** ensemble weights — path is configured via `ensemble_path`.

## Usage

Configure a run via [`PipelineConfig`](src/ree-finder/schemas.py) and hand it
to `Pipeline.run`:

```python
from pathlib import Path
from pipeline import Pipeline
from schemas import PipelineConfig

config = PipelineConfig(
    amberhome=Path('/opt/amber24'),
    ensemble_path=Path('esmbind_weights'),
    ions=['CA', 'MG', 'FE', 'ND'],
    water_cutoff=4.0,
    basis='def2-TZVP',
    functional='B3LYP',
    dispersion='d3bj',
    device='cuda',
)
results = Pipeline(config).run(Path('proteins.fasta'))
```

Or load a config from YAML:

```python
config = PipelineConfig.from_yaml(Path('run.yaml'))
```

FASTA headers must use the `>name|id|...` format — the second pipe-delimited
field is taken as the protein id and used as the stem for every per-protein
artifact (folds, embeddings, sims, clusters).

A Parsl config defining `'gpu'` and (optionally) `'cpu'` executors must be
loaded before calling `run`.

### Outputs

Under `config.run_dir` (defaults to `outputs/<timestamp>/`):

```
folds/                          Boltz YAML schemas + predicted PDBs
embeddings/sequence/{id}.npy    ESM-2 per-residue embeddings
embeddings/structure/{id}.npy   ESM-IF per-residue embeddings
sims/{id}/{ion}/                Amber inputs, MD trajectory, cluster_*.xyz/.json
```

Each cluster `.json` validates against
[`ClusterMetadata`](src/ree-finder/schemas.py); after DFT it is rewritten as
[`OptimizedClusterMetadata`](src/ree-finder/schemas.py) with the converged
energy and DFT settings.

`Pipeline.run` returns a list of `(opt_coords, e_final)` tuples — one per
successfully optimized cluster.

## Development

```bash
pip install -e ".[dev]"
pip install -r requirements-test.txt
ruff check .
mypy src/ree-finder
pytest
```

Documentation:

```bash
pip install -e ".[docs]"
mkdocs serve
```

## License

See [LICENSE](LICENSE).

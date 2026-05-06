# ree-finder

Pipeline for predicting and refining metal (incl. rare-earth) binding sites in proteins.

## Installation

```bash
pip install -e .
```

For GPU support:

```bash
pip install -e ".[gpu]"
```

## Development

```bash
pip install -e ".[dev]"
ruff check .
mypy src/ree-finder
pytest
```

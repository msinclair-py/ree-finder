import sys
from pathlib import Path

# The package is laid out as flat top-level modules under src/ree-finder/
# (matching pyproject's `py-modules = [...]`), so put that dir on sys.path
# instead of treating it as a package.
SRC = Path(__file__).resolve().parent.parent / 'src' / 'ree-finder'
if str(SRC) not in sys.path:
    sys.path.insert(0, str(SRC))

# Installation

yasqat requires Python 3.11 or newer.

```bash
pip install yasqat
```

## Development install

```bash
git clone https://github.com/rexarski/yasqat.git
cd yasqat
uv venv && source .venv/bin/activate  # or activate.fish

# Runtime + dev tooling (pytest, ruff, mypy)
uv pip install -e ".[dev]"

# Documentation toolchain (Sphinx, furo, myst-parser)
uv pip install -e ".[docs]"
```

## Building the docs locally

```bash
uv run --extra docs sphinx-build -b html docs docs/_build/html
```

Open `docs/_build/html/index.html` in a browser. The site also builds and
deploys to GitHub Pages automatically on every push to `main`.

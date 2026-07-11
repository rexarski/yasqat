"""Sphinx configuration for the yasqat documentation site."""

from __future__ import annotations

from importlib.metadata import PackageNotFoundError, version as _pkg_version

project = "yasqat"
author = "rexarski"
copyright = "2026, rexarski"  # noqa: A001

try:
    release = _pkg_version("yasqat")
except PackageNotFoundError:  # pragma: no cover - building without an install
    release = "0.0.0"
version = release

extensions = [
    "sphinx.ext.autodoc",
    "sphinx.ext.napoleon",
    "sphinx.ext.viewcode",
    "sphinx.ext.intersphinx",
    "myst_parser",
]

templates_path: list[str] = []
# adr/ and agents/ are decision records and agent config, not user docs — keep
# them in the repo but out of the rendered site.
exclude_patterns = ["_build", "Thumbs.db", ".DS_Store", "adr/*", "agents/*"]

# --- Autodoc / Napoleon -----------------------------------------------------
# yasqat uses Google-style docstrings (Args:/Returns:/Example:); napoleon parses
# them. autodoc honors each subpackage's ``__all__``, so the API pages document
# exactly the public surface.
autodoc_default_options = {
    "members": True,
    "show-inheritance": True,
}
autodoc_member_order = "bysource"
autodoc_typehints = "description"
napoleon_google_docstring = True
napoleon_numpy_docstring = False

# --- MyST (Markdown) --------------------------------------------------------
myst_enable_extensions = ["colon_fence", "deflist"]
myst_heading_anchors = 3

# --- Intersphinx ------------------------------------------------------------
intersphinx_mapping = {
    "python": ("https://docs.python.org/3", None),
    "numpy": ("https://numpy.org/doc/stable", None),
}

# --- HTML output ------------------------------------------------------------
html_theme = "furo"
html_static_path = ["_static"]
html_title = f"yasqat {release}"

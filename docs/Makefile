# Minimal makefile for Sphinx documentation.
# Prefer: uv run --extra docs sphinx-build -b html docs docs/_build/html

SPHINXBUILD  ?= sphinx-build
SOURCEDIR     = .
BUILDDIR      = _build

.PHONY: help html clean
help:
	@$(SPHINXBUILD) -M help "$(SOURCEDIR)" "$(BUILDDIR)"

html:
	@$(SPHINXBUILD) -b html "$(SOURCEDIR)" "$(BUILDDIR)/html"

clean:
	@rm -rf "$(BUILDDIR)"

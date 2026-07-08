# New documentation website

**Status:** `needs-triage`
**Type:** docs / infrastructure
**Source:** user todo list, 2026-07-07
**Milestone:** 0.5.0 (per user intent; confirm at triage whether it blocks the release)

## Description

yasqat currently has **no documentation site**: the old Quarto site was
removed in `fe171a2` ("chore: remove Quarto documentation site") along with
its deploy workflow — `.github/workflows/` now contains only `publish.yml`.

Build a new one. Decisions needed at triage:

- **Generator** — revive Quarto + quartodoc, or switch (mkdocs-material +
  mkdocstrings, Sphinx, pdoc, …). The API surface is ~90 public
  functions/classes with good docstrings, so autodoc leverage matters.
- **Content plan** — quick-start, per-module guides, API reference,
  glossary (seed from `CONTEXT.md`), changelog, and possibly the technical
  primer (`yasqat-overview-en.html`) content reworked as an "about" page.
- **Hosting/deploy** — GitHub Pages via a new workflow (pin action versions;
  remember the setup-uv floating-tag gotcha from v0.4.1).

## Consequences to fix alongside

- The `yasqat-release` skill's Phase 3/7 still reference `docs/changelog.qmd`
  and `docs/_quarto.yml` — stale since `fe171a2`. Update the skill (or the
  new site must recreate those paths).

## Tasks

- [ ] Triage: choose generator + hosting with the user.
- [ ] Content outline.
- [ ] Deploy workflow (pinned actions).
- [ ] Update the release skill's docs-related phases to match reality.

## Comments

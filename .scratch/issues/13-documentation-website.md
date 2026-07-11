# New documentation website

**Status:** `resolved` (0.5.0)
**Type:** docs / infrastructure
**Source:** user todo list, 2026-07-07
**Milestone:** 0.5.0

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

- [x] Triage: choose generator + hosting with the user.
- [x] Content outline.
- [x] Deploy workflow (pinned actions).
- [x] Update the release skill's docs-related phases to match reality.

## Comments

- 2026-07-11: **Resolved (0.5.0).** Chose **Sphinx** (not Quarto) per user —
  `sphinx.ext.autodoc` + `napoleon` (Google-style docstrings) + `myst-parser`
  (Markdown) with the **furo** theme. Site under `docs/`: `index`,
  `installation`, `quickstart`, a `glossary` and `changelog` that render
  `CONTEXT.md` / `CHANGELOG.md` in place via MyST `{include}` (single source of
  truth — no `.qmd`-style copies that go stale), and an `api/` reference with
  one `automodule` page per subpackage (honors each `__all__`). New `docs`
  optional-dependency group; ghost `quartodoc` dropped from `dev`. Hosting:
  `.github/workflows/docs.yml` builds and deploys to GitHub Pages on push to
  `main` (actions pinned; `setup-uv@v8.2.0`). Builds locally with **zero
  warnings**; fixed two docstring list-formatting nits surfaced by docutils.
  **Consequence fix done:** the `yasqat-release` skill's Phase 3/7/8/12 and
  quick-reference no longer point at `docs/changelog.qmd`, `docs/_quarto.yml`,
  `quarto preview`, or `deploy-docs.yml`.
- **Follow-ups (not blocking):** GitHub Pages must be enabled once in repo
  settings (Settings → Pages → Source: GitHub Actions) before the first deploy
  succeeds. The workflow only fires on `main`, so the site goes live when 0.5.0
  merges. Narrative tutorials (the old Quarto site had per-topic tutorials) were
  not ported — quickstart + API + glossary is the initial scope.

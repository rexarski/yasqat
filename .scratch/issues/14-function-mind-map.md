# Mind map of functions

**Status:** `needs-triage`
**Type:** docs
**Source:** user todo list, 2026-07-07
**Related:** issue 13 (a natural page of the new documentation website);
the technical primer (`yasqat-overview-en.html`) already contains a
module-level pipeline diagram to build on

## Description

A visual map of the public API — the ~90 functions/classes across the seven
modules — showing what exists and how the pieces feed each other
(io → SequencePool → DistanceMatrix → clustering / discrepancy;
statistics and filters hanging off the containers).

Decisions needed at triage:

- **Granularity** — module-level only (7 nodes) vs every public symbol
  (~90 leaves) vs two-tier (modules expanded on demand).
- **Format** — static SVG checked into the repo, interactive HTML
  (e.g. D3/markmap), Mermaid mindmap in markdown, or an Excalidraw/Figma
  artifact. If issue 13 lands, the format should embed in that site.
- **Source of truth** — hand-drawn once, or generated from the `__all__`
  exports so it can't drift (a small script walking
  `src/yasqat/*/__init__.py` would keep it honest).

## Tasks

- [ ] Triage: pick granularity + format + generation strategy with the user.
- [ ] Build it; wire into the docs site if 13 is done first.

## Comments

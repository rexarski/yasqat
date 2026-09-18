# Compile `pitch.md`

**Status:** `resolved` (dev, 2026-09-17)
**Type:** docs
**Source:** migrated from GitHub #53 (closed 2026-06-22)

## Description

Write a compelling pitch document covering:

- What yasqat is and why it exists
- How it compares to TraMineR and TanaT
- Performance advantages (polars, numba)
- API design philosophy
- Roadmap

> **Note:** the original "roadmap to v0.4" framing is stale — v0.4.0 shipped
> 2026-05-19. Update the roadmap section to post-v0.4.0.

## Comments

## Resolution (2026-09-17)

Resolved by the documentation site rather than a standalone `pitch.md`. The
technical primer (`docs/_static/primer/`, written by the maintainer, so the
human voice is there) was folded into maintained pages:

- `docs/why-yasqat.md`: the problem, the approach, what it is and is not
  (including the honest comparison with Spark MLlib), engineering discipline,
  status and roadmap (post-v0.5.0).
- `docs/concepts/`: what sequence analysis is, lineage from TraMineR/TanaT,
  the data model, the pipeline and its four seams.

Performance claims are stated with the measurements from issue 17.

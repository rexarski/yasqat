# Lift distance-matrix quality functions out of clustering/

**Status:** `resolved` (dev, 2026-10-03)
**Type:** enhancement / architecture
**Source:** architecture review 2026-07-11, candidate D (Speculative;
deferred from the 0.5.0 review)

## Description

`clustering/quality.py` exports functions that are coupled only to a
`DistanceMatrix` (+ a labelling), not to any clustering object:
`silhouette_score`, `silhouette_scores`, `distance_to_center`,
`cluster_quality`, `k_range`, `pam_range`. They live in `clustering/` but read
a bare matrix — the dependency direction is inverted (quality needn't import
the algorithms).

Two smaller notes surfaced alongside:

- **Doubled PAM surface.** Both `PAMClustering` (class) and `pam_clustering`
  (function) are public; likewise `HierarchicalClustering` / `hierarchical_clustering`.
  Worth deciding on one canonical entry point (or documenting why both exist).

## Proposed shape

- Relocate the quality functions next to `DistanceMatrix` (e.g. a
  `metrics/quality.py` or a top-level `quality` surface) so quality reads the
  matrix it measures and `clustering/` narrows to the algorithms.
- Decide the canonical PAM/hierarchical entry point.

## Why deferred

This is a **breaking public-API move** — import paths change
(`from yasqat.clustering import silhouette_score` → elsewhere). 0.5.0 already
carries breaking changes, so it *could* ride along; but it is editorial rather
than corrective, and worth its own triage rather than being bundled under the
release wire. Marked Speculative in the review.

## Tasks

- [x] Triage: decide destination module for the quality functions.
- [x] Decide the canonical clustering entry points (class vs. function).
- [x] Move + update imports, docs, and the `clustering`/target `__all__`.

## Comments

- 2026-07-11: Filed from the architecture review (candidate D). See issue 15
  for the sibling deferred candidate (B).
- 2026-10-03: User approved the recommendation (functions only, quality under
  `yasqat.metrics`). Shipped: `metrics/quality.py` holds the four matrix-only
  indices; `clustering/k_selection.py` holds `k_range` / `pam_range` (they run
  PAM, so they stay on the algorithm side); `PAMClustering` and
  `HierarchicalClustering` deleted, `predict` moved onto
  `PAMClusteringResult`. Breaking, listed in the changelog for 0.6.0.

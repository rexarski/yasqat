# n_jobs thread pool is slower than sequential (GIL not released)

**Status:** `resolved` (dev, 2026-09-17)
**Type:** bug (performance)
**Source:** surfaced while fact-checking the v0.5.0 slide deck, 2026-07-20

## Description

`SequencePool.compute_distances(n_jobs=...)` parallelizes the pairwise loop
with a `ThreadPoolExecutor` (`core/pool.py`, ~line 254). The numba kernels in
`metrics/` are compiled with plain `@njit` — **none pass `nogil=True`** — so
compiled calls still hold the GIL and the worker threads serialize. The thread
pool adds pure overhead.

## Measured

Benchmark on 2026-07-20 (macOS arm64, polars 1.37.1, warm JIT), 400 synthetic
Markov sequences × length 24, OM distance, 79 800 pairs:

| n_jobs | wall time |
|---|---|
| 1 | 0.32 s |
| 4 | **0.75 s** (2.3× slower) |

## Fix options

1. **`@njit(nogil=True)`** on the metric kernels — threads then genuinely
   overlap; smallest change, keeps the ThreadPoolExecutor. Verify each kernel
   is object-mode-free (they are `@njit`, so yes) and re-run the benchmark.
2. `numba.prange` inside a matrix-level kernel — bigger refactor, conflicts
   with the one-seam dispatch design (per-pair free functions).
3. ProcessPoolExecutor — avoid: pickling + per-process JIT warm-up costs
   dominate for this workload.

Option 1 is the intended fix. Acceptance: the benchmark above shows n_jobs=4
meaningfully faster than n_jobs=1, and the full test suite stays green.

## Notes

- The v0.5.0 deck's code example originally showed `n_jobs=8`; removed on
  2026-07-20 so the deck doesn't advertise the broken path.
- Docstring currently says "Number of parallel workers" — if the fix lands,
  no doc change needed; if deferred, consider documenting the limitation.

## Resolution (2026-09-17)

Measured on macOS arm64, warm JIT, OM distance, `indel=1.0`.

**`nogil=True` alone was not enough.** All 24 metric kernels now carry
`@numba.jit(nopython=True, cache=True, nogil=True)`, but on the original
benchmark (400 × 24, 79 800 pairs) `n_jobs=4` stayed slower than sequential
(0.33 s vs 0.94 s). Profiling by elimination showed the per-pair cost is
Python glue, not the kernel: the OM wrapper rebuilt the constant substitution
matrix (`np.full` + `fill_diagonal`) on every call, and `sm.astype` copied on
every call when an array was passed. Threads cannot overlap Python glue.

**Three changes shipped:**

1. `nogil=True` on every kernel in `metrics/` (the PAM kernels in
   `clustering/pam.py` are untouched; nothing threads them).
2. `SequencePool.compute_distances` builds the constant OM substitution
   matrix once per call, sized to the alphabet, and passes it through
   `kwargs["sm"]`. Sequential OM on the 400 × 24 benchmark: 0.34 s → 0.19 s.
3. `optimal_matching_distance` converts an array `sm` with `np.asarray`
   instead of `astype`, so a float64 matrix is not copied per pair.

**Where `n_jobs` now pays off:** when the kernel dominates. 120 sequences ×
300 time points, OM: `n_jobs=1` 1.02 s, `n_jobs=4` 0.33 s (3.1×). For short
sequences the remaining per-pair overhead (call, length checks, dispatch)
still dominates and `n_jobs=1` is faster; the docstring now says so.

**Not done, deliberately:** a matrix-level kernel (option 2) that would make
short sequences scale too. That belongs with the distance-engine seam in
issue 15. The OM variants (`omloc`, `omspell`, …) still build their cost
structures per pair; same hoist applies if they show up in profiles.

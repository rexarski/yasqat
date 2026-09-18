# n_jobs thread pool is slower than sequential (GIL not released)

**Status:** `ready-for-agent`
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

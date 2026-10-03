"""Choosing the number of clusters: run PAM over a range of k and score each."""

from __future__ import annotations

import warnings

import numpy as np

from yasqat.metrics.base import DistanceMatrix
from yasqat.metrics.quality import cluster_quality


def k_range(start: int, end: int) -> range:
    """Inclusive range of k values.

    ``k_range(2, 10)`` returns ``range(2, 11)``, i.e. all integers 2..10.
    This exists because Python's built-in ``range(2, 10)`` is *exclusive*
    on the upper bound, which is a common source of off-by-one confusion
    when calling :func:`pam_range`.

    Args:
        start: Smallest k (inclusive).
        end: Largest k (inclusive).

    Returns:
        A ``range`` object that iterates every integer from ``start``
        through ``end``, inclusive on both ends.

    Example:
        >>> list(k_range(2, 5))
        [2, 3, 4, 5]
    """
    return range(start, end + 1)


def pam_range(
    dist_matrix: DistanceMatrix | np.ndarray,
    k_values: range | list[int] | tuple[int, int] | None = None,
    max_iter: int = 100,
    init: str = "build",
    *,
    k_range: range | list[int] | tuple[int, int] | None = None,
) -> dict[int, dict[str, float]]:
    """
    Run PAM clustering for a range of k values and return quality metrics.

    For each k in the range, runs PAM clustering and computes cluster
    quality metrics (ASW, PBC, HG, R2). This helps identify the optimal
    number of clusters.

    Args:
        dist_matrix: Symmetric pairwise distance matrix (n x n).
        k_values: Iterable of k values to try. Accepts ``range``, ``list``,
            or a 2-tuple ``(start, end)`` which is treated as an inclusive
            range with a ``DeprecationWarning`` (use :func:`k_range` instead).
            If ``None``, uses ``k_range(2, min(n - 1, 10))`` — every integer
            k from 2 up to at most 10.
        max_iter: Maximum PAM iterations per k.
        init: PAM initialization method.

    Returns:
        Dictionary mapping k -> quality metrics dict.

    Example:
        >>> import numpy as np
        >>> from yasqat.clustering import pam_range, k_range
        >>> dist = np.array([
        ...     [0, 1, 5, 6],
        ...     [1, 0, 5, 6],
        ...     [5, 5, 0, 1],
        ...     [6, 6, 1, 0],
        ... ], dtype=float)
        >>> results = pam_range(dist, k_values=k_range(2, 3))
        >>> sorted(results.keys())
        [2, 3]
    """
    from yasqat.clustering.pam import pam_clustering

    # Backwards-compat: accept legacy ``k_range=`` kwarg. The parameter was
    # renamed to ``k_values`` to free the name ``k_range`` for a dedicated
    # helper function (see :func:`k_range`).
    if k_range is not None:
        if k_values is not None:
            raise TypeError(
                "pam_range() received both 'k_values' and 'k_range'; "
                "use 'k_values' only (k_range= is a legacy alias)."
            )
        k_values = k_range

    dist_matrix = DistanceMatrix.coerce(dist_matrix).values

    n = dist_matrix.shape[0]

    # Auto-expand a (start, end) 2-tuple into an inclusive range. This is a
    # common foot-gun: users write ``pam_range(dm, (2, 10))`` expecting k=2..10
    # but ``for k in (2, 10)`` only visits the endpoints.
    if isinstance(k_values, tuple) and len(k_values) == 2:
        warnings.warn(
            f"pam_range received a 2-tuple {k_values}; treating it as an "
            f"inclusive (start, end) range. Pass `k_range({k_values[0]}, "
            f"{k_values[1]})` explicitly to silence this warning.",
            DeprecationWarning,
            stacklevel=2,
        )
        k_values = range(k_values[0], k_values[1] + 1)

    if k_values is None:
        # Inclusive upper bound 10, but never exceed n-1. We can't call the
        # module-level ``k_range`` helper here because the keyword-only
        # parameter of the same name shadows it in this scope — spell the
        # inclusive range out explicitly.
        k_values = range(2, min(n - 1, 10) + 1)

    results: dict[int, dict[str, float]] = {}

    for k in k_values:
        if k < 2 or k >= n:
            continue

        pam_result = pam_clustering(
            dist_matrix,
            n_clusters=k,
            max_iter=max_iter,
            init=init,
        )

        quality = cluster_quality(dist_matrix, pam_result.labels)
        quality["total_cost"] = pam_result.total_cost
        quality["n_iterations"] = float(pam_result.n_iterations)
        results[k] = quality

    return results

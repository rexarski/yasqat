"""TraMineR ``seqdist`` distance normalisation.

Transcribed from ``DistanceCalculator::normalizeDistance`` in TraMineR
2.2-14 (``src/distancecalculator.h``). ``l1`` and ``l2`` are the two
sequence lengths multiplied by the indel cost, and ``maxdist`` is the
method's maximum possible cost for the pair.
"""

from __future__ import annotations

import math

import numpy as np

NORMS = ("none", "maxlength", "gmean", "maxdist", "yujianbo")


def resolve_norm(normalize: bool | str, auto: str) -> str:
    """Map a ``normalize`` argument to a TraMineR norm name.

    ``False`` / ``"none"`` → no normalisation; ``True`` / ``"auto"`` → the
    method's TraMineR ``norm="auto"`` choice (``auto``); otherwise one of
    ``"maxlength"``, ``"gmean"``, ``"maxdist"``, ``"yujianbo"``.
    """
    if normalize is False:
        return "none"
    if normalize is True:
        return auto
    name = normalize.lower()
    if name == "auto":
        return auto
    if name not in NORMS:
        raise ValueError(f"Unknown normalization {normalize!r}. Choose from {NORMS}.")
    return name


def normalize_distance(
    raw: float, maxdist: float, l1: float, l2: float, norm: str
) -> float:
    """Apply a TraMineR normalisation to a raw distance."""
    if raw == 0.0:
        return 0.0
    if norm == "none":
        return raw
    if norm == "maxlength":
        longest = max(l1, l2)
        return raw / longest if longest > 0 else 0.0
    if norm == "gmean":
        if l1 * l2 == 0:
            return 1.0 if l1 != l2 else 0.0
        return 1.0 - (maxdist - raw) / (2.0 * math.sqrt(l1) * math.sqrt(l2))
    if norm == "maxdist":
        return raw / maxdist if maxdist != 0 else 1.0
    if norm == "yujianbo":
        return (2.0 * raw) / (raw + maxdist) if maxdist != 0 else 1.0
    raise ValueError(f"Unknown normalization {norm!r}.")


def max_sub_cost(sm: np.ndarray, indel: float, norm: str) -> float:
    """``maxscost`` as ``OMdistance::setParameters`` computes it.

    The largest off-diagonal substitution cost capped at ``2 * indel``, or
    exactly ``2 * indel`` under Yujian & Bo normalisation.
    """
    if norm == "yujianbo":
        return 2.0 * indel
    arr = np.asarray(sm, dtype=np.float64)
    n = arr.shape[0]
    upper = arr[np.triu_indices(n, k=1)]
    biggest = float(upper.max()) if upper.size else 0.0
    return min(biggest, 2.0 * indel)

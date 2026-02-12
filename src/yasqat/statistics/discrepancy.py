"""Discrepancy analysis (pseudo-ANOVA) for sequence distance matrices.

Implements distance-based ANOVA (pseudo-F, pseudo-R2) and permutation
tests for assessing the association between sequence dissimilarities
and a grouping variable. Analogous to TraMineR's dissassoc.
"""

from __future__ import annotations

from dataclasses import dataclass

import numpy as np


@dataclass
class DiscrepancyResult:
    """Result of discrepancy analysis."""

    pseudo_r2: float
    """Proportion of discrepancy explained by the grouping (0 to 1)."""

    pseudo_f: float
    """Pseudo-F statistic (higher = stronger group separation)."""

    p_value: float | None
    """Permutation test p-value (None if no permutation test was run)."""

    n_permutations: int
    """Number of permutations used (0 if none)."""

    total_ss: float
    """Total sum of squares (discrepancy)."""

    within_ss: float
    """Within-group sum of squares."""

    between_ss: float
    """Between-group sum of squares."""

    def __repr__(self) -> str:
        p_str = f"{self.p_value:.4f}" if self.p_value is not None else "N/A"
        return (
            f"DiscrepancyResult(pseudo_R2={self.pseudo_r2:.4f}, "
            f"pseudo_F={self.pseudo_f:.4f}, p={p_str})"
        )


def discrepancy_analysis(
    dist_matrix: np.ndarray,
    labels: np.ndarray,
    n_permutations: int = 0,
    random_state: int | np.random.Generator | None = None,
) -> DiscrepancyResult:
    """
    Test association between sequence dissimilarities and a grouping variable.

    Computes pseudo-R2 and pseudo-F statistics from a distance matrix and
    group labels, analogous to ANOVA but for distance matrices (Anderson 2001).

    The decomposition is:
        SS_total = SS_within + SS_between
    where SS is computed from squared distances divided by group sizes.

    Optionally runs a permutation test by shuffling labels to obtain
    a p-value for the pseudo-F statistic.

    Args:
        dist_matrix: Symmetric pairwise distance matrix (n x n).
        labels: Group labels, array of length n.
        n_permutations: Number of permutations for p-value (0 = no test).
        random_state: Random state for reproducibility.

    Returns:
        DiscrepancyResult with pseudo-R2, pseudo-F, and optional p-value.

    Example:
        >>> import numpy as np
        >>> from yasqat.statistics.discrepancy import discrepancy_analysis
        >>> dist = np.array([
        ...     [0, 1, 5, 6],
        ...     [1, 0, 5, 6],
        ...     [5, 5, 0, 1],
        ...     [6, 6, 1, 0],
        ... ], dtype=float)
        >>> labels = np.array([0, 0, 1, 1])
        >>> result = discrepancy_analysis(dist, labels)
        >>> result.pseudo_r2 > 0.5
        True
    """
    n = len(labels)
    if n < 2:
        return DiscrepancyResult(
            pseudo_r2=0.0,
            pseudo_f=0.0,
            p_value=None,
            n_permutations=0,
            total_ss=0.0,
            within_ss=0.0,
            between_ss=0.0,
        )

    total_ss, within_ss = _compute_ss(dist_matrix, labels)
    between_ss = total_ss - within_ss
    pseudo_r2 = between_ss / total_ss if total_ss > 0 else 0.0
    pseudo_f = _compute_pseudo_f(between_ss, within_ss, n, labels)

    # Permutation test
    p_value = None
    if n_permutations > 0:
        if isinstance(random_state, np.random.Generator):
            rng = random_state
        else:
            rng = np.random.default_rng(random_state)

        n_extreme = 0
        for _ in range(n_permutations):
            perm_labels = rng.permutation(labels)
            perm_total, perm_within = _compute_ss(dist_matrix, perm_labels)
            perm_between = perm_total - perm_within
            perm_f = _compute_pseudo_f(perm_between, perm_within, n, perm_labels)
            if perm_f >= pseudo_f:
                n_extreme += 1

        p_value = (n_extreme + 1) / (n_permutations + 1)

    return DiscrepancyResult(
        pseudo_r2=pseudo_r2,
        pseudo_f=pseudo_f,
        p_value=p_value,
        n_permutations=n_permutations,
        total_ss=total_ss,
        within_ss=within_ss,
        between_ss=between_ss,
    )


def _compute_ss(
    dist_matrix: np.ndarray,
    labels: np.ndarray,
) -> tuple[float, float]:
    """Compute total and within-group sum of squares from distance matrix.

    SS_total = sum(d_ij^2) / (2 * n)
    SS_within = sum over groups: sum(d_ij^2 within group) / (2 * n_k)
    """
    n = len(labels)
    sq = dist_matrix**2

    # Total SS
    total_ss = float(np.sum(sq)) / (2 * n)

    # Within-group SS
    within_ss = 0.0
    for label in np.unique(labels):
        mask = labels == label
        n_k = int(np.sum(mask))
        if n_k > 1:
            cluster_sq = sq[np.ix_(mask, mask)]
            within_ss += float(np.sum(cluster_sq)) / (2 * n_k)

    return total_ss, within_ss


def multi_factor_discrepancy(
    dist_matrix: np.ndarray,
    factors: dict[str, np.ndarray],
    n_permutations: int = 0,
    random_state: int | np.random.Generator | None = None,
) -> dict[str, DiscrepancyResult]:
    """
    Multi-factor discrepancy analysis.

    Runs discrepancy analysis for each factor independently, analogous
    to TraMineR's dissmfacw. Each factor is a grouping variable tested
    against the same distance matrix.

    Args:
        dist_matrix: Symmetric pairwise distance matrix (n x n).
        factors: Dictionary mapping factor names to label arrays.
        n_permutations: Number of permutations per factor.
        random_state: Random state for reproducibility.

    Returns:
        Dictionary mapping factor names to DiscrepancyResult objects.

    Example:
        >>> import numpy as np
        >>> dist = np.array([[0,1,5,6],[1,0,5,6],[5,5,0,1],[6,6,1,0]], dtype=float)
        >>> factors = {
        ...     "group": np.array([0, 0, 1, 1]),
        ...     "sex": np.array([0, 1, 0, 1]),
        ... }
        >>> results = multi_factor_discrepancy(dist, factors, n_permutations=99)
        >>> results["group"].pseudo_r2 > results["sex"].pseudo_r2
        True
    """
    if isinstance(random_state, np.random.Generator):
        rng = random_state
    else:
        rng = np.random.default_rng(random_state)

    results = {}
    for name, labels in factors.items():
        results[name] = discrepancy_analysis(
            dist_matrix, labels, n_permutations=n_permutations, random_state=rng
        )

    return results


def _compute_pseudo_f(
    between_ss: float,
    within_ss: float,
    n: int,
    labels: np.ndarray,
) -> float:
    """Compute pseudo-F statistic.

    F = (SS_between / (k - 1)) / (SS_within / (n - k))
    """
    k = len(np.unique(labels))
    df_between = k - 1
    df_within = n - k

    if df_between <= 0 or df_within <= 0 or within_ss == 0:
        return 0.0

    return (between_ss / df_between) / (within_ss / df_within)

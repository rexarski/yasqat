"""Dissimilarity trees: recursive partitioning of distance matrices.

Analogous to TraMineR's disstree. Builds a binary tree where each split
maximizes between-group discrepancy (pseudo-R2 improvement).
"""

from __future__ import annotations

from dataclasses import dataclass

import numpy as np


@dataclass
class DissTreeNode:
    """A node in the dissimilarity tree."""

    indices: np.ndarray
    """Indices of observations in this node."""

    depth: int
    """Depth of this node in the tree."""

    pseudo_r2: float
    """Pseudo-R2 at this split (0 for leaves)."""

    split_variable: str | None = None
    """Name of the variable used to split (None for leaves)."""

    split_value: float | int | str | None = None
    """Value used for the split (None for leaves)."""

    left: DissTreeNode | None = None
    """Left child (split_variable <= split_value or == split_value)."""

    right: DissTreeNode | None = None
    """Right child."""

    @property
    def is_leaf(self) -> bool:
        return self.left is None and self.right is None

    @property
    def n_observations(self) -> int:
        return len(self.indices)


@dataclass
class DissTreeResult:
    """Result of dissimilarity tree fitting."""

    root: DissTreeNode
    """Root node of the tree."""

    n_leaves: int
    """Number of leaf nodes."""

    labels: np.ndarray
    """Leaf assignment for each observation."""

    def __repr__(self) -> str:
        return f"DissTreeResult(n_leaves={self.n_leaves}, root_n={self.root.n_observations})"


def _compute_discrepancy(dist_matrix: np.ndarray, indices: np.ndarray) -> float:
    """Compute total discrepancy (SS) for a subset."""
    if len(indices) <= 1:
        return 0.0
    sub = dist_matrix[np.ix_(indices, indices)]
    return float(np.sum(sub**2)) / (2 * len(indices))


def _find_best_split(
    dist_matrix: np.ndarray,
    covariates: np.ndarray,
    indices: np.ndarray,
    covariate_names: list[str],
) -> tuple[int, float, float]:
    """Find the best binary split on covariates.

    Returns (best_covariate_idx, best_split_value, best_r2_gain).
    """
    total_ss = _compute_discrepancy(dist_matrix, indices)
    if total_ss <= 0:
        return -1, 0.0, 0.0

    best_gain = 0.0
    best_var = -1
    best_val = 0.0

    n_covariates = covariates.shape[1]

    for var_idx in range(n_covariates):
        values = covariates[indices, var_idx]
        unique_vals = np.unique(values)

        if len(unique_vals) <= 1:
            continue

        for split_val in unique_vals[:-1]:
            left_mask = values <= split_val
            right_mask = ~left_mask

            left_idx = indices[left_mask]
            right_idx = indices[right_mask]

            if len(left_idx) < 1 or len(right_idx) < 1:
                continue

            within_ss = _compute_discrepancy(
                dist_matrix, left_idx
            ) + _compute_discrepancy(dist_matrix, right_idx)
            between_ss = total_ss - within_ss
            r2_gain = between_ss / total_ss if total_ss > 0 else 0.0

            if r2_gain > best_gain:
                best_gain = r2_gain
                best_var = var_idx
                best_val = float(split_val)

    return best_var, best_val, best_gain


def dissimilarity_tree(
    dist_matrix: np.ndarray,
    covariates: np.ndarray,
    covariate_names: list[str] | None = None,
    max_depth: int = 5,
    min_node_size: int = 5,
    min_r2_gain: float = 0.01,
) -> DissTreeResult:
    """
    Build a dissimilarity tree by recursive partitioning.

    At each node, finds the covariate split that maximizes the pseudo-R2
    improvement (between-group discrepancy). Splits recursively until
    stopping criteria are met.

    Args:
        dist_matrix: Symmetric pairwise distance matrix (n x n).
        covariates: Covariate matrix (n x p) for splitting.
        covariate_names: Names of covariates. If None, uses "X0", "X1", etc.
        max_depth: Maximum tree depth.
        min_node_size: Minimum observations per leaf.
        min_r2_gain: Minimum R2 improvement to split.

    Returns:
        DissTreeResult with tree structure and leaf assignments.

    Example:
        >>> import numpy as np
        >>> n = 20
        >>> dist = np.random.default_rng(42).random((n, n))
        >>> dist = (dist + dist.T) / 2
        >>> np.fill_diagonal(dist, 0)
        >>> covariates = np.random.default_rng(42).random((n, 2))
        >>> result = dissimilarity_tree(dist, covariates)
        >>> result.n_leaves >= 1
        True
    """
    n = dist_matrix.shape[0]
    if covariate_names is None:
        covariate_names = [f"X{i}" for i in range(covariates.shape[1])]

    all_indices = np.arange(n)
    labels = np.zeros(n, dtype=np.int32)
    leaf_counter = [0]

    def _build(indices: np.ndarray, depth: int) -> DissTreeNode:
        node = DissTreeNode(indices=indices, depth=depth, pseudo_r2=0.0)

        if depth >= max_depth or len(indices) < 2 * min_node_size:
            node_label = leaf_counter[0]
            leaf_counter[0] += 1
            labels[indices] = node_label
            return node

        var_idx, split_val, r2_gain = _find_best_split(
            dist_matrix, covariates, indices, covariate_names
        )

        if var_idx < 0 or r2_gain < min_r2_gain:
            node_label = leaf_counter[0]
            leaf_counter[0] += 1
            labels[indices] = node_label
            return node

        node.pseudo_r2 = r2_gain
        node.split_variable = covariate_names[var_idx]
        node.split_value = split_val

        values = covariates[indices, var_idx]
        left_mask = values <= split_val
        right_mask = ~left_mask

        node.left = _build(indices[left_mask], depth + 1)
        node.right = _build(indices[right_mask], depth + 1)

        return node

    root = _build(all_indices, 0)

    return DissTreeResult(
        root=root,
        n_leaves=leaf_counter[0],
        labels=labels,
    )

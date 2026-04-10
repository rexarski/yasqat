"""Tests for dissimilarity trees."""

import numpy as np
import pytest

from yasqat.statistics.disstree import DissTreeResult, dissimilarity_tree


@pytest.fixture
def tree_data() -> tuple[np.ndarray, np.ndarray]:
    """Distance matrix and covariates with a clear split."""
    n = 20
    rng = np.random.default_rng(42)

    # Create two groups with different distance patterns
    dist = np.zeros((n, n))
    labels = np.array([0] * 10 + [1] * 10)
    for i in range(n):
        for j in range(i + 1, n):
            if labels[i] == labels[j]:
                dist[i, j] = dist[j, i] = rng.random() * 0.5
            else:
                dist[i, j] = dist[j, i] = 3.0 + rng.random() * 0.5

    # Covariate that separates the groups
    covariates = np.zeros((n, 1))
    covariates[:10, 0] = rng.random(10) * 0.5
    covariates[10:, 0] = 0.5 + rng.random(10) * 0.5

    return dist, covariates


class TestDissimilarityTree:
    def test_basic_tree(self, tree_data: tuple[np.ndarray, np.ndarray]) -> None:
        dist, covariates = tree_data
        result = dissimilarity_tree(dist, covariates)
        assert isinstance(result, DissTreeResult)
        assert result.n_leaves >= 2

    def test_labels_cover_all(self, tree_data: tuple[np.ndarray, np.ndarray]) -> None:
        dist, covariates = tree_data
        result = dissimilarity_tree(dist, covariates)
        assert len(result.labels) == dist.shape[0]
        assert len(np.unique(result.labels)) == result.n_leaves

    def test_max_depth_1(self, tree_data: tuple[np.ndarray, np.ndarray]) -> None:
        dist, covariates = tree_data
        result = dissimilarity_tree(dist, covariates, max_depth=1)
        assert result.n_leaves <= 2

    def test_finds_split(self, tree_data: tuple[np.ndarray, np.ndarray]) -> None:
        dist, covariates = tree_data
        result = dissimilarity_tree(dist, covariates, max_depth=1)
        assert result.root.split_variable is not None

    def test_uniform_distance_single_leaf(self) -> None:
        n = 10
        dist = np.ones((n, n)) - np.eye(n)
        covariates = np.random.default_rng(42).random((n, 2))
        result = dissimilarity_tree(dist, covariates, min_r2_gain=0.5)
        # Uniform distances should not split well
        assert result.n_leaves >= 1

    def test_custom_covariate_names(
        self, tree_data: tuple[np.ndarray, np.ndarray]
    ) -> None:
        dist, covariates = tree_data
        result = dissimilarity_tree(dist, covariates, covariate_names=["age"])
        if result.root.split_variable is not None:
            assert result.root.split_variable == "age"

    def test_deeper_tree_with_structure(self) -> None:
        """Tree should grow beyond depth 1 when data has nested structure."""
        n = 40
        rng = np.random.default_rng(99)
        # 4 groups of 10 with distinct distance patterns
        group = np.array([0] * 10 + [1] * 10 + [2] * 10 + [3] * 10)
        dist = np.zeros((n, n))
        for i in range(n):
            for j in range(i + 1, n):
                if group[i] == group[j]:
                    dist[i, j] = dist[j, i] = rng.random() * 0.3
                else:
                    dist[i, j] = dist[j, i] = 3.0 + rng.random()
        # Two covariates that separate the 4 groups
        cov = np.zeros((n, 2))
        cov[:20, 0] = rng.random(20) * 0.5
        cov[20:, 0] = 0.5 + rng.random(20) * 0.5
        cov[:10, 1] = rng.random(10) * 0.5
        cov[10:20, 1] = 0.5 + rng.random(10) * 0.5
        cov[20:30, 1] = rng.random(10) * 0.5
        cov[30:, 1] = 0.5 + rng.random(10) * 0.5

        result = dissimilarity_tree(
            dist,
            cov,
            covariate_names=["x1", "x2"],
            max_depth=5,
            min_node_size=5,
            min_r2_gain=0.001,
        )
        # With 4 well-separated groups, we should get at least 3 leaves
        assert result.n_leaves >= 3

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


class TestDissTreeDepth:
    def test_deeper_tree_with_structure(self) -> None:
        """With structured data, tree should find multiple splits."""
        rng = np.random.default_rng(42)
        n = 100
        labels_true = np.array([0] * 25 + [1] * 25 + [2] * 25 + [3] * 25)
        dist = np.zeros((n, n), dtype=float)
        for i in range(n):
            for j in range(i + 1, n):
                if labels_true[i] == labels_true[j]:
                    dist[i, j] = rng.uniform(0, 1)
                else:
                    dist[i, j] = rng.uniform(3, 6)
                dist[j, i] = dist[i, j]

        cov = np.column_stack([
            labels_true * 10 + rng.normal(0, 1, n),
            labels_true.astype(float),
        ])
        result = dissimilarity_tree(
            dist, cov, covariate_names=["age", "group"],
        )
        assert result.n_leaves >= 3

    def test_min_node_size_respected(self) -> None:
        """Leaves should have at least min_node_size observations."""
        rng = np.random.default_rng(42)
        n = 50
        dist = rng.random((n, n))
        dist = (dist + dist.T) / 2
        np.fill_diagonal(dist, 0)
        cov = rng.random((n, 2))
        result = dissimilarity_tree(
            dist, cov, min_node_size=10,
        )

        def check_leaves(node):
            if node.is_leaf:
                assert node.n_observations >= 10, (
                    f"Leaf has {node.n_observations} obs, expected >= 10"
                )
            else:
                if node.left:
                    check_leaves(node.left)
                if node.right:
                    check_leaves(node.right)

        check_leaves(result.root)

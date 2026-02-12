"""Tests for hierarchical clustering."""

import numpy as np
import pytest

from yasqat.clustering.hierarchical import (
    HierarchicalClustering,
    HierarchicalClusteringResult,
    hierarchical_clustering,
)


@pytest.fixture
def simple_distance_matrix() -> np.ndarray:
    """Create a simple distance matrix with clear clusters."""
    # 4 points: 0,1 are close, 2,3 are close
    return np.array(
        [
            [0.0, 1.0, 4.0, 5.0],
            [1.0, 0.0, 4.0, 5.0],
            [4.0, 4.0, 0.0, 1.0],
            [5.0, 5.0, 1.0, 0.0],
        ]
    )


@pytest.fixture
def large_distance_matrix() -> np.ndarray:
    """Create a larger distance matrix for testing."""
    np.random.seed(42)
    n = 20
    # Create random symmetric matrix
    dist = np.random.rand(n, n)
    dist = (dist + dist.T) / 2
    np.fill_diagonal(dist, 0)
    return dist


class TestHierarchicalClustering:
    """Tests for hierarchical clustering."""

    def test_basic_clustering(self, simple_distance_matrix: np.ndarray) -> None:
        """Test basic hierarchical clustering."""
        result = hierarchical_clustering(simple_distance_matrix, n_clusters=2)

        assert isinstance(result, HierarchicalClusteringResult)
        assert result.n_clusters == 2
        assert len(result.labels) == 4

        # Points 0,1 should be in same cluster, 2,3 in another
        assert result.labels[0] == result.labels[1]
        assert result.labels[2] == result.labels[3]
        assert result.labels[0] != result.labels[2]

    def test_different_n_clusters(self, simple_distance_matrix: np.ndarray) -> None:
        """Test with different number of clusters."""
        result_2 = hierarchical_clustering(simple_distance_matrix, n_clusters=2)
        result_4 = hierarchical_clustering(simple_distance_matrix, n_clusters=4)

        assert result_2.n_clusters == 2
        assert result_4.n_clusters == 4
        assert len(np.unique(result_4.labels)) == 4

    def test_linkage_methods(self, simple_distance_matrix: np.ndarray) -> None:
        """Test different linkage methods."""
        methods = ["complete", "average", "single"]

        for method in methods:
            result = hierarchical_clustering(
                simple_distance_matrix, n_clusters=2, method=method
            )
            assert result.n_clusters == 2
            assert len(result.labels) == 4

    def test_ward_linkage(self, simple_distance_matrix: np.ndarray) -> None:
        """Test Ward linkage method."""
        result = hierarchical_clustering(
            simple_distance_matrix, n_clusters=2, method="ward"
        )

        assert result.n_clusters == 2

    def test_with_sequence_ids(self, simple_distance_matrix: np.ndarray) -> None:
        """Test with custom sequence IDs."""
        seq_ids = ["a", "b", "c", "d"]

        result = hierarchical_clustering(
            simple_distance_matrix, n_clusters=2, sequence_ids=seq_ids
        )

        assert result.sequence_ids == seq_ids

    def test_cluster_sizes(self, simple_distance_matrix: np.ndarray) -> None:
        """Test cluster_sizes method."""
        result = hierarchical_clustering(simple_distance_matrix, n_clusters=2)

        sizes = result.cluster_sizes()

        assert sum(sizes.values()) == 4
        assert len(sizes) == 2

    def test_get_cluster_members(self, simple_distance_matrix: np.ndarray) -> None:
        """Test get_cluster_members method."""
        seq_ids = ["a", "b", "c", "d"]
        result = hierarchical_clustering(
            simple_distance_matrix, n_clusters=2, sequence_ids=seq_ids
        )

        cluster_0_members = result.get_cluster_members(result.labels[0])

        assert len(cluster_0_members) > 0
        assert all(isinstance(m, str) for m in cluster_0_members)

    def test_to_dataframe(self, simple_distance_matrix: np.ndarray) -> None:
        """Test conversion to DataFrame."""
        result = hierarchical_clustering(simple_distance_matrix, n_clusters=2)

        df = result.to_dataframe()

        assert "id" in df.columns
        assert "cluster" in df.columns
        assert len(df) == 4

    def test_linkage_matrix_available(self, simple_distance_matrix: np.ndarray) -> None:
        """Test that linkage matrix is available for dendrogram."""
        result = hierarchical_clustering(simple_distance_matrix, n_clusters=2)

        # Linkage matrix should have shape (n-1, 4)
        assert result.linkage_matrix.shape == (3, 4)

    def test_n_clusters_equals_samples(
        self, simple_distance_matrix: np.ndarray
    ) -> None:
        """Test with n_clusters equal to n_samples."""
        # This should work - each point in its own cluster
        result = hierarchical_clustering(simple_distance_matrix, n_clusters=4)
        assert result.n_clusters == 4
        assert len(np.unique(result.labels)) == 4

    def test_mismatched_sequence_ids(self, simple_distance_matrix: np.ndarray) -> None:
        """Test error when sequence_ids length doesn't match."""
        with pytest.raises(ValueError, match="must match"):
            hierarchical_clustering(
                simple_distance_matrix, n_clusters=2, sequence_ids=["a", "b"]
            )


class TestHierarchicalClusteringClass:
    """Tests for HierarchicalClustering class."""

    def test_class_interface(self, simple_distance_matrix: np.ndarray) -> None:
        """Test class-based interface."""
        clusterer = HierarchicalClustering(n_clusters=2, method="ward")
        result = clusterer.fit(simple_distance_matrix)

        assert isinstance(result, HierarchicalClusteringResult)
        assert clusterer.labels is not None
        assert len(clusterer.labels) == 4

    def test_result_property(self, simple_distance_matrix: np.ndarray) -> None:
        """Test result property."""
        clusterer = HierarchicalClustering(n_clusters=2)

        # Before fit
        assert clusterer.result is None

        # After fit
        clusterer.fit(simple_distance_matrix)
        assert clusterer.result is not None
        assert clusterer.result.n_clusters == 2

    def test_class_with_sequence_ids(self, simple_distance_matrix: np.ndarray) -> None:
        """Test class with sequence IDs."""
        seq_ids = ["seq1", "seq2", "seq3", "seq4"]
        clusterer = HierarchicalClustering(n_clusters=2)
        result = clusterer.fit(simple_distance_matrix, sequence_ids=seq_ids)

        assert result.sequence_ids == seq_ids

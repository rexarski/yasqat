"""Tests for PAM (k-medoids) clustering."""

import numpy as np
import pytest

from yasqat.clustering.pam import PAMClustering, PAMClusteringResult, pam_clustering


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
    """Create a larger distance matrix."""
    np.random.seed(42)
    n = 20
    dist = np.random.rand(n, n) * 10
    dist = (dist + dist.T) / 2
    np.fill_diagonal(dist, 0)
    return dist


class TestPAMClustering:
    """Tests for PAM clustering."""

    def test_basic_clustering(self, simple_distance_matrix: np.ndarray) -> None:
        """Test basic PAM clustering."""
        result = pam_clustering(simple_distance_matrix, n_clusters=2)

        assert isinstance(result, PAMClusteringResult)
        assert result.n_clusters == 2
        assert len(result.labels) == 4
        assert len(result.medoid_indices) == 2

        # Points 0,1 should be in same cluster, 2,3 in another
        assert result.labels[0] == result.labels[1]
        assert result.labels[2] == result.labels[3]
        assert result.labels[0] != result.labels[2]

    def test_medoids_are_actual_points(
        self, simple_distance_matrix: np.ndarray
    ) -> None:
        """Test that medoids are actual data points."""
        result = pam_clustering(simple_distance_matrix, n_clusters=2)

        # Medoids should be valid indices
        assert all(0 <= m < 4 for m in result.medoid_indices)

    def test_initialization_methods(self, simple_distance_matrix: np.ndarray) -> None:
        """Test different initialization methods."""
        inits = ["build", "random", "k-medoids++"]

        for init in inits:
            result = pam_clustering(
                simple_distance_matrix,
                n_clusters=2,
                init=init,
                random_state=42,
            )
            assert result.n_clusters == 2
            assert len(result.medoid_indices) == 2

    def test_deterministic_with_random_state(
        self, large_distance_matrix: np.ndarray
    ) -> None:
        """Test that results are reproducible with random_state."""
        result1 = pam_clustering(
            large_distance_matrix, n_clusters=3, init="random", random_state=42
        )
        result2 = pam_clustering(
            large_distance_matrix, n_clusters=3, init="random", random_state=42
        )

        np.testing.assert_array_equal(result1.labels, result2.labels)
        np.testing.assert_array_equal(result1.medoid_indices, result2.medoid_indices)

    def test_build_initialization_deterministic(
        self, simple_distance_matrix: np.ndarray
    ) -> None:
        """Test that BUILD initialization is deterministic."""
        result1 = pam_clustering(simple_distance_matrix, n_clusters=2, init="build")
        result2 = pam_clustering(simple_distance_matrix, n_clusters=2, init="build")

        np.testing.assert_array_equal(result1.labels, result2.labels)

    def test_with_sequence_ids(self, simple_distance_matrix: np.ndarray) -> None:
        """Test with custom sequence IDs."""
        seq_ids = ["a", "b", "c", "d"]

        result = pam_clustering(
            simple_distance_matrix, n_clusters=2, sequence_ids=seq_ids
        )

        assert result.sequence_ids == seq_ids

    def test_cluster_sizes(self, simple_distance_matrix: np.ndarray) -> None:
        """Test cluster_sizes method."""
        result = pam_clustering(simple_distance_matrix, n_clusters=2)

        sizes = result.cluster_sizes()

        assert sum(sizes.values()) == 4
        assert len(sizes) == 2

    def test_get_cluster_members(self, simple_distance_matrix: np.ndarray) -> None:
        """Test get_cluster_members method."""
        seq_ids = ["a", "b", "c", "d"]
        result = pam_clustering(
            simple_distance_matrix, n_clusters=2, sequence_ids=seq_ids
        )

        members = result.get_cluster_members(result.labels[0])

        assert len(members) > 0
        assert all(isinstance(m, str) for m in members)

    def test_get_medoid_ids(self, simple_distance_matrix: np.ndarray) -> None:
        """Test get_medoid_ids method."""
        seq_ids = ["a", "b", "c", "d"]
        result = pam_clustering(
            simple_distance_matrix, n_clusters=2, sequence_ids=seq_ids
        )

        medoid_ids = result.get_medoid_ids()

        assert len(medoid_ids) == 2
        assert all(m in seq_ids for m in medoid_ids)

    def test_to_dataframe(self, simple_distance_matrix: np.ndarray) -> None:
        """Test conversion to DataFrame."""
        result = pam_clustering(simple_distance_matrix, n_clusters=2)

        df = result.to_dataframe()

        assert "id" in df.columns
        assert "cluster" in df.columns
        assert "is_medoid" in df.columns
        assert len(df) == 4
        assert df["is_medoid"].sum() == 2  # Two medoids

    def test_total_cost(self, simple_distance_matrix: np.ndarray) -> None:
        """Test that total_cost is computed."""
        result = pam_clustering(simple_distance_matrix, n_clusters=2)

        assert result.total_cost >= 0
        # With good clustering, cost should be relatively low
        # Each point within cluster has distance 1 to medoid
        assert result.total_cost <= 4.0

    def test_n_iterations(self, simple_distance_matrix: np.ndarray) -> None:
        """Test that n_iterations is tracked."""
        result = pam_clustering(simple_distance_matrix, n_clusters=2)

        assert result.n_iterations >= 1

    def test_max_iter_limit(self, large_distance_matrix: np.ndarray) -> None:
        """Test max_iter parameter."""
        result = pam_clustering(large_distance_matrix, n_clusters=3, max_iter=1)

        # Should stop after 1 iteration
        assert result.n_iterations <= 2  # Initial + 1 swap attempt

    def test_too_many_clusters(self, simple_distance_matrix: np.ndarray) -> None:
        """Test error when n_clusters > n_samples."""
        with pytest.raises(ValueError, match="cannot exceed"):
            pam_clustering(simple_distance_matrix, n_clusters=10)

    def test_mismatched_sequence_ids(self, simple_distance_matrix: np.ndarray) -> None:
        """Test error when sequence_ids length doesn't match."""
        with pytest.raises(ValueError, match="must match"):
            pam_clustering(
                simple_distance_matrix, n_clusters=2, sequence_ids=["a", "b"]
            )


class TestPAMClusteringClass:
    """Tests for PAMClustering class."""

    def test_class_interface(self, simple_distance_matrix: np.ndarray) -> None:
        """Test class-based interface."""
        clusterer = PAMClustering(n_clusters=2, init="build")
        result = clusterer.fit(simple_distance_matrix)

        assert isinstance(result, PAMClusteringResult)
        assert clusterer.labels is not None
        assert clusterer.medoid_indices is not None

    def test_result_property(self, simple_distance_matrix: np.ndarray) -> None:
        """Test result property."""
        clusterer = PAMClustering(n_clusters=2)

        # Before fit
        assert clusterer.result is None
        assert clusterer.labels is None
        assert clusterer.medoid_indices is None

        # After fit
        clusterer.fit(simple_distance_matrix)
        assert clusterer.result is not None

    def test_class_with_all_options(self, large_distance_matrix: np.ndarray) -> None:
        """Test class with all options."""
        clusterer = PAMClustering(
            n_clusters=3,
            max_iter=50,
            init="k-medoids++",
            random_state=123,
        )
        result = clusterer.fit(large_distance_matrix)

        assert result.n_clusters == 3
        assert len(result.medoid_indices) == 3

"""Tests for choosing k (yasqat.clustering.k_selection)."""

import numpy as np
import pytest

from yasqat.clustering import k_range, pam_range


class TestPAMRange:
    """Tests for pam_range function."""

    def test_returns_all_k(self) -> None:
        dist = np.array(
            [
                [0.0, 1.0, 5.0, 6.0, 5.0],
                [1.0, 0.0, 5.0, 6.0, 5.0],
                [5.0, 5.0, 0.0, 1.0, 0.5],
                [6.0, 6.0, 1.0, 0.0, 1.5],
                [5.0, 5.0, 0.5, 1.5, 0.0],
            ]
        )
        results = pam_range(dist, k_range=range(2, 5))
        assert set(results.keys()) == {2, 3, 4}

    def test_contains_quality_metrics(self) -> None:
        dist = np.array(
            [
                [0.0, 1.0, 5.0, 6.0],
                [1.0, 0.0, 5.0, 6.0],
                [5.0, 5.0, 0.0, 1.0],
                [6.0, 6.0, 1.0, 0.0],
            ]
        )
        results = pam_range(dist, k_range=[2])
        assert "ASW" in results[2]
        assert "PBC" in results[2]
        assert "HG" in results[2]
        assert "R2" in results[2]
        assert "total_cost" in results[2]

    def test_default_k_range(self) -> None:
        dist = np.array(
            [
                [0.0, 1.0, 5.0, 6.0],
                [1.0, 0.0, 5.0, 6.0],
                [5.0, 5.0, 0.0, 1.0],
                [6.0, 6.0, 1.0, 0.0],
            ]
        )
        results = pam_range(dist)
        # Default is range(2, min(n, 11)) = range(2, 4) for n=4
        assert set(results.keys()) == {2, 3}

    def test_skips_invalid_k(self) -> None:
        dist = np.array([[0.0, 1.0], [1.0, 0.0]])
        results = pam_range(dist, k_range=[1, 2, 5])
        # k=1 is <2, k=5 is >=n, so only k=2 would be valid but k=2>=n=2 also skipped
        assert len(results) == 0

    def test_two_tuple_is_inclusive_with_warning(self) -> None:
        """Regression (v0.3.2 hot-fix D2): a 2-tuple is treated as (start, end)
        inclusive and triggers a DeprecationWarning — previously only the two
        endpoints were iterated, silently returning only k=start and k=end.
        """
        import warnings

        dist = np.array(
            [
                [0.0, 1.0, 5.0, 6.0, 5.0],
                [1.0, 0.0, 5.0, 6.0, 5.0],
                [5.0, 5.0, 0.0, 1.0, 0.5],
                [6.0, 6.0, 1.0, 0.0, 1.5],
                [5.0, 5.0, 0.5, 1.5, 0.0],
            ]
        )
        with warnings.catch_warnings(record=True) as caught:
            warnings.simplefilter("always")
            results = pam_range(dist, k_values=(2, 4))

        assert set(results.keys()) == {2, 3, 4}
        assert any(
            issubclass(w.category, DeprecationWarning) and "2-tuple" in str(w.message)
            for w in caught
        )

    def test_k_range_helper_is_inclusive(self) -> None:
        """k_range(a, b) returns range(a, b+1) — inclusive on both ends."""
        assert list(k_range(2, 5)) == [2, 3, 4, 5]
        assert list(k_range(3, 3)) == [3]

    def test_k_values_and_k_range_together_raise(self) -> None:
        """Passing both ``k_values`` and legacy ``k_range`` is an error."""
        dist = np.array([[0.0, 1.0], [1.0, 0.0]])
        with pytest.raises(TypeError, match="both 'k_values' and 'k_range'"):
            pam_range(dist, k_values=[2], k_range=[2])


class TestPamRangeDistanceMatrix:
    def test_accepts_distance_matrix_object(self) -> None:
        """pam_range should accept a DistanceMatrix, not just np.ndarray."""
        from yasqat.metrics.base import DistanceMatrix

        values = np.array(
            [
                [0, 1, 5, 6],
                [1, 0, 5, 6],
                [5, 5, 0, 1],
                [6, 6, 1, 0],
            ],
            dtype=np.float64,
        )
        dm = DistanceMatrix(values=values, labels=[0, 1, 2, 3])
        result = pam_range(dm, k_range=[2, 3])
        assert 2 in result
        assert 3 in result
        assert "ASW" in result[2]

    def test_pam_range_with_distance_matrix(self) -> None:
        from yasqat.metrics.base import DistanceMatrix

        dist = np.array(
            [[0, 1, 5, 6], [1, 0, 5, 6], [5, 5, 0, 1], [6, 6, 1, 0]],
            dtype=np.float64,
        )
        dm = DistanceMatrix(values=dist)

        from_dm = pam_range(dm, k_values=[2])
        from_array = pam_range(dist, k_values=[2])

        assert from_dm[2]["ASW"] == pytest.approx(from_array[2]["ASW"])

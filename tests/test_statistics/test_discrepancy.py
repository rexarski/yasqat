"""Tests for discrepancy analysis (pseudo-ANOVA)."""

import numpy as np
import pytest

from yasqat.statistics.discrepancy import (
    DiscrepancyResult,
    discrepancy_analysis,
    multi_factor_discrepancy,
)


@pytest.fixture
def well_separated() -> tuple[np.ndarray, np.ndarray]:
    """Distance matrix with well-separated groups."""
    dist = np.array(
        [
            [0.0, 1.0, 5.0, 6.0],
            [1.0, 0.0, 5.0, 6.0],
            [5.0, 5.0, 0.0, 1.0],
            [6.0, 6.0, 1.0, 0.0],
        ]
    )
    labels = np.array([0, 0, 1, 1])
    return dist, labels


@pytest.fixture
def uniform_dist() -> tuple[np.ndarray, np.ndarray]:
    """Uniform distance matrix (no group structure)."""
    dist = np.array(
        [
            [0.0, 1.0, 1.0, 1.0],
            [1.0, 0.0, 1.0, 1.0],
            [1.0, 1.0, 0.0, 1.0],
            [1.0, 1.0, 1.0, 0.0],
        ]
    )
    labels = np.array([0, 0, 1, 1])
    return dist, labels


class TestDiscrepancyAnalysis:
    """Tests for discrepancy_analysis function."""

    def test_well_separated_high_r2(
        self, well_separated: tuple[np.ndarray, np.ndarray]
    ) -> None:
        dist, labels = well_separated
        result = discrepancy_analysis(dist, labels)
        assert result.pseudo_r2 > 0.5

    def test_well_separated_high_f(
        self, well_separated: tuple[np.ndarray, np.ndarray]
    ) -> None:
        dist, labels = well_separated
        result = discrepancy_analysis(dist, labels)
        assert result.pseudo_f > 1.0

    def test_uniform_low_r2(self, uniform_dist: tuple[np.ndarray, np.ndarray]) -> None:
        dist, labels = uniform_dist
        result = discrepancy_analysis(dist, labels)
        # Uniform distances with few points still show some R2 from SS decomposition
        # but it should be much lower than well-separated data
        assert result.pseudo_r2 < 0.5

    def test_ss_decomposition(
        self, well_separated: tuple[np.ndarray, np.ndarray]
    ) -> None:
        dist, labels = well_separated
        result = discrepancy_analysis(dist, labels)
        assert result.total_ss == pytest.approx(result.within_ss + result.between_ss)

    def test_r2_in_range(self, well_separated: tuple[np.ndarray, np.ndarray]) -> None:
        dist, labels = well_separated
        result = discrepancy_analysis(dist, labels)
        assert 0.0 <= result.pseudo_r2 <= 1.0

    def test_no_permutation_p_value_none(
        self, well_separated: tuple[np.ndarray, np.ndarray]
    ) -> None:
        dist, labels = well_separated
        result = discrepancy_analysis(dist, labels, n_permutations=0)
        assert result.p_value is None
        assert result.n_permutations == 0


class TestPermutationTest:
    """Tests for permutation test."""

    def test_permutation_returns_p_value(
        self, well_separated: tuple[np.ndarray, np.ndarray]
    ) -> None:
        dist, labels = well_separated
        result = discrepancy_analysis(dist, labels, n_permutations=99, random_state=42)
        assert result.p_value is not None
        assert 0.0 < result.p_value <= 1.0

    def test_well_separated_small_p(self) -> None:
        # Use larger dataset so permutations are meaningful
        n = 20
        dist = np.zeros((n, n))
        labels = np.array([0] * 10 + [1] * 10)
        for i in range(n):
            for j in range(i + 1, n):
                if labels[i] == labels[j]:
                    dist[i, j] = dist[j, i] = 0.5
                else:
                    dist[i, j] = dist[j, i] = 5.0
        result = discrepancy_analysis(dist, labels, n_permutations=99, random_state=42)
        assert result.p_value is not None
        assert result.p_value < 0.05

    def test_uniform_large_p(self, uniform_dist: tuple[np.ndarray, np.ndarray]) -> None:
        dist, labels = uniform_dist
        result = discrepancy_analysis(dist, labels, n_permutations=99, random_state=42)
        assert result.p_value is not None
        # Uniform distances: no real group structure
        assert result.p_value > 0.1

    def test_reproducible(self, well_separated: tuple[np.ndarray, np.ndarray]) -> None:
        dist, labels = well_separated
        r1 = discrepancy_analysis(dist, labels, n_permutations=50, random_state=123)
        r2 = discrepancy_analysis(dist, labels, n_permutations=50, random_state=123)
        assert r1.p_value == r2.p_value


class TestEdgeCases:
    """Tests for edge cases."""

    def test_single_observation(self) -> None:
        dist = np.array([[0.0]])
        labels = np.array([0])
        result = discrepancy_analysis(dist, labels)
        assert result.pseudo_r2 == 0.0
        assert result.pseudo_f == 0.0

    def test_single_group(self) -> None:
        dist = np.array([[0.0, 1.0], [1.0, 0.0]])
        labels = np.array([0, 0])
        result = discrepancy_analysis(dist, labels)
        assert result.pseudo_f == 0.0

    def test_repr(self) -> None:
        result = DiscrepancyResult(
            pseudo_r2=0.75,
            pseudo_f=12.5,
            p_value=0.01,
            n_permutations=99,
            total_ss=10.0,
            within_ss=2.5,
            between_ss=7.5,
        )
        s = repr(result)
        assert "0.7500" in s
        assert "12.5000" in s
        assert "0.0100" in s

    def test_repr_no_pvalue(self) -> None:
        result = DiscrepancyResult(
            pseudo_r2=0.5,
            pseudo_f=5.0,
            p_value=None,
            n_permutations=0,
            total_ss=10.0,
            within_ss=5.0,
            between_ss=5.0,
        )
        assert "N/A" in repr(result)


class TestMultiFactorDiscrepancy:
    """Tests for multi-factor discrepancy analysis."""

    def test_returns_dict(self, well_separated: tuple[np.ndarray, np.ndarray]) -> None:
        dist, labels = well_separated
        factors = {"group": labels, "other": np.array([0, 1, 0, 1])}
        results = multi_factor_discrepancy(dist, factors)
        assert isinstance(results, dict)
        assert "group" in results
        assert "other" in results
        assert isinstance(results["group"], DiscrepancyResult)

    def test_well_separated_factor_high_r2(
        self, well_separated: tuple[np.ndarray, np.ndarray]
    ) -> None:
        dist, labels = well_separated
        factors = {"group": labels}
        results = multi_factor_discrepancy(dist, factors)
        assert results["group"].pseudo_r2 > 0.5

    def test_with_permutations(
        self, well_separated: tuple[np.ndarray, np.ndarray]
    ) -> None:
        dist, labels = well_separated
        factors = {"group": labels}
        results = multi_factor_discrepancy(
            dist, factors, n_permutations=50, random_state=42
        )
        assert results["group"].p_value is not None

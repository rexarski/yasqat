"""Tests for frequent subsequence mining."""

import math

import polars as pl
import pytest

from yasqat.core.pool import SequencePool
from yasqat.statistics.subsequence_mining import (
    association_rules,
    frequent_subsequences,
)


@pytest.fixture
def mining_pool() -> SequencePool:
    """Pool with clear frequent patterns."""
    data = pl.DataFrame(
        {
            "id": [1, 1, 1, 2, 2, 2, 3, 3, 3],
            "time": [0, 1, 2, 0, 1, 2, 0, 1, 2],
            "state": [
                "A",
                "B",
                "C",  # seq 1
                "A",
                "B",
                "C",  # seq 2
                "A",
                "C",
                "B",  # seq 3
            ],
        }
    )
    return SequencePool(data)


class TestFrequentSubsequences:
    def test_returns_dataframe(self, mining_pool: SequencePool) -> None:
        results = frequent_subsequences(mining_pool, min_support=0.5)
        assert isinstance(results, pl.DataFrame)
        assert "subsequence" in results.columns
        assert "support" in results.columns
        assert "proportion" in results.columns

    def test_all_states_frequent(self, mining_pool: SequencePool) -> None:
        results = frequent_subsequences(mining_pool, min_support=0.5)
        # Extract single-state patterns
        single_state = results.filter(pl.col("subsequence").list.len() == 1)
        patterns = single_state["subsequence"].to_list()
        assert ["A"] in patterns
        assert ["B"] in patterns
        assert ["C"] in patterns

    def test_ab_frequent(self, mining_pool: SequencePool) -> None:
        results = frequent_subsequences(mining_pool, min_support=0.5)
        patterns = results["subsequence"].to_list()
        # "A", "B" subsequence appears in all 3 sequences
        assert ["A", "B"] in patterns

    def test_support_values(self, mining_pool: SequencePool) -> None:
        results = frequent_subsequences(mining_pool, min_support=0.5)
        # min_count = max(1, int(0.5 * 3)) = 1, so support >= 1
        for prop in results["proportion"].to_list():
            assert 0.0 < prop <= 1.0
        # "A" appears in all 3 sequences -> support count must be exactly 3
        single_a = results.filter(pl.col("subsequence").list.len() == 1)
        a_row = single_a.filter(pl.col("subsequence").list.get(0) == "A")
        assert len(a_row) == 1
        assert a_row["support"][0] == 3
        assert a_row["proportion"][0] == pytest.approx(1.0)

    def test_max_length(self, mining_pool: SequencePool) -> None:
        results = frequent_subsequences(mining_pool, min_support=0.5, max_length=1)
        for subseq in results["subsequence"].to_list():
            assert len(subseq) == 1

    def test_high_support_filters(self, mining_pool: SequencePool) -> None:
        results = frequent_subsequences(mining_pool, min_support=1.0)
        # Only patterns in ALL sequences
        for support in results["support"].to_list():
            assert support == 3

    def test_sorted_by_support(self, mining_pool: SequencePool) -> None:
        results = frequent_subsequences(mining_pool, min_support=0.3)
        supports = results["support"].to_list()
        assert supports == sorted(supports, reverse=True)

    def test_filter_by_length(self, mining_pool: SequencePool) -> None:
        """Test that results can be filtered using polars expressions."""
        results = frequent_subsequences(mining_pool, min_support=0.3)
        two_step = results.filter(pl.col("subsequence").list.len() == 2)
        assert isinstance(two_step, pl.DataFrame)

    def test_min_length(self, mining_pool: SequencePool) -> None:
        """Test min_length excludes short patterns from results."""
        all_results = frequent_subsequences(mining_pool, min_support=0.3, min_length=1)
        filtered = frequent_subsequences(mining_pool, min_support=0.3, min_length=2)
        # min_length=2 should have fewer results (no single-state patterns)
        assert len(filtered) < len(all_results)
        for subseq in filtered["subsequence"].to_list():
            assert len(subseq) >= 2

    def test_min_length_filtering_removes_singletons(
        self, mining_pool: SequencePool
    ) -> None:
        """Verify min_length=2 removes all single-state subsequences."""
        unfiltered = frequent_subsequences(mining_pool, min_support=0.3, min_length=1)
        filtered = frequent_subsequences(mining_pool, min_support=0.3, min_length=2)
        n_singletons = len(unfiltered.filter(pl.col("subsequence").list.len() == 1))
        # All singletons should be removed; the count difference is at least n_singletons
        assert n_singletons > 0
        assert len(unfiltered) - len(filtered) >= n_singletons
        # No single-element subsequences remain
        assert len(filtered.filter(pl.col("subsequence").list.len() == 1)) == 0


@pytest.fixture
def rules_pool() -> SequencePool:
    """Four sequences with hand-computable rule measures.

    Sequences: (A B C), (A B D), (A B C), (A C D). Ordered-subsequence
    support counts at n=4: A=4, B=3, C=3, D=2, (A,B)=3, (B,C)=2.
    """
    data = pl.DataFrame(
        {
            "id": [1, 1, 1, 2, 2, 2, 3, 3, 3, 4, 4, 4],
            "time": [0, 1, 2, 0, 1, 2, 0, 1, 2, 0, 1, 2],
            "state": [
                "A",
                "B",
                "C",  # seq 1
                "A",
                "B",
                "D",  # seq 2
                "A",
                "B",
                "C",  # seq 3
                "A",
                "C",
                "D",  # seq 4
            ],
        }
    )
    return SequencePool(data)


class TestAssociationRules:
    def test_returns_dataframe_with_measure_columns(
        self, rules_pool: SequencePool
    ) -> None:
        rules = association_rules(rules_pool, min_support=0.5)
        assert isinstance(rules, pl.DataFrame)
        for col in (
            "antecedent",
            "consequent",
            "support",
            "confidence",
            "lift",
            "leverage",
            "conviction",
        ):
            assert col in rules.columns

    def test_bc_rule_measures(self, rules_pool: SequencePool) -> None:
        """Pin every measure for the rule B => C on the hand-computed pool."""
        rules = association_rules(rules_pool, min_support=0.5)
        bc = [
            r
            for r in rules.to_dicts()
            if r["antecedent"] == ["B"] and r["consequent"] == ["C"]
        ]
        assert len(bc) == 1
        rule = bc[0]
        # support(BC)=2/4, support(B)=3/4, support(C)=3/4
        assert rule["support"] == pytest.approx(0.5)
        assert rule["confidence"] == pytest.approx(2 / 3)
        assert rule["lift"] == pytest.approx(8 / 9)
        assert rule["leverage"] == pytest.approx(-0.0625)
        assert rule["conviction"] == pytest.approx(0.75)

    def test_antecedent_and_consequent_are_nonempty(
        self, rules_pool: SequencePool
    ) -> None:
        rules = association_rules(rules_pool, min_support=0.5)
        for r in rules.to_dicts():
            assert len(r["antecedent"]) >= 1
            assert len(r["consequent"]) >= 1

    def test_support_and_confidence_in_unit_range(
        self, rules_pool: SequencePool
    ) -> None:
        rules = association_rules(rules_pool, min_support=0.5)
        for r in rules.to_dicts():
            assert 0.0 <= r["support"] <= 1.0
            assert 0.0 <= r["confidence"] <= 1.0

    def test_min_confidence_filters(self, rules_pool: SequencePool) -> None:
        loose = association_rules(rules_pool, min_support=0.5, min_confidence=0.0)
        strict = association_rules(rules_pool, min_support=0.5, min_confidence=0.9)
        assert len(strict) < len(loose)
        for r in strict.to_dicts():
            assert r["confidence"] >= 0.9

    def test_sorted_by_confidence_descending(self, rules_pool: SequencePool) -> None:
        rules = association_rules(rules_pool, min_support=0.5)
        confidences = rules["confidence"].to_list()
        assert confidences == sorted(confidences, reverse=True)

    def test_conviction_is_infinite_for_certain_rule(self) -> None:
        """A rule with confidence 1.0 has infinite conviction."""
        data = pl.DataFrame(
            {
                "id": [1, 1, 2, 2],
                "time": [0, 1, 0, 1],
                "state": ["X", "Y", "X", "Y"],
            }
        )
        rules = association_rules(SequencePool(data), min_support=0.5)
        xy = [
            r
            for r in rules.to_dicts()
            if r["antecedent"] == ["X"] and r["consequent"] == ["Y"]
        ]
        assert len(xy) == 1
        assert xy[0]["confidence"] == pytest.approx(1.0)
        assert math.isinf(xy[0]["conviction"])

    def test_accepts_state_sequence(self, state_sequence) -> None:  # type: ignore[no-untyped-def]
        """The SequenceData coercion seam accepts a StateSequence too."""
        rules = association_rules(state_sequence, min_support=0.3)
        assert isinstance(rules, pl.DataFrame)

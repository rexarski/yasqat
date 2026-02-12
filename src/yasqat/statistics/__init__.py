"""Descriptive statistics for sequences."""

from yasqat.statistics.descriptive import (
    complexity_index,
    longitudinal_entropy,
    mean_time_in_state,
    modal_states,
    normalized_turbulence,
    sequence_frequency_table,
    sequence_length,
    sequence_log_probability,
    spell_count,
    state_distribution,
    subsequence_count,
    transition_count,
    transition_proportion,
    turbulence,
    visited_proportion,
    visited_states,
)
from yasqat.statistics.discrepancy import discrepancy_analysis, multi_factor_discrepancy
from yasqat.statistics.disstree import dissimilarity_tree
from yasqat.statistics.normative import (
    badness,
    degradation,
    insecurity,
    integration,
    precarity,
    proportion_positive,
    volatility,
)
from yasqat.statistics.subsequence_mining import frequent_subsequences
from yasqat.statistics.transition import (
    first_occurrence_time,
    state_duration_stats,
    substitution_cost_matrix,
    transition_rate_dataframe,
    transition_rate_matrix,
)

__all__ = [
    "badness",
    "complexity_index",
    "degradation",
    "discrepancy_analysis",
    "dissimilarity_tree",
    "first_occurrence_time",
    "frequent_subsequences",
    "insecurity",
    "integration",
    "longitudinal_entropy",
    "mean_time_in_state",
    "modal_states",
    "multi_factor_discrepancy",
    "normalized_turbulence",
    "precarity",
    "proportion_positive",
    "sequence_frequency_table",
    "sequence_length",
    "sequence_log_probability",
    "spell_count",
    "state_distribution",
    "state_duration_stats",
    "subsequence_count",
    "substitution_cost_matrix",
    "transition_count",
    "transition_proportion",
    "transition_rate_dataframe",
    "transition_rate_matrix",
    "turbulence",
    "visited_proportion",
    "visited_states",
    "volatility",
]

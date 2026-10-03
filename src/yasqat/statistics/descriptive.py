"""Descriptive statistics for sequences."""

from __future__ import annotations

import math
from collections import Counter
from typing import TYPE_CHECKING, cast

import numpy as np
import polars as pl

from yasqat.core.pool import SequencePool
from yasqat.statistics._reduce import reduce_per_sequence

if TYPE_CHECKING:
    from yasqat.core.protocols import SequenceData


def _n_transitions(states: list[str]) -> int:
    """Number of adjacent state changes in one sequence."""
    return sum(1 for i in range(len(states) - 1) if states[i] != states[i + 1])


def longitudinal_entropy(
    sequence: SequenceData,
    normalize: bool = True,
    per_sequence: bool = False,
) -> float | pl.DataFrame:
    """
    Calculate within-sequence entropy.

    Measures the diversity of states visited by each sequence.
    Higher entropy indicates more diverse state usage.

    Formula: H(s) = -Σ p_a * log(p_a)
    where p_a is the proportion of time spent in state a.

    Args:
        sequence: StateSequence or SequencePool.
        normalize: If True, normalize by maximum possible entropy.
        per_sequence: If True, return entropy for each sequence.

    Returns:
        If per_sequence=False: Mean entropy across all sequences.
        If per_sequence=True: DataFrame with sequence IDs and entropies.

    Example:
        >>> entropy = longitudinal_entropy(seq, normalize=True)
    """
    pool = SequencePool.coerce(sequence)

    config = pool.config
    n_states = len(pool.alphabet)
    id_col = config.id_column
    state_col = config.state_column
    data = pool.data

    # Vectorized entropy computation using polars
    state_counts = data.group_by([id_col, state_col]).agg(pl.len().alias("count"))
    seq_lengths = data.group_by(id_col).agg(pl.len().alias("_total"))

    entropy_df = (
        state_counts.join(seq_lengths, on=id_col)
        .with_columns((pl.col("count") / pl.col("_total")).alias("p"))
        .with_columns((-pl.col("p") * pl.col("p").log()).alias("_h_term"))
        .group_by(id_col)
        .agg(pl.col("_h_term").sum().alias("entropy"))
    )

    if normalize and n_states > 1:
        max_entropy = float(np.log(n_states))
        if max_entropy > 0:
            entropy_df = entropy_df.with_columns(
                (pl.col("entropy") / max_entropy).alias("entropy")
            )

    entropy_df = entropy_df.sort(id_col)

    if per_sequence:
        return entropy_df

    return cast(float, entropy_df["entropy"].mean())


def transition_count(
    sequence: SequenceData,
    per_sequence: bool = False,
) -> int | pl.DataFrame:
    """
    Count the number of state transitions.

    A transition occurs when the state changes from one time point
    to the next.

    Args:
        sequence: StateSequence or SequencePool.
        per_sequence: If True, return counts for each sequence.

    Returns:
        If per_sequence=False: Total number of transitions.
        If per_sequence=True: DataFrame with sequence IDs and counts.
    """
    result = reduce_per_sequence(
        sequence, _n_transitions, "n_transitions", per_sequence, aggregate="sum"
    )
    # The summed counts are an int at runtime; the reduce types them float.
    return cast("int | pl.DataFrame", result)


def sequence_length(
    sequence: SequenceData,
    per_sequence: bool = False,
) -> int | float | pl.DataFrame:
    """
    Get sequence length(s).

    Args:
        sequence: StateSequence or SequencePool.
        per_sequence: If True, return length for each sequence.

    Returns:
        If per_sequence=False: Mean sequence length.
        If per_sequence=True: DataFrame with sequence IDs and lengths.
    """
    return reduce_per_sequence(sequence, len, "length", per_sequence)


def complexity_index(
    sequence: SequenceData,
    per_sequence: bool = False,
) -> float | pl.DataFrame:
    """
    Calculate the complexity index (Gabadinho et al., 2010; TraMineR ``seqici``).

    The index is the geometric mean of the normalised number of transitions
    and the normalised longitudinal entropy:

        ``C(x) = sqrt( (n_transitions(x) / (l - 1)) * (H(x) / log(k)) )``

    where ``l`` is the sequence length, ``H`` the Shannon entropy of the
    state distribution within the sequence, and ``k`` the number of states
    in the pool's alphabet. It lies in [0, 1]; a sequence of length 1, or a one-state
    alphabet, scores 0.

    Args:
        sequence: StateSequence or SequencePool.
        per_sequence: If True, return complexity for each sequence.

    Returns:
        If per_sequence=False: Mean complexity across sequences.
        If per_sequence=True: DataFrame with sequence IDs and complexity.
    """
    pool = SequencePool.coerce(sequence)
    h_max = math.log(len(pool.alphabet)) if len(pool.alphabet) > 1 else 0.0

    def _complexity(states: list[str]) -> float:
        n = len(states)
        if n <= 1 or h_max == 0.0:
            return 0.0
        counts = Counter(states)
        entropy = -sum((c / n) * math.log(c / n) for c in counts.values())
        return math.sqrt((_n_transitions(states) / (n - 1)) * (entropy / h_max))

    return reduce_per_sequence(pool, _complexity, "complexity", per_sequence)


def _spell_durations(states: list[str]) -> list[int]:
    """Run-length encode ``states`` and return the spell durations in order."""
    durations: list[int] = []
    for i, s in enumerate(states):
        if i > 0 and s == states[i - 1]:
            durations[-1] += 1
        else:
            durations.append(1)
    return durations


def _to_dss(states: list[str]) -> list[str]:
    """Collapse consecutive repeats: the distinct-successive-states form."""
    return [s for i, s in enumerate(states) if i == 0 or s != states[i - 1]]


def _n_distinct_subsequences(states: list[str]) -> int:
    """Number of distinct subsequences of ``states`` *including* the empty one.

    Standard DP: ``dp[i] = 2 * dp[i-1] - dp[last[c] - 1]`` where ``last[c]`` is
    the previous position of the character just appended. Python big ints
    handle the exponential growth.
    """
    dp = [0] * (len(states) + 1)
    dp[0] = 1
    last_seen: dict[str, int] = {}
    for i in range(1, len(states) + 1):
        dp[i] = 2 * dp[i - 1]
        c = states[i - 1]
        if c in last_seen:
            dp[i] -= dp[last_seen[c] - 1]
        last_seen[c] = i
    return dp[len(states)]


def _turbulence_of(states: list[str]) -> float:
    """Elzinga & Liefbroer (2007) turbulence of one sequence."""
    durations = _spell_durations(states)
    n = len(durations)
    if n == 0:
        return 0.0
    phi = _n_distinct_subsequences(_to_dss(states))
    t_bar = sum(durations) / n
    s2 = sum((d - t_bar) ** 2 for d in durations) / n
    s2_max = (n - 1) * (1 - t_bar) ** 2
    return math.log2(phi) + math.log2((s2_max + 1) / (s2 + 1))


def turbulence(
    sequence: SequenceData,
    per_sequence: bool = False,
) -> float | pl.DataFrame:
    """
    Calculate the turbulence index (Elzinga & Liefbroer, 2007; TraMineR ``seqST``).

        ``T(x) = log2( phi(x) * (s2_max(x) + 1) / (s2(x) + 1) )``

    where ``phi(x)`` is the number of distinct subsequences of the
    distinct-successive-states form of ``x`` (the empty subsequence
    included), ``s2(x)`` the variance of the spell durations, and
    ``s2_max(x) = (n - 1) * (1 - t_bar)^2`` the maximum that variance can
    take for ``n`` spells with mean duration ``t_bar``. The variance uses the
    population denominator ``n``, which is the convention under which the
    ``s2_max`` bound holds. A sequence that never changes state has ``phi = 2``
    and scores exactly 1; the index grows with both the number of state
    changes and the irregularity of spell durations.

    Args:
        sequence: StateSequence or SequencePool.
        per_sequence: If True, return turbulence for each sequence.

    Returns:
        If per_sequence=False: Mean turbulence across sequences.
        If per_sequence=True: DataFrame with sequence IDs and turbulence.
    """
    return reduce_per_sequence(sequence, _turbulence_of, "turbulence", per_sequence)


def state_distribution(
    sequence: SequenceData,
    time_point: int | None = None,
    per_sequence: bool = False,
) -> pl.DataFrame:
    """
    Calculate state distribution (cross-sectional or overall).

    Args:
        sequence: StateSequence or SequencePool.
        time_point: If provided, calculate distribution at specific time.
                   If None, calculate overall distribution.
        per_sequence: If True, return distribution within each sequence.

    Returns:
        DataFrame with states and their frequencies/proportions.
    """
    data = sequence.data
    config = sequence.config

    if time_point is not None:
        data = data.filter(pl.col(config.time_column) == time_point)

    if per_sequence:
        per_seq_counts = data.group_by([config.id_column, config.state_column]).agg(
            pl.len().alias("count")
        )
        per_seq_total = data.group_by(config.id_column).agg(pl.len().alias("total"))
        return (
            per_seq_counts.join(per_seq_total, on=config.id_column)
            .with_columns((pl.col("count") / pl.col("total")).alias("proportion"))
            .drop("total")
            .sort([config.id_column, config.state_column])
        )

    counts = (
        data.group_by(config.state_column)
        .agg(pl.len().alias("count"))
        .sort("count", descending=True)
    )

    total = counts["count"].sum()
    counts = counts.with_columns((pl.col("count") / total).alias("proportion"))

    return counts


def mean_time_in_state(
    sequence: SequenceData,
    per_sequence: bool = False,
) -> pl.DataFrame:
    """
    Calculate mean time spent in each state.

    Args:
        sequence: StateSequence or SequencePool.
        per_sequence: If True, return time-in-state for each sequence.

    Returns:
        If per_sequence=False: DataFrame with states and mean time across sequences.
        If per_sequence=True: DataFrame with sequence IDs, states, and counts.
    """
    data = sequence.data
    config = sequence.config

    if per_sequence:
        return (
            data.group_by([config.id_column, config.state_column])
            .agg(pl.len().alias("time_in_state"))
            .sort([config.id_column, config.state_column])
        )

    n_sequences = data[config.id_column].n_unique()

    # Count total occurrences of each state
    state_counts = (
        data.group_by(config.state_column)
        .agg(pl.len().alias("total_time"))
        .sort(config.state_column)
    )

    # Calculate mean per sequence
    state_counts = state_counts.with_columns(
        (pl.col("total_time") / n_sequences).alias("mean_time")
    )

    return state_counts


def spell_count(
    sequence: SequenceData,
    per_sequence: bool = False,
) -> int | float | pl.DataFrame:
    """
    Count the number of spells (runs) per sequence.

    A spell is a consecutive run of the same state.

    Args:
        sequence: StateSequence or SequencePool.
        per_sequence: If True, return counts for each sequence.

    Returns:
        If per_sequence=False: Mean spell count across sequences.
        If per_sequence=True: DataFrame with sequence IDs and spell counts.
    """

    def _spells(states: list[str]) -> int:
        return 0 if len(states) == 0 else 1 + _n_transitions(states)

    return reduce_per_sequence(sequence, _spells, "n_spells", per_sequence)


def visited_states(
    sequence: SequenceData,
    per_sequence: bool = False,
) -> int | float | pl.DataFrame:
    """
    Count the number of distinct states visited per sequence.

    Args:
        sequence: StateSequence or SequencePool.
        per_sequence: If True, return counts for each sequence.

    Returns:
        If per_sequence=False: Mean number of visited states.
        If per_sequence=True: DataFrame with sequence IDs and counts.
    """

    def _visited(states: list[str]) -> int:
        return len(set(states))

    return reduce_per_sequence(sequence, _visited, "n_visited", per_sequence)


def visited_proportion(
    sequence: SequenceData,
    per_sequence: bool = False,
) -> float | pl.DataFrame:
    """
    Proportion of the alphabet visited per sequence.

    Computed as n_visited_states / n_alphabet_states.

    Args:
        sequence: StateSequence or SequencePool.
        per_sequence: If True, return proportions for each sequence.

    Returns:
        If per_sequence=False: Mean visited proportion.
        If per_sequence=True: DataFrame with sequence IDs and proportions.
    """
    pool = SequencePool.coerce(sequence)
    n_states = len(pool.alphabet)

    def _proportion(states: list[str]) -> float:
        return len(set(states)) / n_states if n_states > 0 else 0.0

    return reduce_per_sequence(pool, _proportion, "visited_proportion", per_sequence)


def transition_proportion(
    sequence: SequenceData,
    per_sequence: bool = False,
) -> float | pl.DataFrame:
    """
    Proportion of positions that are transitions.

    Computed as n_transitions / (length - 1).

    Args:
        sequence: StateSequence or SequencePool.
        per_sequence: If True, return proportions for each sequence.

    Returns:
        If per_sequence=False: Mean transition proportion.
        If per_sequence=True: DataFrame with sequence IDs and proportions.
    """

    def _proportion(states: list[str]) -> float:
        n = len(states)
        return 0.0 if n <= 1 else _n_transitions(states) / (n - 1)

    return reduce_per_sequence(
        sequence, _proportion, "transition_proportion", per_sequence
    )


def modal_states(
    sequence: SequenceData,
    granularity: str | None = None,
) -> pl.DataFrame:
    """
    Get the modal (most frequent) state at each time position.

    Args:
        sequence: StateSequence or SequencePool.
        granularity: Polars ``dt.truncate`` unit string (e.g. ``"1d"``,
            ``"1w"``, ``"1mo"``, ``"1h"``) for re-bucketing the time column
            before computing modes. The time column must be a polars
            datetime/date dtype when this is set. ``None`` (default) uses
            the raw time values as stored.

            Strings only — integer granularities were removed in v0.3.2
            (hot-fix B3). For integer-indexed time, pre-bucket the column
            with ``(pl.col("t") // k * k)`` before constructing the pool.

    Returns:
        DataFrame with columns: time, modal_state, frequency, proportion.

    Raises:
        TypeError: If ``granularity`` is not a string (and not ``None``).
        ValueError: If ``granularity`` is set but the time column is not
            a datetime/date dtype.
    """
    data = sequence.data
    config = sequence.config

    time_col = config.time_column
    state_col = config.state_column

    if granularity is not None:
        if not isinstance(granularity, str):
            raise TypeError(
                f"modal_states: granularity must be a string polars truncate "
                f"unit (e.g. '1d', '1w'); got {type(granularity).__name__}. "
                f"Integer granularities were removed in v0.3.2 (hot-fix B3)."
            )
        time_dtype = data.schema[time_col]
        if not time_dtype.is_temporal():
            raise ValueError(
                f"modal_states: granularity={granularity!r} requires a "
                f"polars datetime/date time column; {time_col!r} has dtype "
                f"{time_dtype}. Either drop granularity or cast the time "
                f"column to Datetime first."
            )
        data = data.with_columns(
            pl.col(time_col).dt.truncate(granularity).alias(time_col)
        )

    # Count state occurrences at each time point
    counts = data.group_by([time_col, state_col]).agg(pl.len().alias("frequency"))

    # Total per time point
    totals = data.group_by(time_col).agg(pl.len().alias("total"))

    # Join and compute proportion
    counts = counts.join(totals, on=time_col).with_columns(
        (pl.col("frequency") / pl.col("total")).alias("proportion")
    )

    # Get the mode (max frequency) per time point
    max_freq = counts.group_by(time_col).agg(
        pl.col("frequency").max().alias("max_frequency")
    )

    result = (
        counts.join(max_freq, on=time_col)
        .filter(pl.col("frequency") == pl.col("max_frequency"))
        .select(
            [
                pl.col(time_col).alias("time"),
                pl.col(state_col).alias("modal_state"),
                pl.col("frequency"),
                pl.col("proportion"),
            ]
        )
        .sort("time")
    )

    return result


def sequence_frequency_table(
    sequence: SequenceData,
    n_top: int | None = None,
) -> pl.DataFrame:
    """
    Create a frequency table of sequence patterns.

    Each sequence is represented as a hyphen-separated string of states.

    When to use which: this counts how often each **complete sequence**
    occurs across the pool — "how many people share this exact
    trajectory?" (TraMineR's ``seqtab``). To measure the internal
    variety of each sequence — "how many distinct sub-patterns does a
    trajectory contain?" — use :func:`subsequence_count` instead. For a
    pool ``[A-B, A-B, A-C]``, this function reports ``A-B`` twice and
    ``A-C`` once; ``subsequence_count`` reports the number of distinct
    subsequences *within* each trajectory.

    Args:
        sequence: StateSequence or SequencePool.
        n_top: If provided, return only the top N patterns.

    Returns:
        DataFrame with columns: pattern, count, proportion.
    """
    pool = SequencePool.coerce(sequence)

    # Build patterns
    patterns: list[str] = []
    for seq_id in pool.sequence_ids:
        states = pool.get_sequence(seq_id)
        patterns.append("-".join(states))

    # Count patterns using polars
    df = pl.DataFrame({"pattern": patterns})
    result = (
        df.group_by("pattern")
        .agg(pl.len().alias("count"))
        .sort("count", descending=True)
    )

    total = result["count"].sum()
    result = result.with_columns((pl.col("count") / total).alias("proportion"))

    if n_top is not None:
        result = result.head(n_top)

    return result


def subsequence_count(
    sequence: SequenceData,
    per_sequence: bool = False,
    states_filter: list[str] | None = None,
    use_log: bool = False,
    dss: bool = True,
) -> int | float | pl.DataFrame:
    """
    Count the distinct subsequences of each sequence (TraMineR ``seqsubsn``).

    Uses the DP formula: dp[i] = 2 * dp[i-1] - dp[last[c]] where last[c]
    is the dp value before the previous occurrence of character c.

    As in TraMineR, the count includes the empty subsequence (so a one-state
    sequence counts 2) and, by default, runs over the distinct-successive-
    states form: consecutive repeats are collapsed first, which is the
    ``phi`` term of :func:`turbulence`. Pass ``dss=False`` to count over the
    full state sequence, one symbol per time point.

    When to use which: this measures the internal variety of each
    sequence — "how many distinct sub-patterns does a trajectory
    contain?". To count how often each **complete sequence** occurs across
    the pool — "how many people share this exact trajectory?" — use
    :func:`sequence_frequency_table` instead.

    For long sequences (>100 states), counts can grow astronomically large
    (exponential in sequence length). Use ``use_log=True`` to return
    log2 of the count instead.

    Args:
        sequence: StateSequence or SequencePool.
        per_sequence: If True, return counts for each sequence.
        states_filter: If provided, only count subsequences whose states
            are all in this list (applied after the ``dss`` collapse).
        use_log: If True, return log2 of the count to avoid huge integers.
        dss: If True (default, TraMineR's ``DSS=TRUE``), count over the
            distinct-successive-states form.

    Returns:
        If per_sequence=False: Mean distinct subsequence count (or log2 mean).
        If per_sequence=True: DataFrame with sequence IDs and counts.
    """
    allowed = set(states_filter) if states_filter is not None else None

    def _count(states: list[str]) -> int | float:
        if dss:
            states = _to_dss(states)
        if allowed is not None:
            states = [s for s in states if s in allowed]

        n = len(states)
        if n == 0:
            return 0.0 if use_log else 1  # only the empty subsequence

        if use_log:
            # Use log2 arithmetic to avoid overflow
            log_dp = [0.0] * (n + 1)  # log_dp[0] = log2(1) = 0
            last: dict[str, int] = {}
            for i in range(1, n + 1):
                log_dp[i] = log_dp[i - 1] + 1.0  # log2(2 * dp[i-1])
                c = states[i - 1]
                if c in last:
                    # log2(2*dp[i-1] - dp[last[c]-1])
                    diff = log_dp[last[c] - 1] - log_dp[i - 1]
                    if diff > -50:
                        log_dp[i] = log_dp[i - 1] + math.log2(2.0 - 2.0**diff)
                last[c] = i
            return log_dp[n]

        return _n_distinct_subsequences(states)

    col_name = "log2_n_subsequences" if use_log else "n_subsequences"
    return reduce_per_sequence(sequence, _count, col_name, per_sequence)


def normalized_turbulence(
    sequence: SequenceData,
    per_sequence: bool = False,
) -> float | pl.DataFrame:
    """
    Calculate turbulence rescaled to [0, 1] as TraMineR ``seqST(norm=TRUE)``.

    TraMineR's reference sequence is one of the pool's maximum length that
    cycles through the alphabet (``A B C A B C ...``): every spell has
    duration 1, so its turbulence is ``T_max = log2(phi_cycle)``. Each
    sequence is then rescaled as ``(T - 1) / (T_max - 1)``, since 1 is the
    minimum of the raw index, and clipped at 0. A one-state alphabet has
    ``T_max = 1`` and every sequence scores 0.

    Args:
        sequence: StateSequence or SequencePool.
        per_sequence: If True, return for each sequence.

    Returns:
        If per_sequence=False: Mean normalized turbulence.
        If per_sequence=True: DataFrame with sequence IDs and values.
    """
    pool = SequencePool.coerce(sequence)
    states = list(pool.alphabet.states)
    max_length = max((len(pool[i]) for i in pool.sequence_ids), default=0)
    if len(states) > 1 and max_length > 0:
        cycle = [states[i % len(states)] for i in range(max_length)]
        t_max = math.log2(_n_distinct_subsequences(cycle))
    else:
        t_max = 1.0

    def _normalized(seq_states: list[str]) -> float:
        if not seq_states or t_max <= 1.0:
            return 0.0
        return max((_turbulence_of(seq_states) - 1.0) / (t_max - 1.0), 0.0)

    return reduce_per_sequence(pool, _normalized, "normalized_turbulence", per_sequence)


def sequence_log_probability(
    sequence: SequenceData,
    per_sequence: bool = False,
) -> float | pl.DataFrame:
    """
    Compute log-probability of each sequence under the empirical transition model.

    For each sequence, sums log(P[s_t -> s_{t+1}]) over all consecutive pairs,
    where P is the transition rate matrix estimated from the pool.

    Sequences with zero-probability transitions get -inf for those steps.

    Args:
        sequence: StateSequence or SequencePool.
        per_sequence: If True, return log-probabilities for each sequence.

    Returns:
        If per_sequence=False: Mean log-probability across sequences.
        If per_sequence=True: DataFrame with sequence IDs and log-probabilities.
    """
    from yasqat.statistics.transition import transition_rate_matrix

    pool = SequencePool.coerce(sequence)
    state_to_idx = {s: i for i, s in enumerate(pool.alphabet.states)}

    # Transition rate matrix estimated from the whole pool
    trate = transition_rate_matrix(pool, as_counts=False)

    def _log_prob(states: list[str]) -> float:
        log_prob = 0.0
        for t in range(len(states) - 1):
            p = trate[state_to_idx[states[t]], state_to_idx[states[t + 1]]]
            if p <= 0:
                return float("-inf")
            log_prob += np.log(p)
        return log_prob

    return reduce_per_sequence(pool, _log_prob, "log_probability", per_sequence)

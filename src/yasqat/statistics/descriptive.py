"""Descriptive statistics for sequences."""

from __future__ import annotations

from typing import TYPE_CHECKING, cast

import numpy as np
import polars as pl

if TYPE_CHECKING:
    from yasqat.core.pool import SequencePool
    from yasqat.core.sequence import StateSequence


def longitudinal_entropy(
    sequence: StateSequence | SequencePool,
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
    from yasqat.core.pool import SequencePool
    from yasqat.core.sequence import StateSequence

    if isinstance(sequence, StateSequence):
        pool = SequencePool(
            data=sequence.data,
            config=sequence.config,
            alphabet=sequence.alphabet,
        )
    else:
        pool = sequence

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
    sequence: StateSequence | SequencePool,
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
    from yasqat.core.pool import SequencePool
    from yasqat.core.sequence import StateSequence

    if isinstance(sequence, StateSequence):
        pool = SequencePool(
            data=sequence.data,
            config=sequence.config,
            alphabet=sequence.alphabet,
        )
    else:
        pool = sequence

    config = pool.config
    counts = []
    seq_ids = []

    for seq_id in pool.sequence_ids:
        states = pool.get_sequence(seq_id)
        n_transitions = sum(
            1 for i in range(len(states) - 1) if states[i] != states[i + 1]
        )
        counts.append(n_transitions)
        seq_ids.append(seq_id)

    if per_sequence:
        return pl.DataFrame(
            {
                config.id_column: seq_ids,
                "n_transitions": counts,
            }
        )

    return sum(counts)


def sequence_length(
    sequence: StateSequence | SequencePool,
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
    from yasqat.core.pool import SequencePool
    from yasqat.core.sequence import StateSequence

    if isinstance(sequence, StateSequence):
        pool = SequencePool(
            data=sequence.data,
            config=sequence.config,
            alphabet=sequence.alphabet,
        )
    else:
        pool = sequence

    config = pool.config
    lengths = []
    seq_ids = []

    for seq_id in pool.sequence_ids:
        states = pool.get_sequence(seq_id)
        lengths.append(len(states))
        seq_ids.append(seq_id)

    if per_sequence:
        return pl.DataFrame(
            {
                config.id_column: seq_ids,
                "length": lengths,
            }
        )

    return float(np.mean(lengths))


def complexity_index(
    sequence: StateSequence | SequencePool,
    per_sequence: bool = False,
) -> float | pl.DataFrame:
    """
    Calculate complexity index (Elzinga's complexity measure).

    The complexity index combines the number of transitions with
    the diversity of states. It is defined as:
    C(s) = sqrt(n_transitions * n_distinct_states) / length

    Args:
        sequence: StateSequence or SequencePool.
        per_sequence: If True, return complexity for each sequence.

    Returns:
        If per_sequence=False: Mean complexity across sequences.
        If per_sequence=True: DataFrame with sequence IDs and complexity.
    """
    from yasqat.core.pool import SequencePool
    from yasqat.core.sequence import StateSequence

    if isinstance(sequence, StateSequence):
        pool = SequencePool(
            data=sequence.data,
            config=sequence.config,
            alphabet=sequence.alphabet,
        )
    else:
        pool = sequence

    config = pool.config
    complexities = []
    seq_ids = []

    for seq_id in pool.sequence_ids:
        states = pool.get_sequence(seq_id)
        n = len(states)

        if n <= 1:
            complexities.append(0.0)
            seq_ids.append(seq_id)
            continue

        n_transitions = sum(1 for i in range(n - 1) if states[i] != states[i + 1])
        n_distinct = len(set(states))

        complexity = np.sqrt(n_transitions * n_distinct) / n
        complexities.append(complexity)
        seq_ids.append(seq_id)

    if per_sequence:
        return pl.DataFrame(
            {
                config.id_column: seq_ids,
                "complexity": complexities,
            }
        )

    return float(np.mean(complexities))


def turbulence(
    sequence: StateSequence | SequencePool,
    per_sequence: bool = False,
) -> float | pl.DataFrame:
    """
    Calculate turbulence index (Elzinga & Liefbroer, 2007).

    Turbulence measures the "unpredictability" of a sequence,
    combining state changes with spell duration variability.

    Formula: T(s) = log2(phi * (st(s) + 1) / tbar(s))
    where:
    - phi is the number of distinct subsequences
    - st(s) is the variance of spell durations
    - tbar(s) is the mean spell duration

    Args:
        sequence: StateSequence or SequencePool.
        per_sequence: If True, return turbulence for each sequence.

    Returns:
        If per_sequence=False: Mean turbulence across sequences.
        If per_sequence=True: DataFrame with sequence IDs and turbulence.
    """
    from yasqat.core.pool import SequencePool
    from yasqat.core.sequence import StateSequence

    if isinstance(sequence, StateSequence):
        pool = SequencePool(
            data=sequence.data,
            config=sequence.config,
            alphabet=sequence.alphabet,
        )
    else:
        pool = sequence

    config = pool.config
    id_col = config.id_column

    # Get spells (run-length encoded) using vectorized to_sps
    state_seq = pool.to_state_sequence()
    sps = state_seq.to_sps()

    # Vectorized turbulence computation using polars group_by
    turb_df = (
        sps.group_by(id_col)
        .agg(
            [
                pl.len().alias("n_spells"),
                pl.col("duration").var(ddof=0).alias("duration_var"),
                pl.col("duration").mean().alias("mean_duration"),
            ]
        )
        .with_columns(pl.col("duration_var").fill_null(0.0))
        .with_columns(
            pl.when(pl.col("n_spells") > 1)
            .then(
                (
                    pl.col("n_spells")
                    * (pl.col("duration_var") + 1)
                    / pl.col("mean_duration")
                )
                .log(2)
                .clip(lower_bound=0.0)
            )
            .otherwise(0.0)
            .alias("turbulence")
        )
        .select([id_col, "turbulence"])
        .sort(id_col)
    )

    if per_sequence:
        return turb_df

    return cast(float, turb_df["turbulence"].mean())


def state_distribution(
    sequence: StateSequence | SequencePool,
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
    from yasqat.core.sequence import StateSequence

    if isinstance(sequence, StateSequence):
        data = sequence.data
        config = sequence.config
    else:
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
    sequence: StateSequence | SequencePool,
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
    from yasqat.core.sequence import StateSequence

    if isinstance(sequence, StateSequence):
        data = sequence.data
        config = sequence.config
    else:
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
    sequence: StateSequence | SequencePool,
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
    from yasqat.core.pool import SequencePool
    from yasqat.core.sequence import StateSequence

    if isinstance(sequence, StateSequence):
        pool = SequencePool(
            data=sequence.data,
            config=sequence.config,
            alphabet=sequence.alphabet,
        )
    else:
        pool = sequence

    config = pool.config
    counts = []
    seq_ids = []

    for seq_id in pool.sequence_ids:
        states = pool.get_sequence(seq_id)
        if len(states) == 0:
            n_spells = 0
        else:
            n_spells = 1 + sum(
                1 for i in range(len(states) - 1) if states[i] != states[i + 1]
            )
        counts.append(n_spells)
        seq_ids.append(seq_id)

    if per_sequence:
        return pl.DataFrame(
            {
                config.id_column: seq_ids,
                "n_spells": counts,
            }
        )

    return float(np.mean(counts))


def visited_states(
    sequence: StateSequence | SequencePool,
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
    from yasqat.core.pool import SequencePool
    from yasqat.core.sequence import StateSequence

    if isinstance(sequence, StateSequence):
        pool = SequencePool(
            data=sequence.data,
            config=sequence.config,
            alphabet=sequence.alphabet,
        )
    else:
        pool = sequence

    config = pool.config
    counts = []
    seq_ids = []

    for seq_id in pool.sequence_ids:
        states = pool.get_sequence(seq_id)
        counts.append(len(set(states)))
        seq_ids.append(seq_id)

    if per_sequence:
        return pl.DataFrame(
            {
                config.id_column: seq_ids,
                "n_visited": counts,
            }
        )

    return float(np.mean(counts))


def visited_proportion(
    sequence: StateSequence | SequencePool,
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
    from yasqat.core.pool import SequencePool
    from yasqat.core.sequence import StateSequence

    if isinstance(sequence, StateSequence):
        pool = SequencePool(
            data=sequence.data,
            config=sequence.config,
            alphabet=sequence.alphabet,
        )
    else:
        pool = sequence

    config = pool.config
    n_states = len(pool.alphabet)
    proportions = []
    seq_ids = []

    for seq_id in pool.sequence_ids:
        states = pool.get_sequence(seq_id)
        n_visited = len(set(states))
        prop = n_visited / n_states if n_states > 0 else 0.0
        proportions.append(prop)
        seq_ids.append(seq_id)

    if per_sequence:
        return pl.DataFrame(
            {
                config.id_column: seq_ids,
                "visited_proportion": proportions,
            }
        )

    return float(np.mean(proportions))


def transition_proportion(
    sequence: StateSequence | SequencePool,
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
    from yasqat.core.pool import SequencePool
    from yasqat.core.sequence import StateSequence

    if isinstance(sequence, StateSequence):
        pool = SequencePool(
            data=sequence.data,
            config=sequence.config,
            alphabet=sequence.alphabet,
        )
    else:
        pool = sequence

    config = pool.config
    proportions = []
    seq_ids = []

    for seq_id in pool.sequence_ids:
        states = pool.get_sequence(seq_id)
        n = len(states)
        if n <= 1:
            proportions.append(0.0)
        else:
            n_trans = sum(1 for i in range(n - 1) if states[i] != states[i + 1])
            proportions.append(n_trans / (n - 1))
        seq_ids.append(seq_id)

    if per_sequence:
        return pl.DataFrame(
            {
                config.id_column: seq_ids,
                "transition_proportion": proportions,
            }
        )

    return float(np.mean(proportions))


def modal_states(
    sequence: StateSequence | SequencePool,
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
    from yasqat.core.sequence import StateSequence

    if isinstance(sequence, StateSequence):
        data = sequence.data
        config = sequence.config
    else:
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
    sequence: StateSequence | SequencePool,
    n_top: int | None = None,
) -> pl.DataFrame:
    """
    Create a frequency table of sequence patterns.

    Each sequence is represented as a hyphen-separated string of states.

    Args:
        sequence: StateSequence or SequencePool.
        n_top: If provided, return only the top N patterns.

    Returns:
        DataFrame with columns: pattern, count, proportion.
    """
    from yasqat.core.pool import SequencePool
    from yasqat.core.sequence import StateSequence

    if isinstance(sequence, StateSequence):
        pool = SequencePool(
            data=sequence.data,
            config=sequence.config,
            alphabet=sequence.alphabet,
        )
    else:
        pool = sequence

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
    sequence: StateSequence | SequencePool,
    per_sequence: bool = False,
    states_filter: list[str] | None = None,
    use_log: bool = False,
) -> int | float | pl.DataFrame:
    """
    Count the number of distinct subsequences from the DSS representation.

    Uses the DP formula: dp[i] = 2 * dp[i-1] - dp[last[c]] where last[c]
    is the dp value before the previous occurrence of character c.

    For long sequences (>100 states), counts can grow astronomically large
    (exponential in sequence length). Use ``use_log=True`` to return
    log2 of the count instead.

    Args:
        sequence: StateSequence or SequencePool.
        per_sequence: If True, return counts for each sequence.
        states_filter: If provided, only count subsequences whose states
            are all in this list.
        use_log: If True, return log2 of the count to avoid huge integers.

    Returns:
        If per_sequence=False: Mean distinct subsequence count (or log2 mean).
        If per_sequence=True: DataFrame with sequence IDs and counts.
    """
    import math

    from yasqat.core.pool import SequencePool
    from yasqat.core.sequence import StateSequence

    if isinstance(sequence, StateSequence):
        pool = SequencePool(
            data=sequence.data,
            config=sequence.config,
            alphabet=sequence.alphabet,
        )
    else:
        pool = sequence

    config = pool.config
    counts: list[int | float] = []
    seq_ids: list[int | str] = []

    for seq_id in pool.sequence_ids:
        states = pool.get_sequence(seq_id)

        # Filter to only the requested states if given
        if states_filter is not None:
            allowed = set(states_filter)
            states = [s for s in states if s in allowed]

        n = len(states)
        if n == 0:
            counts.append(0)
            seq_ids.append(seq_id)
            continue

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
            counts.append(log_dp[n])
        else:
            # Standard DP — Python big ints handle overflow
            dp = [0] * (n + 1)
            dp[0] = 1
            last_seen: dict[str, int] = {}
            for i in range(1, n + 1):
                dp[i] = 2 * dp[i - 1]
                c = states[i - 1]
                if c in last_seen:
                    dp[i] -= dp[last_seen[c] - 1]
                last_seen[c] = i
            counts.append(dp[n] - 1)

        seq_ids.append(seq_id)

    col_name = "log2_n_subsequences" if use_log else "n_subsequences"

    if per_sequence:
        return pl.DataFrame(
            {
                config.id_column: seq_ids,
                col_name: counts,
            }
        )

    return float(np.mean(counts))


def normalized_turbulence(
    sequence: StateSequence | SequencePool,
    per_sequence: bool = False,
) -> float | pl.DataFrame:
    """
    Calculate normalized turbulence, rescaled to [0, 1].

    Divides turbulence by the theoretical maximum for the given sequence
    length. The maximum turbulence occurs when all positions are different
    states with uniform spell durations of 1.

    Args:
        sequence: StateSequence or SequencePool.
        per_sequence: If True, return for each sequence.

    Returns:
        If per_sequence=False: Mean normalized turbulence.
        If per_sequence=True: DataFrame with sequence IDs and values.
    """
    from yasqat.core.pool import SequencePool
    from yasqat.core.sequence import StateSequence

    if isinstance(sequence, StateSequence):
        pool = SequencePool(
            data=sequence.data,
            config=sequence.config,
            alphabet=sequence.alphabet,
        )
    else:
        pool = sequence

    config = pool.config
    id_col = config.id_column

    # Get raw turbulence per sequence
    turb_df = turbulence(pool, per_sequence=True)
    assert isinstance(turb_df, pl.DataFrame)

    # Get sequence lengths using polars
    lengths = pool.data.group_by(id_col).agg(pl.len().alias("_length"))

    # Vectorized normalization: T_max = log2(n)
    result = (
        turb_df.join(lengths, on=id_col)
        .with_columns(
            pl.when(pl.col("_length") > 1)
            .then(pl.col("turbulence") / pl.col("_length").cast(pl.Float64).log(2))
            .otherwise(0.0)
            .alias("normalized_turbulence")
        )
        .select([id_col, "normalized_turbulence"])
        .sort(id_col)
    )

    if per_sequence:
        return result

    return cast(float, result["normalized_turbulence"].mean())


def sequence_log_probability(
    sequence: StateSequence | SequencePool,
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
    from yasqat.core.pool import SequencePool
    from yasqat.core.sequence import StateSequence
    from yasqat.statistics.transition import transition_rate_matrix

    if isinstance(sequence, StateSequence):
        pool = SequencePool(
            data=sequence.data,
            config=sequence.config,
            alphabet=sequence.alphabet,
        )
    else:
        pool = sequence

    config = pool.config
    alphabet = pool.alphabet
    state_to_idx = {s: i for i, s in enumerate(alphabet.states)}

    # Get transition rate matrix from the pool
    trate = transition_rate_matrix(pool, as_counts=False)

    log_probs = []
    seq_ids = []

    for seq_id in pool.sequence_ids:
        states = pool.get_sequence(seq_id)
        log_prob = 0.0

        for t in range(len(states) - 1):
            i = state_to_idx[states[t]]
            j = state_to_idx[states[t + 1]]
            p = trate[i, j]
            if p > 0:
                log_prob += np.log(p)
            else:
                log_prob = float("-inf")
                break

        log_probs.append(log_prob)
        seq_ids.append(seq_id)

    if per_sequence:
        return pl.DataFrame(
            {
                config.id_column: seq_ids,
                "log_probability": log_probs,
            }
        )

    return float(np.mean(log_probs))

# Loading data

Every loader returns a `SequencePool`. The loaders validate the three required
columns, infer the alphabet, and sort by id and time; nothing else happens
until you ask for a distance or a statistic.

## From a file

```python
from yasqat.io import load_csv, load_parquet, load_json

pool = load_csv("trajectories.csv")          # expects id, time, state columns
pool = load_parquet("trajectories.parquet")
pool = load_json("trajectories.json")
```

Keyword arguments pass through to the underlying polars reader, so
`load_csv(path, separator=";")` works as you would expect.

## From a DataFrame

If the data already lives in polars, or comes from Spark, Arrow, or pandas via
an Arrow bridge, hand the frame over directly.

```python
from yasqat.io import load_dataframe
from yasqat.core import SequenceConfig

pool = load_dataframe(
    df,
    config=SequenceConfig(id_column="member_id", time_column="month", state_column="tier"),
    drop_nulls=True,   # drop rows whose state is null instead of raising
)
```

`load_dataframe` is the one seam every file loader routes through, so the
validation and alphabet rules are identical whichever entry point you use.

## From intervals

Source systems often record **episodes** (start, end, state) rather than
observations at fixed time points. `StateSequence.from_intervals` samples
episodes onto the grid you specify.

```python
from yasqat.core import StateSequence

episodes = pl.DataFrame(
    {
        "id":    [1, 1, 2],
        "start": [0, 6, 0],
        "end":   [6, 12, 12],
        "state": ["free", "paid", "free"],
    }
)
ss = StateSequence.from_intervals(episodes, time_points=list(range(0, 12, 3)))
pool = SequencePool.coerce(ss)
```

Read the function's docstring for the column names it expects and how it
treats an observation that falls on an episode boundary.

## Checking what you loaded

```python
pool.describe()
pool.sequence_lengths().head()
pool.alphabet.states
```

Three checks catch most data problems early:

1. **The alphabet is what you expect.** A stray `"Paid"` next to `"paid"`, or
   a null encoded as the string `"None"`, shows up here as an extra state.
2. **Lengths are plausible.** Different lengths are fine; a length of 1 for
   half the population usually means a join went wrong upstream.
3. **Time is a real order.** Time can be integers or dates, but within an id
   it must be sortable, and duplicates of (id, time) will silently produce a
   longer sequence than you meant.

## Recoding states

Collapsing a fine-grained vocabulary is common enough to have its own method.
It returns a new pool with a new alphabet.

```python
coarse = pool.recode_states({"support": "free"})   # treat support contacts as free-tier months
coarse.alphabet.states
# ('churn', 'free', 'paid')
```

## Saving

`save_csv`, `save_parquet`, and `save_json` accept either container and write
the long-format frame back out, so a filtered or recoded pool can be handed
to the next tool in the pipeline.

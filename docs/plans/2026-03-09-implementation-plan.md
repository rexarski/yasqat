# yasqat package upgrade plan

While preparing for a demo notebook, we drafted draft_spark_to_pool.py script and demo_showcase.ipynb notebook. Then we identified some of the issues with current version v0.3.0 of yasqat. To improve the usability, we propose the following changes.

Most of the changes and rationales are coming from the draft_spark_to_pool.py

## Hive table data loader

Add support of loading data frame a hive table.

## FastSequencePool

The og SequencePool._extract_sequences iterates every unique user ID and calls self._data.filter(id==seq_id) for each -- a full table scan per user. For large dataset, this is very costly.

Can we replace the original SequencePool class with the newly implemented FastSequencePool? But before doing so, help me understand whether the og implementation has its advantage. If it does have some good reasons to stick around, we can turn FastSequencePool into an option, if not, then totally replace the original implementation

## Transitions

When we try to find the top transitions, I kinda want to add a switch where we can exclude trivial transitions, for example State A -> State A, aka from_state and to_state are the same, but this is by default turned off.

## Visualization

For index plot I want to double check that if I'm trying to plot 50k sequences at the same time, will plotnine clip away some sequences out of the canvas? Or in other words, will plotnine faithfully plot all 50k sequences?

Another request is that for any visualization with categories > 15, turn off the legends by default please. (we can of course turn it back on.)

## Requirements

- I removed tanat source code. Just an FYI.
- CLAUDE.md is long overdue, please update this to reflect current development principal and status.
- Update demo_showcase.ipynb to merge the new changes (mostly from draft_spark_to_pool) to reflect the changes we made.
- Make sure you update related files including CHANGELOG.md, pyproject.toml, Quarto documentation site, README.md and corresponding unit tests in /tests

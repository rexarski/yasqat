# yasqat

**Yet Another Sequence Analytics Toolkit** — a modern Python library for
categorical sequence analysis, built on [polars](https://pola.rs/). Designed for
social-science and life-course research: labour-market trajectories, health
pathways, educational histories, and similar domains.

Inspired by [TraMineR](http://traminer.unige.ch/) (R) and
[TanaT](https://tanat.gitlabpages.inria.fr/core/tanat/) (Python).

## Data flow

```
raw data → io loaders → SequencePool → metrics → DistanceMatrix → clustering / statistics
```

Every public method returns a polars `DataFrame`; you bring your own plotting
tool. `SequencePool` is the canonical analysis container; `StateSequence` is the
representation view (STS/SPS/DSS formats, per-sequence descriptives). See the
[Glossary](glossary.md) for the full domain vocabulary.

```{toctree}
:maxdepth: 2
:caption: Getting started

installation
quickstart
```

```{toctree}
:maxdepth: 2
:caption: Reference

api/index
glossary
changelog
```

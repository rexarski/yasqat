# yasqat

**Yet Another Sequence Analytics Toolkit**: categorical sequence analysis,
native to Python. Each unit of study, a person, an account, a patient, is an
ordered run of states over time. yasqat measures how alike those runs are,
finds the kinds of trajectory a population contains, and describes how states
flow, all on [polars](https://pola.rs/) DataFrames.

The methods come from [TraMineR](http://traminer.unige.ch/), the reference
toolkit of the field, which yasqat treats as the oracle for correctness,
naming, and scope. The engineering is Python-first: polars for data, numba for
the pairwise kernels, and a polars `DataFrame` out of every public function.

![The yasqat pipeline: loaders build a pool, distances feed clustering, statistics read the pool directly](_static/diagrams/pipeline.svg)

## Where to start

- **New to sequence analysis?** Read [What sequence analysis is](concepts/sequence-analysis.md),
  then the [Quick start](quickstart.md).
- **Know TraMineR?** Skim the [data model](concepts/data-model.md) for the
  container vocabulary, then jump to the [guides](guides/index.md).
- **Deciding whether to adopt it?** [Why yasqat](why-yasqat.md) states the
  problem it solves, the positions it takes, and what it does not do.

## Install

```bash
pip install yasqat
```

Python 3.11 or newer. See [Installation](installation.md) for a development
setup and the documentation toolchain.

## One minute of yasqat

```python
from yasqat.io import load_csv
from yasqat.clustering import pam_clustering
from yasqat.statistics import association_rules, longitudinal_entropy

pool = load_csv("trajectories.csv")               # columns: id, time, state

dm = pool.compute_distances(method="om")           # optimal-matching distances
typology = pam_clustering(dm, n_clusters=4)        # a trajectory typology
longitudinal_entropy(pool, per_sequence=True)      # how diverse each path is
association_rules(pool, min_support=0.2)           # "this, then that", scored
```

Every call above returns a polars `DataFrame` or a small result object that
converts to one. You bring your own plotting library.

## Technical primer

A standalone primer written for technical leadership, covering lineage, data
model, capabilities, and roadmap in ten pages, is available as
[HTML](_static/primer/yasqat-overview-en.html) and
[PDF](_static/primer/yasqat-overview-en.pdf). The [Concepts](concepts/index.md)
and [Why yasqat](why-yasqat.md) pages are the maintained versions of that
material.

```{toctree}
:maxdepth: 1
:caption: Getting started
:hidden:

installation
quickstart
why-yasqat
```

```{toctree}
:maxdepth: 2
:caption: Concepts
:hidden:

concepts/index
```

```{toctree}
:maxdepth: 2
:caption: Guides
:hidden:

guides/index
```

```{toctree}
:maxdepth: 1
:caption: Reference
:hidden:

api/index
glossary
changelog
```

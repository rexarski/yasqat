# Concepts

Three short pages that explain the ideas the API is built on. Read them once
and the function names in the guides will feel obvious.

- [What sequence analysis is](sequence-analysis.md): sequences, alphabets,
  spells, and the three questions the discipline asks.
- [The data model](data-model.md): how a long-format DataFrame becomes an
  `Alphabet`, a `SequencePool`, and a `StateSequence`, and why there is
  exactly one encoding seam.
- [The pipeline](pipeline.md): the one-directional flow from loaders to
  distances to clustering and statistics, and where filters re-enter it.

```{toctree}
:maxdepth: 1
:hidden:

sequence-analysis
data-model
pipeline
```

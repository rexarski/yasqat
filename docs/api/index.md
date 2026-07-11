# API reference

The public surface of yasqat, organized by subpackage. Each page is generated
from the module's docstrings, so it always matches the installed version.

| Subpackage | Contents |
|---|---|
| [core](core.md) | `Alphabet`, `StateSequence`, `SequencePool`, `SequenceConfig` |
| [metrics](metrics.md) | Pairwise distance metrics and `DistanceMatrix` |
| [statistics](statistics.md) | Transition rates, descriptive/normative stats, mining, trees |
| [clustering](clustering.md) | PAM, CLARA, hierarchical, quality indices |
| [filters](filters.md) | Sequence filtering criteria |
| [io](io.md) | CSV/Parquet/JSON/DataFrame loaders |
| [synthetic](synthetic.md) | Markov and financial trajectory generators |

```{toctree}
:maxdepth: 1
:hidden:

core
metrics
statistics
clustering
filters
io
synthetic
```

# Mining patterns

Frequent-subsequence mining finds ordered patterns that recur across a
population; sequential association rules turn each pattern into a claim of the
form "this, then that" with a measured strength. This guide uses the ten
subscription accounts, small enough to derive every number by hand.

The ten accounts again, as an index plot:

![Ten subscription accounts](../_static/diagrams/index-plot.svg)

## Frequent subsequences

A pattern is a **subsequence**: its states must appear in order, but not
necessarily adjacently. `free → paid` matches account 2 (`free free support
paid …`) because `free` occurs before `paid`, even with `support` in between.

```python
from yasqat.statistics import frequent_subsequences

frequent_subsequences(subs, min_support=0.3, max_length=2, min_length=2)
```

```
┌──────────────────────┬─────────┬────────────┐
│ subsequence          ┆ support ┆ proportion │
╞══════════════════════╪═════════╪════════════╡
│ ["free", "paid"]     ┆ 7       ┆ 0.7        │
│ ["paid", "paid"]     ┆ 7       ┆ 0.7        │
│ ["free", "support"]  ┆ 6       ┆ 0.6        │
│ ["free", "free"]     ┆ 5       ┆ 0.5        │
│ ["free", "churn"]    ┆ 4       ┆ 0.4        │
│ ["support", "paid"]  ┆ 4       ┆ 0.4        │
│ ["support", "churn"] ┆ 3       ┆ 0.3        │
└──────────────────────┴─────────┴────────────┘
```

**Support** is the number of sequences containing the pattern; `proportion`
divides by the pool size. Check one: `free → paid` appears in accounts 1, 2,
4, 5, 7, 8, and 10, seven of ten. `support → churn` appears in 3, 5, and 9.

Two parameters shape the search. `min_support` is the threshold below which a
pattern is dropped, and because any extension of an infrequent pattern is also
infrequent, the search prunes aggressively (this is the Apriori principle).
`max_length` caps the pattern length; keep it small, five or less, since
candidate counts grow with the alphabet size to that power.

## Association rules

Every frequent pattern of length two or more is split at each internal
position into an **antecedent** (the prefix) and a **consequent** (the
suffix). The rule `antecedent ⇒ consequent` is then scored.

```python
from yasqat.statistics import association_rules

rules = association_rules(subs, min_support=0.3, min_confidence=0.5, max_length=2)
rules.with_columns(pl.col("antecedent").list.join("→"), pl.col("consequent").list.join("→"))
```

```
┌────────────┬────────────┬─────────┬────────────┬──────────┬──────────┬────────────┐
│ antecedent ┆ consequent ┆ support ┆ confidence ┆ lift     ┆ leverage ┆ conviction │
╞════════════╪════════════╪═════════╪════════════╪══════════╪══════════╪════════════╡
│ paid       ┆ paid       ┆ 0.7     ┆ 1.0        ┆ 1.428571 ┆ 0.21     ┆ inf        │
│ free       ┆ paid       ┆ 0.7     ┆ 0.7        ┆ 1.0      ┆ 0.0      ┆ 1.0        │
│ support    ┆ paid       ┆ 0.4     ┆ 0.666667   ┆ 0.952381 ┆ -0.02    ┆ 0.9        │
│ free       ┆ support    ┆ 0.6     ┆ 0.6        ┆ 1.0      ┆ 0.0      ┆ 1.0        │
│ support    ┆ churn      ┆ 0.3     ┆ 0.5        ┆ 1.25     ┆ 0.06     ┆ 1.2        │
│ free       ┆ free       ┆ 0.5     ┆ 0.5        ┆ 0.5      ┆ -0.5     ┆ 0.0        │
└────────────┴────────────┴─────────┴────────────┴──────────┴──────────┴────────────┘
```

## The measures, derived on paper

Take the rule `support ⇒ churn`. Write `s(X)` for the proportion of sequences
containing pattern `X`.

| Measure | Definition | For `support ⇒ churn` |
|---|---|---|
| support | `s(support → churn)` | 3 of 10 = **0.3** |
| confidence | `s(support → churn) / s(support)` | 0.3 / 0.6 = **0.5** |
| lift | `confidence / s(churn)` | 0.5 / 0.4 = **1.25** |
| leverage | `s(support → churn) − s(support) · s(churn)` | 0.3 − 0.6 × 0.4 = **0.06** |
| conviction | `(1 − s(churn)) / (1 − confidence)` | 0.6 / 0.5 = **1.2** |

In words:

- **Confidence** is the rate: among accounts that contacted support, half
  later churned.
- **Lift** compares that rate with the base rate. Churn happens in 40% of
  accounts overall, so 50% after a support contact is 1.25× the baseline.
  A lift of 1 means the antecedent tells you nothing; below 1 it is
  protective.
- **Leverage** is the same comparison on an additive scale: 6 percentage
  points more joint occurrence than independence would give.
- **Conviction** is how much more often the rule would be wrong if antecedent
  and consequent were independent. It is infinite when confidence is 1, as
  for `paid ⇒ paid`, and 0 when the rule is exactly wrong.

The rule `free ⇒ free` shows lift below 1: 0.5 / 1.0. Every account contains
`free`, so a second `free` month is *less* likely than the base rate of
`free` predicts. Lift is only meaningful when the consequent is not
universal, which is also why `free ⇒ paid` scores exactly 1.0.

## Reading a rules table

Sort by confidence to find reliable rules; filter by lift above 1 to keep
those that beat the base rate; then weigh by support so that a perfect rule
on three sequences does not outrank a good rule on three thousand.

```python
rules.filter(pl.col("lift") > 1.0).sort("support", descending=True)
```

Reach matters as much as strength: a 95% rate on a handful of sequences is a
curiosity, the same rate on a large share of the population is a programme.
Report all three.

## Given a situation

The most useful rules are often conditional on a situation you define first.
`pool.filter` returns a pool, so it drops straight into `association_rules`.

```python
from yasqat.filters import ContainsStateCriterion

situation = subs.filter(ContainsStateCriterion(states=["support"]))
situation.sequence_ids     # [2, 3, 5, 7, 9, 10]
association_rules(situation, min_support=0.3, min_confidence=0.5, max_length=2)
```

Within the six accounts that contacted support, `support ⇒ churn` has
confidence 0.5 and lift 1.0, because churn's base rate inside the situation
is also 0.5. The lift of 1.25 on the whole population was carried by the
comparison with accounts that never contacted support. That is the kind of
thing conditioning reveals, and it is the subject of
[Conditioning on a situation](filtering.md).

## Limits to know

The miner is a level-wise, generate-and-test search in pure Python. It is
exact and complete above the support threshold, and comfortable for cohorts
of thousands to a few hundred thousand sequences with alphabets of a couple
of dozen states and patterns up to length five. With a large alphabet and a
low support threshold, candidate generation grows quickly; raise
`min_support` or lower `max_length` first. Rules have no gap or time-window
constraints yet; a pattern matches wherever its states occur in order.

# Compile terminology glossary (`v0.3.2-appendix.md`)

**Status:** `resolved` (0.5.0)
**Type:** docs
**Source:** migrated from GitHub #54 (closed 2026-06-22)
**Related:** this is effectively the seed of the repo's `CONTEXT.md`

## Description

Cover the terminology used throughout the documentation (alphabet, state,
spell, sequence, distance metric, OM, substitution cost, turbulence,
complexity, discrepancy, etc.).

## Tasks

- [x] Compile the glossary. Consider writing it directly as `CONTEXT.md` at the
      repo root (the single-context domain glossary the agent-skills setup
      expects) rather than a standalone appendix — see `docs/agents/domain.md`.

## Comments

- 2026-07-07 (resolved, 0.5.0): written directly into `CONTEXT.md` (created
  by ADR-0002 work earlier this cycle, expanded here). Sections: containers,
  sequence formats, sequence anatomy (state/sequence/spell/transition), and
  analysis vocabulary (metric, OM, indel, substitution matrix, entropy,
  turbulence, complexity, discrepancy, normative indicators). No standalone
  appendix file — `CONTEXT.md` is the single-context home per
  `docs/agents/domain.md`.

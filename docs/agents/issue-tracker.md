# Issue tracker: Local Markdown

Issues and PRDs for this repo live as markdown files in `.scratch/`.

## Conventions

- **`.scratch/README.md` is the board** — a table of every issue with its type
  and `Status:`. Keep it in sync whenever you add an issue or change a status.
- **Standalone backlog issues** live flat: `.scratch/issues/<NN>-<slug>.md`,
  numbered from `01`. This is the default for independent bugs/tasks.
- **Multi-issue features** get their own directory: `.scratch/<feature-slug>/`
  with a `PRD.md` and `issues/<NN>-<slug>.md` files numbered from `01`.
- **Triage state** is a `**Status:**` line near the top of each issue file (see
  `triage-labels.md` for the role strings).
- **Architecture decisions** are not issues — they go in `docs/adr/` (see
  `domain.md`).
- Comments and conversation history append to the bottom under a `## Comments`
  heading.

## When a skill says "publish to the issue tracker"

Create a new file under `.scratch/<feature-slug>/` (creating the directory if needed).

## When a skill says "fetch the relevant ticket"

Read the file at the referenced path. The user will normally pass the path or the issue number directly.

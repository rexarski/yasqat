# yasqat — PyPI Publishing Plan

Based on: https://docs.astral.sh/uv/guides/package/
Tool: `uv` (build + publish), backend: `hatchling`

---

## Version decision

Current `pyproject.toml` version: `0.2.1` (never published; no git tags exist).

Since the code review session introduced user-facing API changes that break
calling code written against the pre-review codebase:
- `SequencePool.compute_distances()` now returns `DistanceMatrix` instead of
  `np.ndarray` — any code that did `dm.shape` or `np.diag(dm)` directly breaks
- `sequence_ids` changed from a method call `seq.sequence_ids()` to a property
  access `seq.sequence_ids` on all `BaseSequence` subclasses

**Recommendation: bump to `0.3.0` before the first public release.**

This is the first ever release, so strict semver doesn't technically apply, but
`0.3.0` signals "post-review, API-stabilised alpha" and avoids confusion if
anyone has installed from source.

---

## Pre-release checklist

Before running any publish commands, verify each item:

- [ ] All tests pass: `uv run pytest`
- [ ] Linter clean: `uv run ruff check src/ tests/`
- [ ] Types check: `uv run mypy src/yasqat/`
- [ ] `pyproject.toml` metadata is complete (see gaps below)
- [ ] `README.md` quick-start is accurate against current API
- [ ] `CHANGELOG.md` written (see step 1)
- [ ] `dev` branch merged into `main`
- [ ] Git tag created for the release commit

### Known gaps in `pyproject.toml`

1. **Author email missing** — PyPI shows it in the sidebar. Add:
   ```toml
   authors = [
       { name = "rexarski", email = "your@email.com" }
   ]
   ```

2. **No `Documentation` URL** — optional but good practice once docs exist:
   ```toml
   [project.urls]
   Documentation = "https://..."
   ```

3. **`sdist` includes `tests/`** — valid, but increases sdist size. Intentional
   choice; leave as-is unless you want a leaner source distribution.

---

## Step 1 — Write CHANGELOG.md

Create `CHANGELOG.md` in the repo root. Minimum content for a first release:

```markdown
# Changelog

## 0.3.0 (2026-02-23) — first public release

### Breaking changes
- `SequencePool.compute_distances()` now returns `DistanceMatrix` (with `.values`
  and `.labels`) instead of a raw `np.ndarray`.
- `sequence_ids` is now a `@property` on all `BaseSequence` subclasses
  (`StateSequence`, `EventSequence`, `IntervalSequence`). Call it without
  parentheses: `seq.sequence_ids` not `seq.sequence_ids()`.
- `IntervalSequence.to_state_sequence()` now returns `StateSequence` instead of
  `pl.DataFrame`.

### Bug fixes
- Clustering functions (`pam_clustering`, `clara_clustering`,
  `hierarchical_clustering`) now correctly unwrap `DistanceMatrix` input via
  `.values` (previously used non-existent `.matrix` attribute).
- `yasqat.__version__` now reads the installed version from `importlib.metadata`
  instead of a hardcoded string that disagreed with `pyproject.toml`.

### Additions
- `SequenceConfig` is now exported from `yasqat` and `yasqat.core`.
- Integration test suite added (`tests/test_integration.py`).
```

---

## Step 2 — Bump version to 0.3.0

Use uv's built-in version command (updates `pyproject.toml` in place):

```bash
uv version 0.3.0
```

Verify:

```bash
uv run python -c "import yasqat; print(yasqat.__version__)"
# → 0.3.0
```

> `__version__` reads from `importlib.metadata` so it will reflect the new
> version only after reinstalling (editable install picks it up immediately).

---

## Step 3 — Final quality gate

```bash
uv run pytest                          # 532 tests must pass
uv run ruff check src/ tests/          # zero errors
uv run mypy src/yasqat/               # zero errors (with configured overrides)
```

Do not proceed past this step with any failures.

---

## Step 4 — Merge `dev` → `main` and tag

```bash
git checkout main
git merge --no-ff dev -m "release: 0.3.0"
git tag v0.3.0
# Do NOT push yet — verify build first (step 5)
```

---

## Step 5 — Build the distributions

```bash
uv build
```

This creates two files in `dist/`:
- `yasqat-0.3.0.tar.gz` — source distribution (sdist)
- `yasqat-0.3.0-py3-none-any.whl` — wheel

**Verify the build is self-contained** (no reliance on local source paths):

```bash
uv build --no-sources
```

Inspect the wheel contents:

```bash
unzip -l dist/yasqat-0.3.0-py3-none-any.whl | grep -v __pycache__
```

Check that `yasqat/` is present and no `tests/` appear in the wheel (tests only
go in the sdist, which is correct per the current `pyproject.toml`).

---

## Step 6 — Smoke-test on TestPyPI ~~SKIPPED~~

> **Status: skipped (2026-02-23)**
>
> TestPyPI's `/legacy/` upload endpoint returned HTTP 503 with its own
> maintenance page on two separate attempts. The status page (status.python.org)
> showed no reported outage — the upload endpoint appears to have a partial
> outage not tracked there.
>
> Mitigations applied before proceeding to real PyPI:
> - Confirmed `yasqat` name is free on TestPyPI (`curl` to the JSON API → 404).
> - Verified the wheel installs cleanly from the local `dist/` directory.
> - All 532 tests pass; `ruff` and `mypy` are clean.
>
> Risk accepted: proceed directly to step 7.

### Local wheel verification (substitute for 6d)

```bash
uv run --with dist/yasqat-0.3.0-py3-none-any.whl --no-project \
  -- python -c "import yasqat; print(yasqat.__version__)"
# → 0.3.0
```

---

## Step 7 — Publish to PyPI

### 7a. Create a PyPI account and API token

1. Register at https://pypi.org/account/register/
2. Enable 2FA (required for new accounts)
3. Go to Account Settings → API tokens → Add API token
4. Scope: "Entire account" for first upload, then narrow to project

### 7b. Publish

```bash
uv publish --token pypi-<YOUR_PYPI_TOKEN>
```

Or:

```bash
export UV_PUBLISH_TOKEN=pypi-<YOUR_PYPI_TOKEN>
uv publish
```

### 7c. Verify the live release

```bash
uv run --with yasqat --no-project \
       -- python -c "import yasqat; print(yasqat.__version__)"
# → 0.3.0
```

Check the PyPI page at https://pypi.org/project/yasqat/

---

## Step 8 — Post-publish

```bash
# Push the release commit and tag
git push origin main
git push origin v0.3.0

# Create a GitHub Release (optional but recommended)
gh release create v0.3.0 \
  --title "yasqat 0.3.0" \
  --notes-file CHANGELOG.md \
  dist/yasqat-0.3.0.tar.gz \
  dist/yasqat-0.3.0-py3-none-any.whl
```

---

## Step 9 (optional) — Automate future releases with GitHub Actions

Create `.github/workflows/publish.yml`:

```yaml
name: Publish to PyPI

on:
  push:
    tags:
      - "v*"

jobs:
  build-and-publish:
    runs-on: ubuntu-latest

    permissions:
      id-token: write   # required for OIDC trusted publishing

    steps:
      - uses: actions/checkout@v4

      - uses: astral-sh/setup-uv@v5

      - name: Run tests
        run: uv run pytest

      - name: Build
        run: uv build --no-sources

      - name: Publish
        run: uv publish
        # Uses OIDC trusted publishing — no token needed if configured on PyPI
        # Otherwise: env: UV_PUBLISH_TOKEN: ${{ secrets.PYPI_API_TOKEN }}
```

To use OIDC trusted publishing (no stored secrets):
1. Go to https://pypi.org/manage/project/yasqat/settings/publishing/
2. Add a trusted publisher: GitHub, owner `rexarski`, repo `yasqat`,
   workflow `publish.yml`, environment (leave blank or set one)

---

## Summary

| Step | Command | Notes |
|------|---------|-------|
| 1 | Write `CHANGELOG.md` | Manual |
| 2 | `uv version 0.3.0` | Bumps `pyproject.toml` |
| 3 | `uv run pytest && ruff check && mypy` | Must be green |
| 4 | `git merge dev` + `git tag v0.3.0` | Don't push yet |
| 5 | `uv build --no-sources` | Creates `dist/` |
| 6 | `uv publish --index testpypi` | Smoke-test first |
| 7 | `uv publish` | Real PyPI |
| 8 | `git push origin main v0.3.0` | Then GitHub Release |
| 9 | Add `publish.yml` | Automate future releases |

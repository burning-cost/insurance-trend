# Changelog

## [0.1.4] - 2026-03-27

### Fixed
- Switch PELT break detection from `model="l2"` to `model="rbf"`. The L2 cost
  function detects mean shifts, which fails to fire when log-frequency or
  log-severity series have non-zero within-segment slopes (which they always do
  in practice). RBF is kernel-based and detects distributional changes
  irrespective of local slope, correctly firing on the COVID lockdown step-down
  and similar large actuarial events.
- Fixed misleading README quickstart comment `# e.g. 0.085 — 8.5% pa loss cost
  trend` which was never reachable with the example data. Updated to reflect the
  actual output of the example (~4.7% pa) after the break detection fix.
- Updated README Structural breaks section to document the RBF model choice and
  explain why L2 is inappropriate for trending insurance series.
- Updated README Performance table to reflect correct break detection results
  with the RBF model.


## [0.1.3] - 2026-03-23

### Fixed
- Bumped numpy minimum version from >=1.24 to >=1.25 to ensure compatibility with scipy's use of numpy.exceptions (added in numpy 1.25)


## v0.1.2 (2026-03-22) [unreleased]
- Add quickstart notebook and Colab badge
- Add missing MIT licence footer; remove emoji from discussion CTA
- fix(benchmark): rebuild break detection benchmark so it actually fires
- fix: sync __version__ with pyproject.toml (0.1.0 -> 0.1.2)

## v0.1.2 (2026-03-21)
- Add cross-links to related libraries in README
- docs: replace pip install with uv add in README
- Fix benchmark: break detection now fires on 36-quarter step-change DGP
- Add community CTA to README
- Add MIT license
- Add PyPI classifiers for financial/insurance audience
- Update Performance section with post-review benchmark results
- Fix scipy>=1.11 floor for Python 3.12 compatibility
- Fix P0/P1 bugs: superimposed inflation double-deflation, piecewise bootstrap CI, seasonal phase reset, docstring/warning issues (v0.1.2)
- Add benchmark: automated break detection vs naive OLS trend line
- fix: remove scipy<1.11 upper bound — incompatible with Python 3.12
- pin statsmodels>=0.14.5 for scipy compat
- docs: add Databricks notebook link
- Fix: pin scipy<1.11 for Databricks serverless compat
- Add Related Libraries section to README
- fix(docs): make quick-start self-contained — remove undefined df

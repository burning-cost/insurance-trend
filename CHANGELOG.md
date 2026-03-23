# Changelog

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


# Changelog

## [0.1.6] - 2026-04-01

### Added
- `BreakEventCalendar`: new Phase 3 class for attributing detected structural
  breaks to known UK insurance market events. Ships with a built-in calendar of
  22 major UK personal lines events covering Ogden rate changes (2017, 2019),
  IPT rises (2015, 2016, 2017), COVID lockdowns (2020Q1, 2021Q1) and
  post-lockdown bounce, Civil Liability Act whiplash tariff (2021Q2), GIPP
  dual pricing ban (2022Q1), semiconductor supply chain shock (2021Q3), FCA
  Consumer Duty (2023Q3), and more.
- `CalendarEvent`: frozen dataclass holding a single registry entry (period,
  description, category, impact direction, optional source citation).
- `BreakAttribution`: result dataclass for a single break → event match.
- `AttributionReport`: full attribution result with `summary()` and
  `to_dataframe()` helpers.
- `BreakEventCalendar.attribute()`: match a list of break period strings to
  calendar events within a configurable tolerance window.
- `BreakEventCalendar.attribute_indices()`: convenience wrapper for integer
  break indices returned by `detect_breakpoints()`.
- `BreakEventCalendar.filter_events()`: return a sub-calendar filtered by
  category, impact, and/or date range.
- `BreakEventCalendar.events_dataframe()`: export all events as a Polars
  DataFrame.


## [0.1.5] - 2026-03-31

### Added
- `InflationDecomposer`: Harvey structural time series decomposition of claims
  inflation into structural trend, stochastic cycle, seasonal, and irregular
  components using statsmodels UnobservedComponents (Kalman filter/smoother).
- `InflationDecompositionResult` dataclass with `summary()`, `decomposition_table()`,
  and `plot()` helpers.
- Databricks demo notebook: `notebooks/demo_inflation_decomposer.py`.


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

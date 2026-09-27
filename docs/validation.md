# Release validation

Local checks for version 0.1.0 were performed on macOS arm64 with Python
3.12.9 on 2026-09-26. This records what was actually checked. Subsequent
cross-platform runs are available in the repository's
[GitHub Actions history](https://github.com/liangling98/GWIPG/actions).

## Installation and tests

| Environment | NumPy | SciPy | POT | Result |
| --- | --- | --- | --- | --- |
| Local reproduction environment | 2.2.4 | 1.15.3 | 0.9.6 | 51 tests passed |
| Fresh virtual environment, installed wheel | 2.5.3 | 1.18.1 | 0.9.7.post1 | 51 tests passed |

The fresh environment also used pandas 3.0.6, scikit-learn 1.9.1, NetworkX
3.7, and Matplotlib 3.11.2. A core-only wheel installation imported and solved
a small problem outside the source tree before installing the example extras.
Wheel and source builds passed `twine check`; Ruff lint and format checks passed.

Numerical tests cover an independently solved constrained projection problem,
analytic projections, nonuniform marginals and non-unit mass, symmetric and
asymmetric GW regressions, degenerate steps, input validation, and reporting of
unsatisfied projection criteria. CLI and output tests cover benchmark wiring,
offline/cache behavior, metrics, and archived-result checksums.

The README and script quick starts now have explicit convergence tests.
With a 10,000-step outer budget, tolerance 1e-6, constant Lipschitz steps,
and `projection_failure="raise"`, the seeded 12-by-15 example reaches
`status="converged"` with every projection passing its residual test. The
local reproduction run used 5,241 outer steps and returned marginal residual
3.53e-9. The earlier 100-step budget stopped too soon.

Windows and Linux/Python 3.10 CI exposed variations up to 6.2e-8 in the
approximate symmetric regression coupling. The portable comparison now
checks the coupling and marginals at absolute accuracy 1e-6 and separately
checks its substantial support; the objective tolerance and frozen baseline
values are unchanged. See `tests/fixtures/README.md`.

## Examples and archived results

- All six `gwipg-paper all --quick` pipelines ran successfully from the
  installed wheel outside the source tree, using a verified offline data cache.
  Each wrote result files and configuration metadata. These short runs test
  execution and finite outputs, not numerical convergence.
- The quick-start, rectangular projection, graph plotting, and optional
  low-rank examples ran. The projection example met its residual target.
- A fresh download of all three SCOT input files passed checksum validation.
- SNARE-seq iPG n=250 and n=1047, each with seed 0, reproduced archived loss,
  feasibility, projection-call count, and mean inner iterations in the local
  reproduction environment. The n=50, seed 0, S¹/S² run also reproduced those
  archived quantities. Full multi-seed sweeps were not rerun during cleanup.
- Original-versus-packaged solver comparisons on symmetric and asymmetric
  inputs agreed exactly within each tested environment. Frozen regression
  fixtures allow small differences across numerical-library versions; see
  `tests/fixtures/README.md`.

The local environment reports an upstream POT/JAX deprecation warning. Quick
runs may also report unsatisfied projection criteria because their budgets are
deliberately small; these warnings are retained rather than suppressed.

## Optimization check

For a 500-by-500 adjacency matrix, edge-flip probability 0.1, and seed 42,
the old and vectorized graph-noise functions produced identical arrays.
Median time over seven local runs decreased from 24.01 ms to 1.21 ms, about
20 times faster for this preprocessing function. This is not a claim about
end-to-end GW solver speed. The solver also avoids a duplicate initial
gradient and reuses projection scratch arrays.

## Preservation checks

SHA-256 comparisons confirmed that all six manuscript source/bibliography/style
files and 200 existing figure, dataset, and result files were unchanged.
Archived paper CSVs are copied verbatim and covered by a checksum manifest.
The separate SNARE-seq manuscript setting discrepancy is described in
[review notes](review_notes.md); no manuscript correction was made here.

# Code review and migration notes

This release reorganizes the research scripts into an installable package.
The original source and all experiment outputs were backed up before changes.
No manuscript source or existing figure asset was changed by this code review.

## Correctness fixes

- Validate finite real cost matrices, dimensions, nonnegative marginals,
  positive equal mass, initialization, and iteration/tolerance parameters.
- Handle zero-cost inputs without an infinite step, and normalize the default
  product coupling for non-unit total mass.
- Skip degenerate/nonfinite secant updates rather than dividing by zero.
  Preserve the original lagged update timing for reproducibility; a constant
  step is available through `adaptive_step=False`.
- Recheck the returned projection residual after the final inner update.
  Distinguish budget hits from failed residual tests, expose convergence
  status, and optionally refuse inexact projections.
- Check symmetry with zero relative tolerance. Nearly asymmetric matrices
  no longer silently enter the symmetric branch.
- Reject non-finite benchmark outputs and remove process-wide warning
  suppression. Solver warnings are visible; archived results are unchanged.
- Remove graph-run global marginals, respect `normalize_cost=False`, prevent
  unbounded graph resampling, and report fully failed settings with zero
  successes instead of a NaN failure count.
- Correct peak RSS unit conversion across macOS/Linux and provide a current
  RSS fallback on Windows for the optional low-rank benchmark.

## Optimizations that preserve the numerical iteration

- Avoid the duplicate first GW-gradient evaluation.
- Reuse the projection's matrix and residual scratch buffers across outer
  iterations and recover primal matrices without chained full-size temporaries.
- Collect small projection statistics instead of full inner trajectories for
  outer progress reporting, and skip component timer calls unless requested.
- Vectorize graph edge flips while preserving seeded random draws exactly.
- Remove unused chromatin normalization from each application-metric evaluation.

## Reproducibility finding requiring manuscript review

The SNARE-seq appendix currently says `epsilon_k=(k+1)^(-3)`. An original-code
and packaged-code rerun of n=250, seed 0 matches the archived record with
`epsilon_k=0.05*(k+1)^(-2.1)` and projection step multiplier 1.5 instead:

| Quantity | Reproduced archived value |
| --- | --- |
| GW loss | 0.01805603271909921 |
| Marginal feasibility | 2.5496531840323614e-7 |
| Outer/projection calls | 166 |
| Mean inner iterations | 246.06626506024097 |

The exponent-3 alternatives did not reproduce this run. The presets record the
matching settings; the manuscript was left for a separately approved edit.
The four-exponent ablation legitimately uses scale 1 and is distinct.

A packaged full-data run (n=1047, seed 0) independently matched the archived
loss `0.03504169674065518`, feasibility `4.810099035582104e-6`, 41 projection
calls, and mean 184.1219512195122 inner iterations with the same settings.

## API migration

Install the project and keep `from gwipg import gwipg`. For projections use
`from gwipg import proj_transport_polytope`. The solver remains quiet by
default and still returns `(T, log)` unless `log=False` is requested.

Old top-level benchmark scripts now live under `gwipg.benchmarks` with thin
launchers under `examples/`. For example, `GW_sphere.py` becomes
`python examples/spheres.py` or `gwipg-spheres`; `test_graph_alignment.py` is
now an actual benchmark, while `tests/` contains automated tests.

Output defaults use the caller's working directory, not installed package
directories. Downloaded data and fresh outputs are excluded from Git and
build artifacts. Frozen regression fixtures were generated from a preserved
copy of the original solver on symmetric and asymmetric inputs and include
iterations beyond the heuristic-update boundary at iteration 100. Their
provenance is documented in `tests/fixtures/README.md`.

# Solver API and numerical conventions

```python
from gwipg import gwipg, proj_transport_polytope
```

The library implements dense, CPU, float64 transport. `C1` and `C2` may have
different sizes; their shapes are `(m, m)` and `(n, n)`. Marginals are
nonnegative vectors of equal positive mass. Omitting both gives uniform
probability vectors. Inputs are validated and never modified. Cost matrices
can be asymmetric; symmetry is detected unless explicitly specified.

## GW solver

```python
T, info = gwipg(C1, C2, p, q, log=True, verbose=False)
T = gwipg(C1, C2, p, q, log=False)
```

`help(gwipg)` lists all parameters. The README's
[complete option table](../README.md#solver-options) includes every argument,
accepted value, and default. Existing paper parameter names are retained:

| Parameter | Default | Meaning |
| --- | --- | --- |
| `max_iter` | 10000 | Outer iteration budget |
| `max_iter_proj` | 5000 | Inner residual-evaluation budget per projection |
| `tol` | 1e-9 | **Absolute** successive-iterate Frobenius difference |
| `iter_check` | 100 | Check at zero-based outer indices 0, 100, 200, ... |
| `expon` | 2.1 | Inner tolerance decay exponent |
| `proj_tol_scale` | 1 | Numerator scale in the inner tolerance schedule |
| `lr_const` | 1 | Initial step multiplier over `4*||C1||₂*||C2||₂` |
| `proj_lr_const` | 2 | Projection step multiplier over `m+n` |
| `adaptive_step` | True | Original practical secant-step heuristic |
| `projection_failure` | `"warn"` | `"warn"`, `"raise"`, or `"ignore"` |
| `log` | True | Return a log with the coupling |
| `verbose` | False | Print progress |

The paper implementation's adaptive update uses lagged gradient differences.
That timing is retained for reproducibility, with guards for zero or nonfinite
denominators. `adaptive_step=False` uses a constant step. Use `lr_const=1`
and `proj_lr_const=1` for the usual outer and inner Lipschitz step sizes;
the larger paper steps are empirical settings.

The inner schedule is `proj_tol_scale/(k+1)**expon`, with `k` starting at zero.
The projection tests the Euclidean norm of the concatenated marginal
residual against that schedule divided by `6 + ||y||₂`. The reported
`feasibility` instead sums the two marginal residual norms, matching the
paper's tables. These quantities should not be compared as if identical.

### Logs and stopping

| Key | Meaning |
| --- | --- |
| `gw_dist` | POT's full squared-loss score; no root or factor of 1/2 |
| `err`, `loss` | Values recorded at outer stopping checks |
| `iter`, `n_iter` | Last zero-based outer index and number of outer steps |
| `final_err` | Actual final successive-iterate difference, even off a check index |
| `feasibility` | Sum of row and column marginal residual norms |
| `outer_converged` | An outer stopping check passed |
| `final_projection_converged` | The returned inner primal/dual pair passed its residual test |
| `converged` | Both the outer and final inner checks passed |
| `all_projections_converged` | Every inner residual check passed |
| `projection_criterion_failures` | Count of returned inner pairs failing that test |
| `status` | `converged`, `max_iter`, or `inexact_projection` |
| `final_projection_residual`, `final_projection_threshold` | Final inner residual and target |
| `step_size`, `skipped_step_updates` | Last outer step and rejected degenerate secant updates |

With `track_proj_iters=True`, the log additionally includes `proj_iters`,
`proj_calls`, `proj_iter_mean`, `proj_iter_max`, `proj_cap_hits`, and
`proj_cap_hit_fraction`. A cap hit counts use of the full budget and is
**independent** of whether the final residual passes. The legacy paper's
saved cap counts alone cannot certify the residual of each capped call.

With `track_timing=True`, `solver_internal_time`, `gw_gradient_time`,
`projection_time`, `other_time`, and component time fractions are recorded.
Timing is disabled by default to avoid per-iteration timer overhead.

Neither a small successive change nor these convergence flags certify global
optimality. A failed inner test can return an infeasible coupling. Set
`projection_failure="raise"` to refuse such steps; the default reports one
warning per solver call. Theoretical guarantees require the theorem's other
assumptions as well. In particular, heuristic large steps are not certified
by simply making the projections accurate.

For infeasible couplings, the POT fixed-marginal score is not a directly
comparable feasible GW objective. Inspect feasibility alongside losses, as
the paper does for BAPG. The algorithm stores dense matrices even when the
returned coupling is sparse.

## Euclidean projection

```python
T, dual, info = proj_transport_polytope(
    X, p, q, tol=1e-8, max_iter=10000, lr_const=1,
    return_log=False, return_stats=True,
)
```

The projection minimizes `0.5*||T-X||_F**2` over nonnegative matrices with
the requested marginals. Pass `dual` back as `y0` to warm-start another
projection. `return_log=True` additionally records residuals in `err`;
disabling both return flags gives `(T, dual)`.

`n_iter` counts residual evaluations. `iter` retains the old zero-based
index, or `max_iter` on exhaustion, for compatibility. Use `n_iter` in new
code. `converged`, `final_err`, `threshold`, and `hit_iteration_limit` refer
to the actually returned primal/dual pair, including after budget exhaustion.

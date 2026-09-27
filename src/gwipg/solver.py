"""Dense NumPy implementation of inexact projected-gradient GW transport."""

import time
import warnings

import numpy as np
import ot

from ._validation import array, marginals, positive_float, positive_int
from .projection import _project


class ProjectionConvergenceWarning(RuntimeWarning):
    """At least one projection did not meet its residual stopping target."""


def gwipg(
    C1,
    C2,
    p=None,
    q=None,
    G0=None,
    max_iter=10000,
    max_iter_proj=5000,
    tol=1e-9,
    iter_check=100,
    verbose=False,
    log=True,
    expon=2.1,
    lr_const=1.0,
    proj_tol_scale=1.0,
    proj_lr_const=2.0,
    track_proj_iters=False,
    track_timing=False,
    symmetric=None,
    *,
    adaptive_step=True,
    projection_failure="warn",
):
    r"""Solve the squared-loss Gromov-Wasserstein problem approximately.

    Parameters
    ----------
    C1, C2 : array-like, shapes (m, m) and (n, n)
        Finite cost/dissimilarity matrices. Symmetric and asymmetric costs
        are supported. Computation uses dense NumPy float64 arrays on CPU.
    p, q : array-like, optional
        Nonnegative marginals of equal positive mass; defaults are uniform.
    G0 : array-like, shape (m, n), optional
        Nonnegative initial coupling. Defaults to ``outer(p, q) / p.sum()``.
    max_iter, max_iter_proj : int
        Outer and per-projection iteration budgets, respectively.
    tol : float
        Absolute successive-iterate Frobenius norm stopping tolerance.
    iter_check : int
        Check the outer tolerance at zero-based iterations 0, iter_check,
        2*iter_check, etc., retaining the paper's stopping schedule.
    verbose, log : bool
        Print progress; return a diagnostics dictionary. With ``log=False``
        only the coupling is returned. The default is quiet with logging.
    expon, proj_tol_scale : float
        Inner schedule is ``proj_tol_scale / (k+1)**expon``. The projection
        tests this divided by ``6 + ||y||_2`` against its marginal residual.
    lr_const : float
        Initial outer step multiplier: ``lr_const / (4*||C1||_2*||C2||_2)``.
    proj_lr_const : float
        Dual projection step multiplier. See ``proj_transport_polytope``.
    track_proj_iters, track_timing : bool
        Store each projection's iteration count and/or component timings.
        Projection failure counts are always available when ``log=True``.
    symmetric : bool, optional
        Infer symmetry by default. ``True`` requires symmetric costs;
        ``False`` forces the general POT gradient.
    adaptive_step : bool
        Retain the original paper code's heuristic secant step updates,
        including their lagged-gradient timing. Set False for a constant
        step. Updates with a zero/nonfinite denominator are skipped.
    projection_failure : {"warn", "raise", "ignore"}
        Report failed inner residual checks once at the end, raise on the
        first failure, or continue quietly. A cap is not proof of convergence.

    Returns
    -------
    T : numpy.ndarray, shape (m, n)
        Nonnegative coupling; a capped inner solve may leave it infeasible.
    history : dict, optional
        ``gw_dist`` is POT's squared-loss score, not its square root.
        ``converged`` requires the outer change and final inner residual
        tests to pass. ``all_projections_converged`` records whether every
        inner solve passed. Neither flag certifies global optimality.

    Notes
    -----
    Heuristic large steps and capped projections in the paper experiments
    are practical settings, not a check of all theorem assumptions. For a
    constant Lipschitz step use ``adaptive_step=False, lr_const=1``;
    use ``projection_failure='raise'`` to enforce inner residual checks.
    See ``docs/api.md`` for the complete log schema and stopping semantics.
    """
    start_time = time.perf_counter() if track_timing else 0.0
    C1, C2 = array(C1, "C1", 2), array(C2, "C2", 2)
    if C1.shape[0] != C1.shape[1] or C2.shape[0] != C2.shape[1]:
        raise ValueError("C1 and C2 must be square")
    m, n = len(C1), len(C2)
    p, q = marginals(p, q, m, n)
    max_iter = positive_int(max_iter, "max_iter")
    max_iter_proj = positive_int(max_iter_proj, "max_iter_proj")
    iter_check = positive_int(iter_check, "iter_check")
    tol = positive_float(tol, "tol", allow_zero=True)
    expon = positive_float(expon, "expon")
    lr_const = positive_float(lr_const, "lr_const")
    proj_tol_scale = positive_float(proj_tol_scale, "proj_tol_scale")
    proj_lr_const = positive_float(proj_lr_const, "proj_lr_const")
    if projection_failure not in {"warn", "raise", "ignore"}:
        raise ValueError("projection_failure must be 'warn', 'raise', or 'ignore'")
    if symmetric is not None and not isinstance(symmetric, (bool, np.bool_)):
        raise ValueError("symmetric must be True, False, or None")
    is_symmetric = np.allclose(C1, C1.T, rtol=0, atol=1e-14) and np.allclose(
        C2, C2.T, rtol=0, atol=1e-14
    )
    if symmetric is not None and bool(symmetric) and not is_symmetric:
        raise ValueError("symmetric=True requires symmetric cost matrices")
    symmetric = is_symmetric if symmetric is None else symmetric
    if G0 is None:
        T = np.outer(p, q)
        # Preserve the original arithmetic exactly for unit-mass inputs.
        if not np.isclose(p.sum(), 1.0, rtol=1e-12, atol=0):
            T /= p.sum()
    else:
        T = array(G0, "G0", 2).copy()
        if T.shape != (m, n) or np.any(T < 0):
            raise ValueError(f"G0 must be nonnegative with shape {(m, n)}")
    y = np.zeros(m + n)
    workspace = (np.empty(m + n), np.empty_like(T))
    constC, hC1, hC2 = ot.gromov.init_matrix(C1, C2, p, q, loss_fun="square_loss")

    def gw_loss(plan):
        return float(ot.gromov.gwloss(constC, hC1, hC2, plan))

    if symmetric:

        def gw_grad(plan):
            return ot.gromov.gwggrad(constC, hC1, hC2, plan)
    else:
        constCt, hC1t, hC2t = ot.gromov.init_matrix(C1.T, C2.T, p, q, "square_loss")

        def gw_grad(plan):
            return 0.5 * (
                ot.gromov.gwggrad(constC, hC1, hC2, plan)
                + ot.gromov.gwggrad(constCt, hC1t, hC2t, plan)
            )

    lipschitz = 4.0 * np.linalg.norm(C1, 2) * np.linalg.norm(C2, 2)
    if not np.isfinite(lipschitz):
        raise ValueError("cost scale is too large; rescale C1 and C2")
    lr = lr_const / lipschitz if lipschitz > 0 else 1.0
    if not np.isfinite(lr):
        raise ValueError("cost scale is too small; rescale C1 and C2")
    err = np.inf
    history = {"err": [], "loss": []} if log else None
    projection_iters = [] if track_proj_iters and log else None
    gradient_time = projection_time = 0.0
    timer = time.perf_counter() if track_timing else 0.0
    dfT = gw_grad(T)
    if track_timing:
        gradient_time += time.perf_counter() - timer
    failures = 0
    skipped_step_updates = 0
    outer_converged = False
    Tprev, dfTprev = T, dfT

    for cpt in range(max_iter):
        if adaptive_step and err < 1e-4 and cpt % 100 == 0 and cpt >= 100:
            denominator = np.linalg.norm(dfT - dfTprev, "fro")
            numerator = np.linalg.norm(T - Tprev, "fro")
            candidate = numerator / denominator if denominator > 0 else np.nan
            if np.isfinite(candidate) and candidate > 0:
                lr = candidate
            else:
                skipped_step_updates += 1
        Tprev, dfTprev = T, dfT
        # The first gradient was already computed above; avoid a duplicate
        # dense matrix product without changing the subsequent trajectory.
        if cpt:
            timer = time.perf_counter() if track_timing else 0.0
            dfT = gw_grad(Tprev)
            if track_timing:
                gradient_time += time.perf_counter() - timer
        Tin = Tprev - lr * dfT
        tol_proj = proj_tol_scale * np.exp(-expon * np.log(cpt + 1.0))
        # Direct division reproduces the original schedule where representable.
        try:
            tol_proj = proj_tol_scale / (cpt + 1) ** expon
        except OverflowError:
            pass
        timer = time.perf_counter() if track_timing else 0.0
        T, y, stats = _project(
            Tin,
            p,
            q,
            y,
            max_iter_proj,
            tol_proj,
            lr_const=proj_lr_const,
            workspace=workspace,
        )
        if track_timing:
            projection_time += time.perf_counter() - timer
        if not stats["converged"]:
            failures += 1
            if projection_failure == "raise":
                raise RuntimeError(
                    f"projection {cpt} failed its residual test: "
                    f"{stats['final_err']:.3e} > {stats['threshold']:.3e}"
                )
        if projection_iters is not None:
            projection_iters.append(stats["n_iter"])
        if cpt % iter_check == 0:
            err = float(np.linalg.norm(T - Tprev, "fro"))
            if log or verbose:
                loss = gw_loss(T)
            if history is not None:
                history["err"].append(err)
                history["loss"].append(loss)
            if verbose:
                print(
                    f"iter={cpt:4d}, err={err:.2e}, loss={loss:.8e}, "
                    f"iter_proj={stats['n_iter']}, lr={lr:.1e}"
                )
            if err <= tol:
                outer_converged = True
                break

    if failures and projection_failure == "warn":
        warnings.warn(
            f"{failures} of {cpt + 1} projections did not meet the inner residual target. "
            "Increase max_iter_proj or inspect the convergence diagnostics.",
            ProjectionConvergenceWarning,
            stacklevel=2,
        )
    if history is None:
        return T
    final_loss = gw_loss(T)
    final_err = float(np.linalg.norm(T - Tprev, "fro"))
    converged = outer_converged and stats["converged"]
    history.update(
        gw_dist=final_loss,
        iter=cpt,
        n_iter=cpt + 1,
        final_err=final_err,
        converged=converged,
        outer_converged=outer_converged,
        status="converged"
        if converged
        else ("inexact_projection" if outer_converged else "max_iter"),
        projection_criterion_failures=failures,
        all_projections_converged=failures == 0,
        final_projection_converged=stats["converged"],
        final_projection_residual=stats["final_err"],
        final_projection_threshold=stats["threshold"],
        feasibility=float(np.linalg.norm(T.sum(axis=1) - p) + np.linalg.norm(T.sum(axis=0) - q)),
        step_size=float(lr),
        skipped_step_updates=skipped_step_updates,
    )
    if projection_iters is not None:
        counts = np.asarray(projection_iters)
        hits = int(np.count_nonzero(counts >= max_iter_proj))
        history.update(
            proj_iters=projection_iters,
            proj_calls=len(counts),
            proj_iter_mean=float(counts.mean()),
            proj_iter_max=int(counts.max()),
            proj_cap_hits=hits,
            proj_cap_hit_fraction=hits / len(counts),
        )
    if track_timing:
        elapsed = time.perf_counter() - start_time
        history.update(
            solver_internal_time=elapsed,
            gw_gradient_time=gradient_time,
            projection_time=projection_time,
            other_time=max(0.0, elapsed - gradient_time - projection_time),
            gw_gradient_time_fraction=gradient_time / elapsed,
            projection_time_fraction=projection_time / elapsed,
        )
    return T, history

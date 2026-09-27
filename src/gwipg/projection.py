"""Euclidean projection onto a transport polytope using its smooth dual."""

import numpy as np

from ._validation import array, marginals, positive_float, positive_int


def proj_transport_polytope(
    Tin,
    p,
    q,
    y0=None,
    max_iter=1000,
    tol=1e-9,
    verbose=False,
    return_log=True,
    return_stats=False,
    accelerated=True,
    restart=True,
    restart_iter=500,
    lr_const=2.0,
):
    r"""Approximately project ``Tin`` onto ``{T >= 0: T1=p, T.T1=q}``.

    Parameters
    ----------
    Tin : array-like, shape (m, n)
        Real matrix to project; it need not be nonnegative.
    p, q : array-like
        Nonnegative marginals with equal positive total mass.
    y0 : array-like, shape (m+n,), optional
        Initial dual row and column potentials; useful for warm starts.
    max_iter : int
        Maximum number of dual residual evaluations.
    tol : float
        The stopping target is ``tol / (6 + ||y||_2)`` for the combined
        Euclidean marginal residual. It is not an unscaled primal tolerance.
    verbose : bool
        Print residuals during the inner solve.
    return_log, return_stats : bool
        Return a third dictionary when either flag is true. ``return_log``
        additionally stores the residual trajectory in ``err``.
    accelerated, restart : bool
        Use Nesterov acceleration and periodic momentum restarts.
    restart_iter : int
        Restart interval (positive even when restarting is disabled).
    lr_const : float
        Dual step is ``lr_const / (m+n)``. The paper uses the practical
        value 2; use at most 1 for the usual Lipschitz step for AGM.

    Returns
    -------
    T, y : numpy.ndarray
        Approximate primal projection and final dual potentials.
    stats : dict, optional
        ``converged`` checks the returned primal/dual pair against the
        stopping target. ``hit_iteration_limit`` means the full budget was
        used and is independent of convergence. ``n_iter`` is the number
        of evaluations. Legacy ``iter`` is zero-based, or ``max_iter`` on
        exhaustion, and is retained for old experiment readers.

    Notes
    -----
    An iteration limit can return an infeasible matrix. Check ``converged``
    and ``final_err`` when an accurate Euclidean projection is required.
    Inputs are never modified.
    """
    Tin = array(Tin, "Tin", 2)
    p, q = marginals(p, q, *Tin.shape)
    max_iter = positive_int(max_iter, "max_iter")
    restart_iter = positive_int(restart_iter, "restart_iter")
    tol = positive_float(tol, "tol", allow_zero=True)
    lr_const = positive_float(lr_const, "lr_const")
    if y0 is not None:
        y0 = array(y0, "y0", 1)
        if y0.shape != (sum(Tin.shape),):
            raise ValueError(f"y0 must have shape {(sum(Tin.shape),)}")
    T, y, stats = _project(
        Tin,
        p,
        q,
        y0,
        max_iter,
        tol,
        verbose,
        return_log,
        accelerated,
        restart,
        restart_iter,
        lr_const,
    )
    return (T, y, stats) if return_log or return_stats else (T, y)


def _project(
    Tin,
    p,
    q,
    y0,
    max_iter,
    tol,
    verbose=False,
    return_log=False,
    accelerated=True,
    restart=True,
    restart_iter=500,
    lr_const=2.0,
    workspace=None,
):
    """Validated kernel; GW iterations can reuse the large scratch arrays."""
    m, n = Tin.shape
    lr = lr_const / (m + n)
    y = np.zeros(m + n) if y0 is None else y0.copy()
    z = y.copy()
    t = 1.0
    if workspace is None:
        grad, Tz = np.empty(m + n), np.empty_like(Tin)
    else:
        grad, Tz = workspace
    history = {"err": []} if return_log else {}
    exhausted = True

    for k in range(max_iter):
        np.copyto(Tz, Tin)
        Tz += z[:m, None]
        Tz += z[None, m:]
        np.maximum(Tz, 0.0, out=Tz)
        grad[:m] = Tz.sum(axis=1) - p
        grad[m:] = Tz.sum(axis=0) - q
        err = np.linalg.norm(grad)
        normz = np.linalg.norm(z)
        if not np.isfinite(err + normz):
            raise FloatingPointError("projection diverged; reduce lr_const or rescale inputs")
        if return_log:
            history["err"].append(float(err))
        if verbose:
            print(f"iter={k:4d}, err={err:.2e}, normy={normz:.2e}")
        if err <= tol / (6.0 + normz):
            y = z
            exhausted = False
            break

        y_next = z - lr * grad
        t_next = 0.5 * (1.0 + np.sqrt(1.0 + 4.0 * t * t)) if accelerated else t
        if restart and (k + 1) % restart_iter == 0:
            t = t_next = 1.0
        beta = (t - 1.0) / t_next
        z = y_next + beta * (y_next - y)
        y, t = y_next, t_next

    if not exhausted:
        # The residual was already checked at this pair. Copy to prevent
        # aliasing the scratch space reused by the next outer iteration.
        Tproj = Tz.copy()
    else:
        Tproj = np.empty_like(Tin)
        np.add(Tin, y[:m, None], out=Tproj)
        Tproj += y[None, m:]
        np.maximum(Tproj, 0.0, out=Tproj)
    grad[:m] = Tproj.sum(axis=1) - p
    grad[m:] = Tproj.sum(axis=0) - q
    final_err = float(np.linalg.norm(grad))
    threshold = float(tol / (6.0 + np.linalg.norm(y)))
    if not np.isfinite(final_err) or not np.isfinite(y).all():
        raise FloatingPointError("projection returned non-finite values")
    converged = final_err <= threshold
    history.update(
        iter=max_iter if exhausted else k,
        n_iter=k + 1,
        final_err=final_err,
        threshold=threshold,
        converged=converged,
        hit_iteration_limit=k + 1 == max_iter,
        status="converged" if converged else "max_iter",
    )
    return Tproj, y, history

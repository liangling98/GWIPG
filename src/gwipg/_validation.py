"""Shared input validation, kept outside the numerical iteration loops."""

from numbers import Integral, Real

import numpy as np


def array(value, name, ndim):
    if np.iscomplexobj(value):
        raise ValueError(f"{name} must contain real values")
    result = np.asarray(value, dtype=float)
    if result.ndim != ndim or any(size == 0 for size in result.shape):
        raise ValueError(f"{name} must be a nonempty {ndim}-dimensional array")
    if not np.isfinite(result).all():
        raise ValueError(f"{name} must contain only finite values")
    return result


def positive_int(value, name):
    if isinstance(value, bool) or not isinstance(value, Integral) or value < 1:
        raise ValueError(f"{name} must be a positive integer")
    return int(value)


def positive_float(value, name, allow_zero=False):
    if (
        isinstance(value, bool)
        or not isinstance(value, Real)
        or not np.isfinite(value)
        or (value < 0 if allow_zero else value <= 0)
    ):
        adjective = "nonnegative" if allow_zero else "positive"
        raise ValueError(f"{name} must be a finite {adjective} number")
    return float(value)


def marginals(p, q, m, n):
    p = np.full(m, 1.0 / m) if p is None else array(p, "p", 1)
    q = np.full(n, 1.0 / n) if q is None else array(q, "q", 1)
    if p.shape != (m,) or q.shape != (n,):
        raise ValueError(f"p and q must have shapes {(m,)} and {(n,)}")
    if np.any(p < 0) or np.any(q < 0):
        raise ValueError("p and q must be nonnegative")
    mass_p, mass_q = float(p.sum()), float(q.sum())
    if not np.isfinite(mass_p + mass_q) or mass_p <= 0 or mass_q <= 0:
        raise ValueError("p and q must have positive finite total mass")
    if not np.isclose(mass_p, mass_q, rtol=1e-12, atol=0.0):
        raise ValueError("p and q must have equal total mass")
    return p, q

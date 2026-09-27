import numpy as np
import pytest
from scipy.optimize import minimize

from gwipg import proj_transport_polytope


@pytest.mark.parametrize("accelerated", [False, True])
def test_two_by_two_projection_matches_closed_form(accelerated):
    X = np.array([[0.3, -0.2], [0.4, 0.1]])
    p = np.array([0.5, 0.5])
    original = X.copy()
    t = np.clip(0.25 + (X[0, 0] - X[0, 1] - X[1, 0] + X[1, 1]) / 4, 0, 0.5)
    expected = np.array([[t, 0.5 - t], [0.5 - t, t]])
    T, y, info = proj_transport_polytope(
        X,
        p,
        p,
        tol=1e-10,
        lr_const=1.0,
        accelerated=accelerated,
        max_iter=10000,
    )
    np.testing.assert_allclose(T, expected, atol=1e-9)
    np.testing.assert_array_equal(X, original)
    assert info["converged"]
    assert info["final_err"] <= 1e-10 / (6 + np.linalg.norm(y))


def test_rectangular_projection_matches_independent_quadratic_program():
    rng = np.random.default_rng(2)
    X = rng.normal(size=(3, 4))
    p = np.array([0.2, 0.5, 0.3])
    q = np.array([0.1, 0.2, 0.3, 0.4])

    def constraints(x):
        T = x.reshape(3, 4)
        return np.r_[T.sum(1) - p, (T.sum(0) - q)[:-1]]

    reference = minimize(
        lambda x: 0.5 * np.sum((x - X.ravel()) ** 2),
        np.outer(p, q).ravel(),
        jac=lambda x: x - X.ravel(),
        method="SLSQP",
        bounds=[(0, None)] * 12,
        constraints={"type": "eq", "fun": constraints},
        options={"ftol": 1e-12, "maxiter": 1000},
    )
    assert reference.success
    T, _, info = proj_transport_polytope(X, p, q, max_iter=10000, tol=1e-9, lr_const=1.0)
    assert info["converged"]
    np.testing.assert_allclose(T.ravel(), reference.x, atol=1e-7)


def test_budget_hit_is_independent_of_residual_convergence():
    p = np.array([0.5, 0.5])
    T, _, info = proj_transport_polytope(np.outer(p, p), p, p, max_iter=1)
    assert info["hit_iteration_limit"] and info["converged"]
    assert info["n_iter"] == 1
    T, y, info = proj_transport_polytope(np.zeros((2, 2)), p, p, max_iter=1, tol=1e-14)
    measured = np.linalg.norm(np.r_[T.sum(1) - p, T.sum(0) - p])
    assert info["final_err"] == pytest.approx(measured)
    assert info["converged"] == (measured <= 1e-14 / (6 + np.linalg.norm(y)))


def test_zero_marginals_and_warm_start_are_supported():
    p, q = np.array([1.0, 0]), np.array([0.3, 0.2, 0.5])
    X = np.zeros((2, 3))
    T, y, info = proj_transport_polytope(X, p, q, max_iter=10000, tol=1e-8, lr_const=1)
    assert info["converged"]
    np.testing.assert_allclose(T[0], q, atol=1e-8)
    np.testing.assert_allclose(T[1], 0, atol=1e-8)
    _, _, warm = proj_transport_polytope(X, p, q, y0=y, tol=1e-8)
    assert warm["n_iter"] == 1


@pytest.mark.parametrize(
    "kwargs",
    [
        {"max_iter": 0},
        {"max_iter": 1.5},
        {"restart_iter": 0},
        {"tol": -1},
        {"lr_const": float("nan")},
        {"y0": [0]},
    ],
)
def test_invalid_parameters_raise_clear_errors(kwargs):
    with pytest.raises(ValueError):
        proj_transport_polytope(np.zeros((2, 2)), [0.5, 0.5], [0.5, 0.5], **kwargs)


def test_return_modes():
    args = (np.zeros((2, 2)), [0.5, 0.5], [0.5, 0.5])
    assert len(proj_transport_polytope(*args, return_log=False)) == 2
    *_, stats = proj_transport_polytope(*args, return_log=False, return_stats=True)
    assert "err" not in stats and "converged" in stats

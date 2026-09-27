from pathlib import Path

import numpy as np
import pytest

from gwipg import ProjectionConvergenceWarning, gwipg


@pytest.mark.parametrize("name", ["symmetric", "asymmetric"])
def test_packaging_preserves_legacy_solver_trajectory(name):
    """Frozen outputs of the original code, including iterations beyond 100."""
    data = np.load(Path(__file__).parent / "fixtures" / f"legacy_{name}.npz")
    T, info = gwipg(
        data["C1"],
        data["C2"],
        data["p"],
        data["q"],
        max_iter=205,
        max_iter_proj=500,
        tol=0.0,
        iter_check=5,
        expon=3.0,
        lr_const=70.0,
        proj_lr_const=1.5,
        track_proj_iters=True,
        projection_failure="ignore",
    )
    # These capped, adaptive solves are approximate: BLAS differences change
    # near-threshold inner stopping. Compare at 1e-6 primal accuracy, while
    # checking support, marginals, and the objective separately.
    np.testing.assert_allclose(T, data["plan"], atol=1e-6, rtol=0)
    np.testing.assert_array_equal(T > 1e-5, data["plan"] > 1e-5)
    np.testing.assert_allclose(T.sum(axis=1), data["p"], atol=1e-6, rtol=0)
    np.testing.assert_allclose(T.sum(axis=0), data["q"], atol=1e-6, rtol=0)
    assert info["gw_dist"] == pytest.approx(float(data["loss"]), abs=1e-8, rel=1e-7)
    # Allow BLAS-level differences in near-threshold inner stopping counts.
    assert len(info["proj_iters"]) == len(data["proj_iters"])


def test_zero_costs_nonunit_mass_and_input_immutability():
    C1, C2 = np.zeros((2, 2)), np.zeros((3, 3))
    p, q = np.array([0.6, 1.4]), np.array([0.4, 0.6, 1.0])
    T, info = gwipg(C1, C2, p, q, max_iter=2)
    np.testing.assert_allclose(T, np.outer(p, q) / 2)
    assert info["gw_dist"] == 0
    assert info["converged"]
    np.testing.assert_array_equal(p, [0.6, 1.4])


def test_degenerate_secant_update_keeps_a_finite_step():
    p = np.full(3, 1 / 3)
    initial = np.outer(p, p) + 1e-7 * np.eye(3)
    T, info = gwipg(
        np.zeros((3, 3)),
        np.ones((3, 3)),
        p,
        p,
        G0=initial,
        max_iter=205,
        iter_check=1000,
        tol=0,
        proj_tol_scale=1e-9,
        proj_lr_const=1,
        projection_failure="ignore",
    )
    assert np.isfinite(T).all() and np.isfinite(info["step_size"])
    assert info["skipped_step_updates"] >= 1
    np.testing.assert_allclose(initial, np.outer(p, p) + 1e-7 * np.eye(3))


def test_failed_projections_are_reported_and_can_be_enforced():
    C = np.array([[0, 1, 3], [1, 0, 2], [3, 2, 0]], dtype=float)
    kwargs = dict(max_iter=2, max_iter_proj=1, proj_tol_scale=1e-12, lr_const=10)
    with pytest.warns(ProjectionConvergenceWarning):
        _, info = gwipg(C, C, **kwargs)
    assert info["projection_criterion_failures"] > 0
    assert not info["all_projections_converged"]
    with pytest.raises(RuntimeError, match="residual test"):
        gwipg(C, C, **kwargs, projection_failure="raise")


def test_quiet_return_modes_and_timing(capsys):
    C = np.array([[0, 1], [1, 0]])
    T, info = gwipg(C, C, max_iter=1, track_timing=True, track_proj_iters=True)
    assert capsys.readouterr().out == ""
    assert info["n_iter"] == info["proj_calls"] == 1
    assert info["solver_internal_time"] > 0
    assert 0 <= info["projection_time_fraction"] <= 1
    np.testing.assert_allclose(gwipg(C, C, max_iter=1, log=False), T)


@pytest.mark.parametrize(
    "first,second,p,q,kwargs",
    [
        (np.zeros((2, 3)), np.eye(2), None, None, {}),
        (np.eye(2), np.eye(2), [1, -1], [0.5, 0.5], {}),
        (np.eye(2), np.eye(2), [0.5, 0.5], [1, 1], {}),
        (np.eye(2), np.eye(2), [0, 0], [0, 0], {}),
        (np.eye(2), np.eye(2), [[0.5, 0.5]], [0.5, 0.5], {}),
        (np.full((2, 2), np.nan), np.eye(2), None, None, {}),
        (np.eye(2), np.eye(2), None, None, {"max_iter": 0}),
        (np.eye(2), np.eye(2), None, None, {"max_iter_proj": 0}),
        (np.eye(2), np.eye(2), None, None, {"iter_check": 0}),
        (np.eye(2), np.eye(2), None, None, {"expon": float("inf")}),
        (np.eye(2), np.eye(2), None, None, {"projection_failure": "invalid"}),
        (np.eye(2), np.eye(2), None, None, {"G0": np.ones((3, 2))}),
        (np.eye(2), np.eye(2), None, None, {"G0": -np.ones((2, 2))}),
    ],
)
def test_invalid_inputs(first, second, p, q, kwargs):
    with pytest.raises(ValueError):
        gwipg(first, second, p, q, **kwargs)


def test_symmetric_override_rejects_asymmetric_cost():
    with pytest.raises(ValueError, match="symmetric"):
        gwipg([[0, 1], [2, 0]], np.eye(2), symmetric=True)

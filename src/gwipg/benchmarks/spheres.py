"""Benchmark GW solvers against exact distances between unit spheres.

Arya et al. give the exact (4, 2)-Gromov--Wasserstein distance between
Euclidean unit spheres with uniform measures.  Their distortion satisfies

    dis_{4,2}(Pi)^4 = integral (d_X^2 - d_Y^2)^2 dPi dPi.

Consequently, POT's square-loss GW objective matches this distortion to the
fourth power when its input cost matrices contain *squared*, unnormalized
Euclidean distances.  If ``loss`` is the resulting full GW objective, then

    estimated d_GW,4,2 = (loss / 16) ** 0.25.

The analytical target is for continuous uniform spheres.  Finite sampled
problems have empirical marginals, so the reported difference from the target
is a sampling-and-optimization estimation error, not a certified nonnegative
objective gap for the discrete problem.

Examples
--------
Quick end-to-end check:

    python -m gwipg.benchmarks.spheres --smoke-test

Recommended experiment:

    python -m gwipg.benchmarks.spheres \
        --pairs 1-2,1-3,2-3 \
        --sizes 50,100,200 \
        --seeds 0,1,2,3,4

Reference
---------
S. Arya et al., "The Gromov--Wasserstein Distance Between Spheres,"
Foundations of Computational Mathematics 26, 75--130 (2026).
https://doi.org/10.1007/s10208-024-09678-3
"""

from __future__ import annotations

import argparse
import sys
import time
import warnings
from pathlib import Path

import numpy as np
import pandas as pd
from scipy.spatial.distance import cdist
from scipy.special import gammaln

from ._common import (
    write_run_metadata,
    METHOD_ORDER,
    dataframe_to_latex,
    mean_std,
    parse_csv_values,
    solver_specs,
    unregularized_gw_loss,
)


def parse_sphere_pairs(text: str) -> list[tuple[int, int]]:
    """Parse comma-separated pairs such as ``1-2,1-3,2-3``."""
    pairs: list[tuple[int, int]] = []
    for item in text.split(","):
        item = item.strip()
        if not item:
            continue
        parts = item.split("-")
        if len(parts) != 2:
            raise argparse.ArgumentTypeError(f"invalid sphere pair {item!r}; expected M-N")
        try:
            first, second = (int(part) for part in parts)
        except ValueError as exc:
            raise argparse.ArgumentTypeError(
                f"invalid sphere pair {item!r}; dimensions must be integers"
            ) from exc
        if first < 0 or second < 0:
            raise argparse.ArgumentTypeError("sphere dimensions must be nonnegative")
        pairs.append(tuple(sorted((first, second))))
    if not pairs:
        raise argparse.ArgumentTypeError("expected at least one sphere pair")
    return list(dict.fromkeys(pairs))


def exact_sphere_distance(m: int, n: int) -> tuple[float, float]:
    """Return exact d_GW,4,2 and its corresponding full square-loss value."""
    if not 0 <= m <= n:
        raise ValueError(f"expected 0 <= m <= n, got {(m, n)}")

    log_ratio = (
        gammaln((m + 2.0) / 2.0)
        + gammaln((n + 1.0) / 2.0)
        - gammaln((m + 1.0) / 2.0)
        - gammaln((n + 2.0) / 2.0)
    )
    bracket = 1.0 / (m + 1.0) + 1.0 / (n + 1.0) - 2.0 * np.exp(2.0 * log_ratio) / (m + 1.0)
    # Protect the identical-sphere case from a tiny negative roundoff error.
    bracket = max(0.0, float(bracket))
    exact_distance = (bracket**0.25) / np.sqrt(2.0)
    exact_loss = 4.0 * bracket
    return float(exact_distance), float(exact_loss)


def sample_unit_sphere(
    sphere_dimension: int, n_samples: int, rng: np.random.Generator
) -> np.ndarray:
    """Draw uniform points from S^dimension by normalizing Gaussian vectors."""
    if sphere_dimension == 0:
        return rng.choice((-1.0, 1.0), size=(n_samples, 1))
    points = rng.standard_normal((n_samples, sphere_dimension + 1))
    norms = np.linalg.norm(points, axis=1, keepdims=True)
    return points / norms


def sphere_problem(
    m: int, n: int, n_samples: int, seed: int
) -> tuple[np.ndarray, np.ndarray, np.ndarray, np.ndarray]:
    """Generate an empirical pair of uniformly weighted unit spheres."""
    seed_sequence = np.random.SeedSequence([seed, m, n, n_samples])
    rng_x, rng_y = (np.random.default_rng(child) for child in seed_sequence.spawn(2))
    samples_x = sample_unit_sphere(m, n_samples, rng_x)
    samples_y = sample_unit_sphere(n, n_samples, rng_y)

    # Do not normalize these matrices: the analytical formula assumes unit
    # spheres and squared Euclidean distances on their original scale.
    cost_x = cdist(samples_x, samples_x, metric="sqeuclidean")
    cost_y = cdist(samples_y, samples_y, metric="sqeuclidean")
    np.fill_diagonal(cost_x, 0.0)
    np.fill_diagonal(cost_y, 0.0)
    marginal = np.full(n_samples, 1.0 / n_samples)
    return cost_x, cost_y, marginal, marginal.copy()


def optimization_metrics(
    coupling: np.ndarray,
    cost_x: np.ndarray,
    cost_y: np.ndarray,
    p: np.ndarray,
    q: np.ndarray,
    exact_distance: float,
    exact_loss: float,
) -> dict[str, float]:
    """Evaluate one coupling in the sphere paper's normalization."""
    loss = unregularized_gw_loss(coupling, cost_x, cost_y, p, q)
    estimated_distance = max(loss, 0.0) ** 0.25 / 2.0
    signed_distance_error = estimated_distance - exact_distance
    signed_loss_error = loss - exact_loss
    return {
        "loss": loss,
        "exact_loss": exact_loss,
        "loss_error": signed_loss_error,
        "loss_abs_error": abs(signed_loss_error),
        "estimated_distance": estimated_distance,
        "exact_distance": exact_distance,
        "distance_error": signed_distance_error,
        "distance_abs_error": abs(signed_distance_error),
        "distance_relative_error": (
            abs(signed_distance_error) / exact_distance if exact_distance > 0 else np.nan
        ),
        "sparsity": float(np.count_nonzero(coupling == 0.0) / coupling.size),
        "feasibility": float(
            np.linalg.norm(coupling.sum(axis=1) - p) + np.linalg.norm(coupling.sum(axis=0) - q)
        ),
    }


def run_benchmark(
    pairs: list[tuple[int, int]],
    sizes: list[int],
    seeds: list[int],
    epsilons: list[float],
    methods: list[str],
    max_iter: int,
    tol: float,
    ipg_tol: float,
    ipg_expon: float,
    ipg_lr_scale: float,
    ipg_max_iter_proj: int,
    ipg_proj_tol_scale: float,
    ipg_proj_lr_const: float,
    ipg_iter_check: int,
    ipg_track_proj_iters: bool,
    ipg_track_timing: bool,
) -> pd.DataFrame:
    """Run every requested solver and preserve failed configurations."""
    records: list[dict] = []
    diagnostic_names = [
        "proj_calls",
        "proj_iter_mean",
        "proj_iter_max",
        "proj_cap_hits",
        "proj_cap_hit_fraction",
        "outer_iterations",
        "gw_gradient_time",
        "projection_time",
        "other_time",
        "solver_internal_time",
        "gw_gradient_time_fraction",
        "projection_time_fraction",
    ]

    for m, n in pairs:
        exact_distance, exact_loss = exact_sphere_distance(m, n)
        for n_samples in sizes:
            for seed in seeds:
                print(f"\nPreparing S^{m} vs S^{n}, N={n_samples}, seed={seed}")
                cost_x, cost_y, p, q = sphere_problem(m, n, n_samples, seed)
                specs = solver_specs(
                    methods=methods,
                    epsilons=epsilons,
                    cost_x=cost_x,
                    cost_y=cost_y,
                    p=p,
                    q=q,
                    max_iter=max_iter,
                    tol=tol,
                    ipg_tol=ipg_tol,
                    ipg_expon=ipg_expon,
                    ipg_lr_const=ipg_lr_scale * n_samples,
                    ipg_max_iter_proj=ipg_max_iter_proj,
                    ipg_proj_tol_scale=ipg_proj_tol_scale,
                    ipg_proj_lr_const=ipg_proj_lr_const,
                    ipg_iter_check=ipg_iter_check,
                    ipg_track_proj_iters=ipg_track_proj_iters,
                    ipg_track_timing=ipg_track_timing,
                )

                for method, epsilon, solve in specs:
                    epsilon_label = "-" if np.isnan(epsilon) else f"{epsilon:.0e}"
                    print(
                        f"Running {method:>4s}, S^{m} vs S^{n}, "
                        f"N={n_samples}, seed={seed}, "
                        f"epsilon={epsilon_label}"
                    )
                    record = {
                        "sphere_m": m,
                        "sphere_n": n,
                        "n_samples": n_samples,
                        "seed": seed,
                        "method": method,
                        "epsilon": epsilon,
                        "status": "success",
                        "error": "",
                    }
                    try:
                        start = time.perf_counter()
                        result = solve()
                        runtime = time.perf_counter() - start
                        coupling = result[0] if isinstance(result, tuple) else result
                        solver_log = (
                            result[1]
                            if isinstance(result, tuple)
                            and len(result) > 1
                            and isinstance(result[1], dict)
                            else {}
                        )
                        coupling = np.asarray(coupling, dtype=float)
                        expected_shape = (n_samples, n_samples)
                        if coupling.shape != expected_shape:
                            raise ValueError(
                                f"solver returned shape {coupling.shape}, expected {expected_shape}"
                            )
                        if not np.isfinite(coupling).all():
                            raise FloatingPointError("solver returned a non-finite coupling")

                        metrics = optimization_metrics(
                            coupling,
                            cost_x,
                            cost_y,
                            p,
                            q,
                            exact_distance,
                            exact_loss,
                        )
                        record.update(metrics)
                        record.update(
                            {
                                "proj_calls": solver_log.get("proj_calls", np.nan),
                                "proj_iter_mean": solver_log.get("proj_iter_mean", np.nan),
                                "proj_iter_max": solver_log.get("proj_iter_max", np.nan),
                                "proj_cap_hits": solver_log.get("proj_cap_hits", np.nan),
                                "proj_cap_hit_fraction": solver_log.get(
                                    "proj_cap_hit_fraction", np.nan
                                ),
                                "outer_iterations": (
                                    solver_log.get("iter", np.nan) + 1
                                    if "iter" in solver_log
                                    else np.nan
                                ),
                                "gw_gradient_time": solver_log.get("gw_gradient_time", np.nan),
                                "projection_time": solver_log.get("projection_time", np.nan),
                                "other_time": solver_log.get("other_time", np.nan),
                                "solver_internal_time": solver_log.get(
                                    "solver_internal_time", np.nan
                                ),
                                "gw_gradient_time_fraction": solver_log.get(
                                    "gw_gradient_time_fraction", np.nan
                                ),
                                "projection_time_fraction": solver_log.get(
                                    "projection_time_fraction", np.nan
                                ),
                            }
                        )
                        record["time"] = runtime
                        print(
                            f"  d_hat={metrics['estimated_distance']:.6f}, "
                            f"d_exact={exact_distance:.6f}, "
                            f"|error|={metrics['distance_abs_error']:.3e}, "
                            f"loss={metrics['loss']:.6e}, "
                            f"feas={metrics['feasibility']:.2e}, "
                            f"time={runtime:.2f}s"
                        )
                    except Exception as exc:
                        empty_metrics = [
                            "loss",
                            "exact_loss",
                            "loss_error",
                            "loss_abs_error",
                            "estimated_distance",
                            "exact_distance",
                            "distance_error",
                            "distance_abs_error",
                            "distance_relative_error",
                            "sparsity",
                            "feasibility",
                            *diagnostic_names,
                            "time",
                        ]
                        record.update({name: np.nan for name in empty_metrics})
                        record.update(
                            {
                                "exact_loss": exact_loss,
                                "exact_distance": exact_distance,
                                "status": "fail",
                                "error": f"{type(exc).__name__}: {exc}",
                            }
                        )
                        print(f"  FAILED: {record['error']}", file=sys.stderr)
                    records.append(record)
    return pd.DataFrame.from_records(records)


def aggregate_results(raw: pd.DataFrame) -> pd.DataFrame:
    """Aggregate successful trials by sphere pair, size, method, and epsilon."""
    keys = ["sphere_m", "sphere_n", "n_samples", "method", "epsilon"]
    metrics = [
        "loss",
        "exact_loss",
        "loss_error",
        "loss_abs_error",
        "estimated_distance",
        "exact_distance",
        "distance_error",
        "distance_abs_error",
        "distance_relative_error",
        "sparsity",
        "feasibility",
        "proj_calls",
        "proj_iter_mean",
        "proj_iter_max",
        "proj_cap_hits",
        "proj_cap_hit_fraction",
        "outer_iterations",
        "gw_gradient_time",
        "projection_time",
        "other_time",
        "solver_internal_time",
        "gw_gradient_time_fraction",
        "projection_time_fraction",
        "time",
    ]
    successful = raw[raw["status"] == "success"]
    if successful.empty:
        summary = pd.DataFrame(columns=keys)
    else:
        summary = successful.groupby(keys, dropna=False)[metrics].agg(["mean", "std"]).reset_index()
        summary.columns = [
            "_".join(part for part in column if part) if isinstance(column, tuple) else column
            for column in summary.columns
        ]

    attempts = raw.groupby(keys, dropna=False).size().reset_index(name="n_attempts")
    successes = successful.groupby(keys, dropna=False).size().reset_index(name="n_success")
    summary = attempts.merge(successes, on=keys, how="left").merge(summary, on=keys, how="left")
    summary["n_success"] = summary["n_success"].fillna(0).astype(int)
    summary["n_fail"] = summary["n_attempts"] - summary["n_success"]
    summary["_method_order"] = summary["method"].map(METHOD_ORDER)
    return (
        summary.sort_values(
            [
                "sphere_m",
                "sphere_n",
                "n_samples",
                "_method_order",
                "epsilon",
            ],
            na_position="first",
        )
        .drop(columns="_method_order")
        .reset_index(drop=True)
    )


def paper_table(summary: pd.DataFrame) -> pd.DataFrame:
    """Create a compact Markdown/LaTeX-ready summary."""

    def six_decimals(mean: float, std: float = np.nan) -> str:
        if pd.isna(mean):
            return "-"
        if pd.isna(std):
            return f"{mean:.6f}"
        return f"{mean:.6f} +/- {std:.6f}"

    rows: list[dict] = []
    for _, result in summary.iterrows():
        rows.append(
            {
                "Spheres": (f"S^{int(result['sphere_m'])}-S^{int(result['sphere_n'])}"),
                "N": int(result["n_samples"]),
                "Method": result["method"],
                "epsilon": ("-" if pd.isna(result["epsilon"]) else f"{result['epsilon']:.0e}"),
                "d_hat": six_decimals(
                    result.get("estimated_distance_mean"),
                    result.get("estimated_distance_std"),
                ),
                "d_exact": six_decimals(result.get("exact_distance_mean")),
                "|d error|": mean_std(
                    result.get("distance_abs_error_mean"),
                    result.get("distance_abs_error_std"),
                    True,
                ),
                "Rel. error": mean_std(
                    result.get("distance_relative_error_mean"),
                    result.get("distance_relative_error_std"),
                    True,
                ),
                "Loss": mean_std(result.get("loss_mean"), result.get("loss_std"), True),
                "Feasibility": mean_std(
                    result.get("feasibility_mean"),
                    result.get("feasibility_std"),
                    True,
                ),
                "Sparsity": mean_std(
                    result.get("sparsity_mean"),
                    result.get("sparsity_std"),
                ),
                "Time (s)": mean_std(result.get("time_mean"), result.get("time_std")),
                "Success": (f"{int(result['n_success'])}/{int(result['n_attempts'])}"),
            }
        )
    return pd.DataFrame(rows)


def write_outputs(
    raw: pd.DataFrame,
    summary: pd.DataFrame,
    table: pd.DataFrame,
    output_dir: Path,
) -> None:
    """Write detailed, aggregate, and paper-style result files."""
    output_dir.mkdir(parents=True, exist_ok=True)
    raw.to_csv(output_dir / "gw_sphere_raw.csv", index=False)
    summary.to_csv(output_dir / "gw_sphere_summary.csv", index=False)
    table.to_csv(output_dir / "gw_sphere_table.csv", index=False)
    (output_dir / "gw_sphere_table.md").write_text(
        table.to_markdown(index=False) + "\n", encoding="utf-8"
    )
    (output_dir / "gw_sphere_table.tex").write_text(dataframe_to_latex(table), encoding="utf-8")


def build_parser() -> argparse.ArgumentParser:
    script_dir = Path.cwd()
    parser = argparse.ArgumentParser(
        description=("Compare GW solvers against exact continuous-sphere targets.")
    )
    parser.add_argument("--pairs", default="1-2,1-3,2-3")
    parser.add_argument("--sizes", default="50,100,200")
    parser.add_argument("--seeds", default="0,1,2,3,4")
    parser.add_argument("--epsilons", default="5e-4,1e-3,1e-2,1e-1,1e0")
    parser.add_argument("--methods", default="CG,PPA,EPGD,BAPG,iPG")
    parser.add_argument("--max-iter", type=int, default=5000)
    parser.add_argument("--tol", type=float, default=1e-9)
    parser.add_argument(
        "--ipg-tol",
        type=float,
        default=1e-4,
        help="iPG outer stopping tolerance.",
    )
    parser.add_argument("--ipg-expon", type=float, default=2.1)
    parser.add_argument("--ipg-lr-scale", type=float, default=10.0)
    parser.add_argument("--ipg-max-iter-proj", type=int, default=500)
    parser.add_argument("--ipg-proj-tol-scale", type=float, default=0.05)
    parser.add_argument("--ipg-proj-lr-const", type=float, default=1.5)
    parser.add_argument("--ipg-iter-check", type=int, default=5)
    parser.add_argument("--ipg-track-proj-iters", action="store_true")
    parser.add_argument("--ipg-track-timing", action="store_true")
    parser.add_argument(
        "--output-dir",
        type=Path,
        default=script_dir / "outputs" / "gw_sphere",
    )
    parser.add_argument(
        "--smoke-test",
        action="store_true",
        help="Use one pair, N=20, one seed, and two outer iterations.",
    )
    return parser


def main(argv=None) -> None:
    args = build_parser().parse_args(argv)
    if args.smoke_test:
        pairs = [(1, 2)]
        sizes = [20]
        seeds = [0]
        epsilons = [1e-1]
        methods = ["CG", "PPA", "EPGD", "BAPG", "iPG"]
        max_iter = 2
        ipg_max_iter_proj = 50
        output_dir = args.output_dir / "smoke_test"
    else:
        pairs = parse_sphere_pairs(args.pairs)
        sizes = parse_csv_values(args.sizes, int)
        seeds = parse_csv_values(args.seeds, int)
        epsilons = parse_csv_values(args.epsilons, float)
        methods = [method.strip() for method in args.methods.split(",") if method.strip()]
        max_iter = args.max_iter
        ipg_max_iter_proj = args.ipg_max_iter_proj
        output_dir = args.output_dir

    if min(sizes) < 2:
        raise ValueError("all sample sizes must be at least 2")

    with warnings.catch_warnings():
        warnings.simplefilter("default")
        raw = run_benchmark(
            pairs=pairs,
            sizes=sizes,
            seeds=seeds,
            epsilons=epsilons,
            methods=methods,
            max_iter=max_iter,
            tol=args.tol,
            ipg_tol=args.ipg_tol,
            ipg_expon=args.ipg_expon,
            ipg_lr_scale=args.ipg_lr_scale,
            ipg_max_iter_proj=ipg_max_iter_proj,
            ipg_proj_tol_scale=args.ipg_proj_tol_scale,
            ipg_proj_lr_const=args.ipg_proj_lr_const,
            ipg_iter_check=args.ipg_iter_check,
            ipg_track_proj_iters=args.ipg_track_proj_iters,
            ipg_track_timing=args.ipg_track_timing,
        )

    write_run_metadata(args, output_dir)
    summary = aggregate_results(raw)
    table = paper_table(summary)
    write_outputs(raw, summary, table, output_dir)

    print("\nFinal paper-style table\n")
    print(table.to_string(index=False))
    print(f"\nOutputs written to {output_dir.resolve()}")
    failures = raw[raw["status"] == "fail"]
    if not failures.empty:
        print("\nFailed configurations:")
        print(
            failures[
                [
                    "sphere_m",
                    "sphere_n",
                    "n_samples",
                    "seed",
                    "method",
                    "epsilon",
                    "error",
                ]
            ].to_string(index=False)
        )


if __name__ == "__main__":
    main()

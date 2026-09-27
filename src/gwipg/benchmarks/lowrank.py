#!/usr/bin/env python3
"""Benchmark dense iPG/EPGD and POT low-rank GW on the POT curve example.

The data generation follows:
https://pythonot.github.io/auto_examples/others/plot_lowrank_GW.html

Dense methods are guarded by a conservative memory preflight because an
N-by-N float64 array alone requires 8*N**2 bytes. Use --force-dense only when
the host has enough available memory for several such arrays.
"""

from __future__ import annotations

import argparse
import csv
import gc
import json
import math
import sys

try:
    import resource
except ImportError:
    resource = None
import time
from pathlib import Path

import numpy as np
import ot
import psutil
from scipy.spatial.distance import cdist

from gwipg import gwipg
from ._common import write_run_metadata


def make_curves(n: int) -> tuple[np.ndarray, np.ndarray]:
    """Generate the 2D and 3D curves used in POT's low-rank GW example."""
    theta = np.linspace(-4 * np.pi, 4 * np.pi, n)
    z = np.linspace(1, 2, n)
    radius = z**2 + 1
    x = radius * np.sin(theta)
    y = radius * np.cos(theta)
    source = np.column_stack((x, z))
    target = np.column_stack((x, y, z))
    return source, target


def max_sqeuclidean(points: np.ndarray, block: int = 1000) -> float:
    """Compute the exact maximum squared distance without storing N-by-N costs."""
    maximum = 0.0
    for start in range(0, len(points), block):
        distances = cdist(points[start : start + block], points, "sqeuclidean")
        maximum = max(maximum, float(distances.max()))
    return maximum


def factorized_cost(points: np.ndarray, scale: float) -> tuple[np.ndarray, np.ndarray]:
    left, right = ot.lowrank.compute_lr_sqeuclidean_matrix(points, points, rescale_cost=False)
    root_scale = math.sqrt(scale)
    return left / root_scale, right / root_scale


def dense_cost(points: np.ndarray, scale: float) -> np.ndarray:
    return ot.dist(points, points, metric="sqeuclidean") / scale


def feasibility_dense(plan: np.ndarray, p: np.ndarray, q: np.ndarray) -> float:
    return float(np.linalg.norm(plan.sum(axis=1) - p) + np.linalg.norm(plan.sum(axis=0) - q))


def feasibility_lowrank(
    q_factor: np.ndarray,
    r_factor: np.ndarray,
    weights: np.ndarray,
    p: np.ndarray,
    q: np.ndarray,
) -> float:
    """Marginal residual of Q diag(1/g) R.T without forming the plan."""
    row = q_factor @ (r_factor.sum(axis=0) / weights)
    col = r_factor @ (q_factor.sum(axis=0) / weights)
    return float(np.linalg.norm(row - p) + np.linalg.norm(col - q))


def peak_rss_gb() -> float:
    if resource is None:
        return psutil.Process().memory_info().rss / 1024**3
    value = resource.getrusage(resource.RUSAGE_SELF).ru_maxrss
    return value / (1024**3 if sys.platform == "darwin" else 1024**2)


def dense_memory_estimate_gb(n: int, method: str) -> float:
    """Conservative working-set estimate for the current float64 implementations."""
    one = 8 * n * n / 1e9
    multiplier = 10 if method == "iPG" else 8
    return multiplier * one


def row(
    n: int,
    method: str,
    epsilon: float | None,
    rank: int | None,
    status: str,
    runtime: float | None = None,
    loss: float | None = None,
    feasibility: float | None = None,
    sparsity: float | None = None,
    peak_memory: float | None = None,
    note: str = "",
) -> dict[str, object]:
    return {
        "n": n,
        "method": method,
        "epsilon": epsilon,
        "rank": rank,
        "status": status,
        "runtime_s": runtime,
        "gw_loss": loss,
        "feasibility": feasibility,
        "sparsity": sparsity,
        "peak_rss_gb": peak_memory,
        "note": note,
    }


def run_benchmark(args: argparse.Namespace) -> list[dict[str, object]]:
    source, target = make_curves(args.n)
    p = ot.unif(args.n)
    q = ot.unif(args.n)

    scale_source = max_sqeuclidean(source, args.distance_block)
    scale_target = max_sqeuclidean(target, args.distance_block)
    factors_source = factorized_cost(source, scale_source)
    factors_target = factorized_cost(target, scale_target)

    results: list[dict[str, object]] = []
    for epsilon in args.epsilons:
        for rank in args.ranks:
            start = time.perf_counter()
            q_factor, r_factor, weights, log = ot.lowrank_gromov_wasserstein_samples(
                source,
                target,
                a=p,
                b=q,
                reg=epsilon,
                rank=rank,
                rescale_cost=False,
                cost_factorized_Xs=factors_source,
                cost_factorized_Xt=factors_target,
                seed_init=args.seed,
                numItermax=args.max_iter,
                stopThr=args.tol,
                log=True,
            )
            runtime = time.perf_counter() - start
            results.append(
                row(
                    args.n,
                    "LowRankGW",
                    epsilon,
                    rank,
                    "ok",
                    runtime,
                    float(log["value_quad"]),
                    feasibility_lowrank(q_factor, r_factor, weights, p, q),
                    None,
                    peak_rss_gb(),
                    "Coupling stored as Q diag(1/g) R^T; not materialized.",
                )
            )
            print(json.dumps(results[-1]), flush=True)
            del q_factor, r_factor, weights, log
            gc.collect()

    available_gb = psutil.virtual_memory().available / 1e9
    dense_requests = [("EPGD", epsilon) for epsilon in args.epsilons]
    dense_requests.append(("iPG", None))

    cost_source = cost_target = None
    for method, epsilon in dense_requests:
        estimate = dense_memory_estimate_gb(args.n, method)
        if not args.force_dense and estimate > args.memory_fraction * available_gb:
            results.append(
                row(
                    args.n,
                    method,
                    epsilon,
                    None,
                    "skipped-memory",
                    note=(
                        f"Estimated peak {estimate:.1f} GB exceeds "
                        f"{args.memory_fraction:.0%} of initial available "
                        f"memory ({available_gb:.1f} GB)."
                    ),
                )
            )
            print(json.dumps(results[-1]), flush=True)
            continue

        if cost_source is None:
            cost_source = dense_cost(source, scale_source)
            cost_target = dense_cost(target, scale_target)

        start = time.perf_counter()
        if method == "EPGD":
            plan, log = ot.gromov.entropic_gromov_wasserstein(
                cost_source,
                cost_target,
                p,
                q,
                loss_fun="square_loss",
                epsilon=epsilon,
                solver="PGD",
                max_iter=args.max_iter,
                tol=args.tol,
                log=True,
                verbose=False,
            )
            loss = float(log["gw_dist"])
        else:
            plan, log = gwipg(
                cost_source,
                cost_target,
                p,
                q,
                max_iter=args.max_iter,
                max_iter_proj=args.ipg_max_iter_proj,
                tol=args.tol,
                iter_check=args.ipg_iter_check,
                verbose=False,
                log=True,
                expon=args.ipg_expon,
                lr_const=args.ipg_lr_const,
                proj_tol_scale=args.ipg_proj_tol_scale,
                proj_lr_const=args.ipg_proj_lr_const,
            )
            loss = float(log["gw_dist"])
        runtime = time.perf_counter() - start
        results.append(
            row(
                args.n,
                method,
                epsilon,
                None,
                "ok",
                runtime,
                loss,
                feasibility_dense(plan, p, q),
                float(np.mean(plan == 0)),
                peak_rss_gb(),
            )
        )
        print(json.dumps(results[-1]), flush=True)
        del plan, log
        gc.collect()

    return results


def write_results(results: list[dict[str, object]], output_dir: Path) -> None:
    output_dir.mkdir(parents=True, exist_ok=True)
    fields = list(results[0])
    with (output_dir / "lowrank_gw_raw.csv").open("w", newline="") as handle:
        writer = csv.DictWriter(handle, fieldnames=fields)
        writer.writeheader()
        writer.writerows(results)

    lines = [
        "| n | Method | epsilon | Rank | Status | GW loss | Feasibility | "
        "Sparsity | Time (s) | Peak RSS (GB) |",
        "|---:|---|---:|---:|---|---:|---:|---:|---:|---:|",
    ]
    for result in results:

        def fmt(key: str, spec: str = ".3g") -> str:
            value = result[key]
            return "--" if value is None else format(float(value), spec)

        lines.append(
            f"| {result['n']} | {result['method']} | "
            f"{fmt('epsilon')} | {fmt('rank', '.0f')} | {result['status']} | "
            f"{fmt('gw_loss', '.6g')} | {fmt('feasibility', '.3e')} | "
            f"{fmt('sparsity', '.2%')} | {fmt('runtime_s', '.2f')} | "
            f"{fmt('peak_rss_gb', '.2f')} |"
        )
    (output_dir / "lowrank_gw_table.md").write_text("\n".join(lines) + "\n")


def parse_args(argv=None) -> argparse.Namespace:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--n", type=int, default=10_000)
    parser.add_argument("--epsilons", type=float, nargs="+", default=[0.1, 1.0])
    parser.add_argument("--ranks", type=int, nargs="+", default=[10, 50])
    parser.add_argument("--tol", type=float, default=2.5e-4)
    parser.add_argument("--max-iter", type=int, default=1000)
    parser.add_argument("--seed", type=int, default=49)
    parser.add_argument("--distance-block", type=int, default=1000)
    parser.add_argument("--output-dir", type=Path, default=Path("outputs/lowrank_gw"))
    parser.add_argument("--force-dense", action="store_true")
    parser.add_argument("--memory-fraction", type=float, default=0.8)
    parser.add_argument("--ipg-max-iter-proj", type=int, default=500)
    parser.add_argument("--ipg-iter-check", type=int, default=5)
    parser.add_argument("--ipg-expon", type=float, default=2.1)
    parser.add_argument("--ipg-lr-const", type=float, default=10.0)
    parser.add_argument("--ipg-proj-tol-scale", type=float, default=0.05)
    parser.add_argument("--ipg-proj-lr-const", type=float, default=1.5)
    return parser.parse_args(argv)


def main(argv=None):
    parsed = parse_args(argv)
    write_run_metadata(parsed, parsed.output_dir)
    benchmark_results = run_benchmark(parsed)
    write_results(benchmark_results, parsed.output_dir)


if __name__ == "__main__":
    main()

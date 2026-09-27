"""Reproduce the paper's iPG graph-alignment runs with inner-cost diagnostics.

The instance generation, seeds, and iPG hyperparameters match
``test_graph_alignment.py``.  In addition to the paper's metrics, this script
records accelerated-gradient iteration counts and a runtime breakdown between
GW-gradient evaluations and projection solves.
"""

from __future__ import annotations

import argparse
import time
from pathlib import Path

import numpy as np
import pandas as pd

from gwipg import gwipg
from ._common import write_run_metadata
from .graph_alignment import (
    generate_gw_graph_instance_noisy,
    matching_accuracy,
)


def parse_args(argv=None) -> argparse.Namespace:
    parser = argparse.ArgumentParser()
    parser.add_argument(
        "--sizes",
        type=int,
        nargs="+",
        default=[100, 200, 300, 400, 500],
    )
    parser.add_argument("--n-seeds", type=int, default=20)
    parser.add_argument("--edge-prob", type=float, default=0.2)
    parser.add_argument("--noise-level", type=float, default=0.1)
    parser.add_argument("--max-iter", type=int, default=5000)
    parser.add_argument("--max-iter-proj", type=int, default=5000)
    parser.add_argument("--tol", type=float, default=1e-9)
    parser.add_argument("--expon", type=float, default=3.0)
    parser.add_argument("--output-dir", type=Path, required=True)
    return parser.parse_args(argv)


def summarize(raw: pd.DataFrame) -> pd.DataFrame:
    metrics = [
        "loss",
        "feasibility",
        "sparsity",
        "accuracy",
        "total_time",
        "solver_internal_time",
        "outer_iterations",
        "projection_calls",
        "mean_inner_iterations",
        "max_inner_iterations",
        "total_inner_iterations",
        "projection_cap_hit_fraction",
        "gw_gradient_time",
        "projection_time",
        "other_time",
        "gw_gradient_time_fraction",
        "projection_time_fraction",
    ]
    aggregations = {}
    for metric in metrics:
        aggregations[f"{metric}_mean"] = (metric, "mean")
        aggregations[f"{metric}_std"] = (metric, "std")
    aggregations["n_success"] = ("seed", "count")
    return raw.groupby("n_samples", as_index=False).agg(**aggregations)


def markdown_table(summary: pd.DataFrame) -> str:
    header = (
        "| n | Outer iters | Inner iters/call | Total inner iters | "
        "Cap-hit rate | Total time (s) | GW-gradient time | Projection time | Other time |\n"
        "|---:|---:|---:|---:|---:|---:|---:|---:|---:|"
    )
    rows = [header]
    for _, row in summary.iterrows():
        rows.append(
            "| {n:d} | {outer:.1f}±{outer_sd:.1f} | "
            "{inner:.1f}±{inner_sd:.1f} | {total_inner:.0f}±{total_inner_sd:.0f} | "
            "{cap:.2%} | {runtime:.2f}±{runtime_sd:.2f} | "
            "{grad:.2f}s ({grad_frac:.1%}) | {proj:.2f}s ({proj_frac:.1%}) | "
            "{other:.2f}s |".format(
                n=int(row["n_samples"]),
                outer=row["outer_iterations_mean"],
                outer_sd=row["outer_iterations_std"],
                inner=row["mean_inner_iterations_mean"],
                inner_sd=row["mean_inner_iterations_std"],
                total_inner=row["total_inner_iterations_mean"],
                total_inner_sd=row["total_inner_iterations_std"],
                cap=row["projection_cap_hit_fraction_mean"],
                runtime=row["total_time_mean"],
                runtime_sd=row["total_time_std"],
                grad=row["gw_gradient_time_mean"],
                grad_frac=(row["gw_gradient_time_mean"] / row["solver_internal_time_mean"]),
                proj=row["projection_time_mean"],
                proj_frac=(row["projection_time_mean"] / row["solver_internal_time_mean"]),
                other=row["other_time_mean"],
            )
        )
    return "\n".join(rows) + "\n"


def main(argv=None) -> None:
    args = parse_args(argv)
    write_run_metadata(args, args.output_dir)

    seed_rng = np.random.default_rng(0)
    seeds = seed_rng.integers(0, 10**9, size=args.n_seeds).tolist()
    records: list[dict[str, float | int]] = []

    for n_samples in args.sizes:
        for seed_index, seed in enumerate(seeds, start=1):
            _, _, C1, C2, p, q, perm = generate_gw_graph_instance_noisy(
                n=n_samples,
                edge_prob=args.edge_prob,
                noise_level=args.noise_level,
                seed=seed,
            )

            start = time.perf_counter()
            coupling, log = gwipg(
                C1,
                C2,
                p,
                q,
                max_iter=args.max_iter,
                max_iter_proj=args.max_iter_proj,
                tol=args.tol,
                verbose=False,
                log=True,
                expon=args.expon,
                lr_const=10.0 * n_samples,
                track_proj_iters=True,
                track_timing=True,
            )
            total_time = time.perf_counter() - start
            proj_iters = np.asarray(log["proj_iters"], dtype=int)

            record = {
                "n_samples": n_samples,
                "seed_index": seed_index,
                "seed": seed,
                "loss": log["gw_dist"],
                "feasibility": (
                    np.linalg.norm(coupling.sum(axis=1) - p)
                    + np.linalg.norm(coupling.sum(axis=0) - q)
                ),
                "sparsity": np.mean(coupling == 0.0),
                "accuracy": matching_accuracy(coupling, perm),
                "total_time": total_time,
                "solver_internal_time": log["solver_internal_time"],
                "outer_iterations": log["iter"] + 1,
                "projection_calls": log["proj_calls"],
                "mean_inner_iterations": log["proj_iter_mean"],
                "max_inner_iterations": log["proj_iter_max"],
                "total_inner_iterations": int(proj_iters.sum()),
                "projection_cap_hits": log["proj_cap_hits"],
                "projection_cap_hit_fraction": log["proj_cap_hit_fraction"],
                "gw_gradient_time": log["gw_gradient_time"],
                "projection_time": log["projection_time"],
                "other_time": log["other_time"],
                "gw_gradient_time_fraction": log["gw_gradient_time_fraction"],
                "projection_time_fraction": log["projection_time_fraction"],
                "final_outer_change": log["final_err"],
            }
            records.append(record)

            raw = pd.DataFrame(records)
            raw.to_csv(args.output_dir / "graph_alignment_ipg_raw.csv", index=False)
            print(
                f"n={n_samples:3d} seed={seed_index:02d}/{args.n_seeds} "
                f"outer={record['outer_iterations']:4d} "
                f"inner/call={record['mean_inner_iterations']:7.1f} "
                f"time={total_time:7.2f}s "
                f"grad={record['gw_gradient_time_fraction']:6.1%} "
                f"proj={record['projection_time_fraction']:6.1%}",
                flush=True,
            )

    raw = pd.DataFrame(records)
    summary = summarize(raw)
    raw.to_csv(args.output_dir / "graph_alignment_ipg_raw.csv", index=False)
    summary.to_csv(
        args.output_dir / "graph_alignment_ipg_summary.csv",
        index=False,
    )
    (args.output_dir / "graph_alignment_ipg_table.md").write_text(
        markdown_table(summary),
        encoding="utf-8",
    )
    print("\n" + markdown_table(summary), flush=True)


if __name__ == "__main__":
    main()

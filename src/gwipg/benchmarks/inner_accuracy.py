"""Grid-search practical iPG parameters on the paired SNARE-seq problem."""

from __future__ import annotations

import argparse
import itertools
import time
from pathlib import Path

import numpy as np
import pandas as pd

from gwipg import gwipg
from ._common import parse_csv_values, write_run_metadata
from .snare_seq import (
    coupling_metrics,
    load_snare_data,
    paired_subset,
    scot_cost_matrix,
)


def build_parser() -> argparse.ArgumentParser:
    script_dir = Path.cwd()
    parser = argparse.ArgumentParser(description="Tune iPG on a fixed paired SNARE-seq subset.")
    parser.add_argument("--offline", action="store_true")
    parser.add_argument("--size", type=int, default=250)
    parser.add_argument("--seed", type=int, default=0)
    parser.add_argument("--neighbors", type=int, default=50)
    parser.add_argument("--outer-tols", default="1e-6")
    parser.add_argument("--exponents", default="2.1")
    parser.add_argument("--proj-max-iters", default="1000")
    parser.add_argument("--proj-tol-scales", default="1.0")
    parser.add_argument("--proj-lr-consts", default="0.5,1.0,1.5,2.0")
    parser.add_argument("--iter-checks", default="10")
    parser.add_argument("--outer-lr-scales", default="10.0")
    parser.add_argument("--max-iter", type=int, default=5000)
    parser.add_argument("--label-neighbors", type=int, default=5)
    parser.add_argument(
        "--data-dir",
        type=Path,
        default=script_dir / "data" / "snare_seq",
    )
    parser.add_argument(
        "--output",
        type=Path,
        default=script_dir / "outputs" / "snare_seq" / "ipg_tuning" / "tuning_raw.csv",
    )
    return parser


def main(argv=None) -> None:
    args = build_parser().parse_args(argv)
    write_run_metadata(args, args.output.parent)
    outer_tols = parse_csv_values(args.outer_tols, float)
    exponents = parse_csv_values(args.exponents, float)
    proj_max_iters = parse_csv_values(args.proj_max_iters, int)
    proj_tol_scales = parse_csv_values(args.proj_tol_scales, float)
    proj_lr_consts = parse_csv_values(args.proj_lr_consts, float)
    iter_checks = parse_csv_values(args.iter_checks, int)
    outer_lr_scales = parse_csv_values(args.outer_lr_scales, float)

    chromatin, rna, cell_types = load_snare_data(args.data_dir, offline=args.offline)
    (
        chromatin_subset,
        rna_subset,
        chromatin_types,
        rna_types,
        true_col_for_row,
    ) = paired_subset(
        chromatin,
        rna,
        cell_types,
        n_samples=args.size,
        seed=args.seed,
    )
    cost_x = scot_cost_matrix(chromatin_subset, args.neighbors)
    cost_y = scot_cost_matrix(rna_subset, args.neighbors)
    p = np.full(args.size, 1.0 / args.size)
    q = p.copy()

    configurations = list(
        itertools.product(
            outer_tols,
            exponents,
            proj_max_iters,
            proj_tol_scales,
            proj_lr_consts,
            iter_checks,
            outer_lr_scales,
        )
    )
    print(f"Running {len(configurations)} iPG configurations")
    records: list[dict] = []
    args.output.parent.mkdir(parents=True, exist_ok=True)

    for index, configuration in enumerate(configurations, start=1):
        (
            outer_tol,
            exponent,
            proj_max_iter,
            proj_tol_scale,
            proj_lr_const,
            iter_check,
            outer_lr_scale,
        ) = configuration
        record = {
            "n_samples": args.size,
            "seed": args.seed,
            "outer_tol": outer_tol,
            "exponent": exponent,
            "proj_max_iter": proj_max_iter,
            "proj_tol_scale": proj_tol_scale,
            "proj_lr_const": proj_lr_const,
            "iter_check": iter_check,
            "outer_lr_scale": outer_lr_scale,
            "status": "success",
            "error": "",
        }
        print(
            f"[{index}/{len(configurations)}] tol={outer_tol:.0e}, "
            f"alpha={exponent:g}, proj_max={proj_max_iter}, "
            f"proj_scale={proj_tol_scale:g}, proj_lr={proj_lr_const:g}, "
            f"check={iter_check}, outer_lr_scale={outer_lr_scale:g}"
        )
        try:
            start = time.perf_counter()
            cpu_start = time.process_time()
            coupling, history = gwipg(
                cost_x,
                cost_y,
                p,
                q,
                max_iter=args.max_iter,
                max_iter_proj=proj_max_iter,
                tol=outer_tol,
                iter_check=iter_check,
                verbose=False,
                log=True,
                expon=exponent,
                lr_const=outer_lr_scale * args.size,
                proj_tol_scale=proj_tol_scale,
                proj_lr_const=proj_lr_const,
                track_proj_iters=True,
            )
            runtime = time.perf_counter() - start
            cpu_time = time.process_time() - cpu_start
            if not np.isfinite(coupling).all():
                raise FloatingPointError("non-finite coupling")
            metrics = coupling_metrics(
                coupling,
                cost_x,
                cost_y,
                p,
                q,
                chromatin_subset,
                rna_subset,
                true_col_for_row,
                chromatin_types,
                rna_types,
                args.label_neighbors,
            )
            record.update(metrics)
            record.update(
                {
                    "time": runtime,
                    "cpu_time": cpu_time,
                    "outer_iter": int(history["iter"]),
                    "outer_final_err": float(history["final_err"]),
                    "proj_calls": int(history["proj_calls"]),
                    "proj_iter_mean": float(history["proj_iter_mean"]),
                    "proj_iter_max": int(history["proj_iter_max"]),
                    "proj_cap_hits": int(history["proj_cap_hits"]),
                    "proj_cap_hit_fraction": float(history["proj_cap_hit_fraction"]),
                }
            )
            print(
                f"  time={runtime:.2f}s, cpu={cpu_time:.2f}s, "
                f"iter={history['iter']}, "
                f"loss={metrics['loss']:.8f}, "
                f"feas={metrics['feasibility']:.2e}, "
                f"FOSCTTM={metrics['foscttm']:.4f}, "
                f"LTA={metrics['lta_x_to_y']:.3f}, "
                f"proj_mean={history['proj_iter_mean']:.1f}, "
                f"cap_rate={history['proj_cap_hit_fraction']:.3f}"
            )
        except Exception as exc:
            record.update(
                {
                    "status": "fail",
                    "error": f"{type(exc).__name__}: {exc}",
                }
            )
            print(f"  FAILED: {record['error']}")

        records.append(record)
        pd.DataFrame.from_records(records).to_csv(args.output, index=False)

    results = pd.DataFrame.from_records(records)
    successful = results[results["status"] == "success"].copy()
    if not successful.empty:
        successful = successful.sort_values(
            ["time", "loss", "feasibility"], ascending=[True, True, True]
        )
        print("\nSuccessful configurations sorted by runtime:")
        columns = [
            "outer_tol",
            "exponent",
            "proj_max_iter",
            "proj_tol_scale",
            "proj_lr_const",
            "iter_check",
            "outer_lr_scale",
            "outer_iter",
            "proj_iter_mean",
            "proj_iter_max",
            "proj_cap_hit_fraction",
            "loss",
            "feasibility",
            "foscttm",
            "lta_x_to_y",
            "sparsity",
            "time",
            "cpu_time",
        ]
        print(successful[columns].to_string(index=False))
    print(f"\nResults written to {args.output.resolve()}")


if __name__ == "__main__":
    main()

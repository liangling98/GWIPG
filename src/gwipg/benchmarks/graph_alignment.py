"""Synthetic graph alignment benchmark with the paper's seeded protocol."""

from __future__ import annotations

import argparse
from pathlib import Path
from ._common import write_run_metadata

from gwipg import gwipg
import numpy as np
import ot
import networkx as nx
from scipy.sparse.csgraph import shortest_path
from scipy.optimize import linear_sum_assignment
import time
import pandas as pd


def connected_er_graph(n, edge_prob, seed=None):
    if n < 2 or not 0 < edge_prob <= 1:
        raise ValueError("n >= 2 and 0 < edge_prob <= 1 are required")
    rng = np.random.default_rng(seed)
    for _ in range(10000):
        s = int(rng.integers(1_000_000_000))
        G = nx.erdos_renyi_graph(n, edge_prob, seed=s)
        if nx.is_connected(G):
            return G
    raise RuntimeError("could not sample a connected graph; increase edge_prob")


def graph_to_cost_matrix(G, normalize=True):
    A = nx.to_numpy_array(G, dtype=float)
    C = shortest_path(A, directed=False, unweighted=True)
    if normalize and C.max() > 0:
        C = C / C.max()
    return C


def add_edge_flip_noise(A, noise_level, seed=None):
    """
    Flip each off-diagonal edge independently with probability noise_level.
    Keeps symmetry and zero diagonal.
    """
    rng = np.random.default_rng(seed)
    n = A.shape[0]

    if not 0 <= noise_level <= 1:
        raise ValueError("noise_level must be in [0, 1]")
    noisy_A = A.copy()
    rows, cols = np.triu_indices(n, k=1)
    flip = rng.random(len(rows)) < noise_level
    rows, cols = rows[flip], cols[flip]
    noisy_A[rows, cols] = 1.0 - noisy_A[rows, cols]
    noisy_A[cols, rows] = noisy_A[rows, cols]

    np.fill_diagonal(noisy_A, 0.0)
    return noisy_A


def generate_gw_graph_instance_noisy(
    n=30, edge_prob=0.25, noise_level=0.05, seed=0, normalize_cost=True
):
    rng = np.random.default_rng(seed)

    # Graph 1
    G1 = connected_er_graph(n, edge_prob, seed=seed)
    A1 = nx.to_numpy_array(G1, dtype=float)

    # Clean permuted Graph 2
    perm = rng.permutation(n)
    A2_clean = A1[perm][:, perm]

    # Add noise until connected
    for _ in range(10000):
        A2 = add_edge_flip_noise(
            A2_clean, noise_level=noise_level, seed=int(rng.integers(1_000_000_000))
        )
        G2 = nx.from_numpy_array(A2)
        if nx.is_connected(G2):
            break
    else:
        raise RuntimeError("could not generate connected noisy graph")

    # Cost matrices for POT
    C1 = graph_to_cost_matrix(G1, normalize=normalize_cost)
    C2 = graph_to_cost_matrix(G2, normalize=normalize_cost)

    # Uniform masses
    p = np.ones(n) / n
    q = np.ones(n) / n

    return G1, G2, C1, C2, p, q, perm


def perm_to_ground_truth_matrix(perm):
    n = len(perm)
    P = np.zeros((n, n))
    for j, i in enumerate(perm):
        P[i, j] = 1
    return P


def coupling_to_perm_matrix(T, method="hungarian"):
    """
    Convert a coupling matrix T into a hard matching matrix P.

    Parameters
    ----------
    T : ndarray of shape (n, m)
        Coupling / score matrix. Larger T[i, j] means node i prefers node j.
    method : {"hungarian", "argmax"}
        - "hungarian": one-to-one matching via linear_sum_assignment
        - "argmax": row-wise argmax, not necessarily one-to-one

    Returns
    -------
    P : ndarray of shape (n, m)
        Hard matching matrix with entries in {0,1}.
    """
    T = np.asarray(T)
    n, m = T.shape
    P = np.zeros_like(T, dtype=float)

    if method == "hungarian":
        row_ind, col_ind = linear_sum_assignment(-T)  # maximize total mass
        P[row_ind, col_ind] = 1.0

    elif method == "argmax":
        col_ind = np.argmax(T, axis=1)
        P[np.arange(n), col_ind] = 1.0

    else:
        raise ValueError("method must be either 'hungarian' or 'argmax'")

    return P


def matching_accuracy(T, perm):
    P_true = perm_to_ground_truth_matrix(perm)
    P_pred = coupling_to_perm_matrix(T)
    return (P_true * P_pred).sum() / len(perm)


def append_result(
    records,
    n_samples,
    seed,
    method,
    epsilon,
    loss,
    sparsity,
    feas,
    acc,
    runtime,
    status="success",
    error_msg=None,
):
    records.append(
        {
            "n_samples": n_samples,
            "seed": seed,
            "method": method,
            "epsilon": epsilon,
            "loss": loss,
            "sparsity": sparsity,
            "feas": feas,
            "acc": acc,
            "time": runtime,
            "status": status,
            "error": error_msg,
        }
    )


def pm(mean, std, fmt=".2f"):
    if pd.isna(mean):
        return "-"
    if pd.isna(std):
        return f"{mean:{fmt}}"
    return f"{mean:{fmt}} ± {std:{fmt}}"


def run_solver(records, n_samples, seed, method, epsilon, solver_fn, perm, p, q):
    try:
        start = time.perf_counter()
        gw, log = solver_fn()
        if not np.isfinite(gw).all():
            raise FloatingPointError("solver returned a non-finite coupling")
        runtime = time.perf_counter() - start

        sparsity = (gw == 0.0).sum() / gw.size
        feas = np.linalg.norm(gw.sum(axis=1) - p) + np.linalg.norm(gw.sum(axis=0) - q)
        acc = matching_accuracy(gw, perm)
        loss = log["gw_dist"] if isinstance(log, dict) and "gw_dist" in log else np.nan

        append_result(
            records,
            n_samples,
            seed,
            method,
            epsilon,
            loss,
            sparsity,
            feas,
            acc,
            runtime,
            status="success",
        )

        eps_str = "-" if pd.isna(epsilon) else f"{epsilon:.1e}"
        print(
            f"{method:>4s} (n={n_samples}, eps={eps_str}), "
            f"loss={loss:5.4e}, sp={sparsity:.2f}, feas={feas:.1e}, acc={acc:.2f}, t={runtime:.2f}"
        )

    except Exception as e:
        eps_str = "-" if pd.isna(epsilon) else f"{epsilon:.1e}"
        print(f"{method} failed at n={n_samples}, epsilon={eps_str}: {e}")
        append_result(
            records,
            n_samples,
            seed,
            method,
            epsilon,
            np.nan,
            np.nan,
            np.nan,
            np.nan,
            np.nan,
            status="fail",
            error_msg=str(e),
        )


def run_gw_experiments(
    n_samples_list,
    seeds,
    epsilons,
    edge_prob=0.1,
    noise_level=3.5e-2,
    max_iter=10,
    tol=1e-9,
    run_methods=None,
    ipg_expon=2.15,
    ipg_lr_const_fn=None,
    ipg_max_iter_proj=5000,
    ipg_iter_check=100,
):
    """
    Parameters
    ----------
    n_samples_list : list[int]
        List of problem sizes to test, e.g. [100, 200, 500, 1000]
    seeds : list[int]
    epsilons : list[float]
        Used for entropic methods only.
    run_methods : dict or None
        Example:
        {
            "CG": True,
            "PPA": True,
            "PGD": False,
            "BAPG": True,
            "iPG": True,
        }
    ipg_lr_const_fn : callable or None
        Function of n_samples returning lr_const for iPG.
        Default: lambda n: 20.0 * n
    """

    if run_methods is None:
        run_methods = {
            "CG": True,
            "PPA": True,
            "PGD": True,
            "BAPG": True,
            "iPG": True,
        }

    if ipg_lr_const_fn is None:

        def ipg_lr_const_fn(n):
            return 20.0 * n

    records = []

    for n_samples in n_samples_list:
        print(f"\n{'#' * 80}")
        print(f"Starting experiments for n_samples = {n_samples}")
        print(f"{'#' * 80}")

        for seed in seeds:
            print(f"\n{'=' * 70}")
            print(f"Testing n_samples = {n_samples}, seed = {seed} ...")

            # ------------------------------------------------------------
            # Instance generation
            # ------------------------------------------------------------
            try:
                G1, G2, C1, C2, p, q, perm = generate_gw_graph_instance_noisy(
                    n=n_samples, edge_prob=edge_prob, noise_level=noise_level, seed=seed
                )
                C1 = C1 / C1.max()
                C2 = C2 / C2.max()
            except Exception as e:
                print(f"[n={n_samples}, seed={seed}] Instance generation failed: {e}")
                continue

            # ------------------------------------------------------------
            # CG
            # ------------------------------------------------------------
            if run_methods.get("CG", False):
                run_solver(
                    records,
                    n_samples,
                    seed,
                    "CG",
                    np.nan,
                    lambda: ot.gromov.gromov_wasserstein(
                        C1,
                        C2,
                        p,
                        q,
                        loss_fun="square_loss",
                        verbose=False,
                        log=True,
                        max_iter=max_iter,
                        tol=tol,
                    ),
                    perm,
                    p,
                    q,
                )

            # ------------------------------------------------------------
            # Entropic methods
            # ------------------------------------------------------------
            for epsilon in epsilons:
                if run_methods.get("PPA", False):
                    run_solver(
                        records,
                        n_samples,
                        seed,
                        "PPA",
                        epsilon,
                        lambda eps=epsilon: ot.gromov.entropic_gromov_wasserstein(
                            C1,
                            C2,
                            p,
                            q,
                            loss_fun="square_loss",
                            epsilon=eps,
                            solver="PPA",
                            log=True,
                            verbose=False,
                            max_iter=max_iter,
                            tol=tol,
                        ),
                        perm,
                        p,
                        q,
                    )

                if run_methods.get("PGD", False):
                    run_solver(
                        records,
                        n_samples,
                        seed,
                        "PGD",
                        epsilon,
                        lambda eps=epsilon: ot.gromov.entropic_gromov_wasserstein(
                            C1,
                            C2,
                            p,
                            q,
                            loss_fun="square_loss",
                            epsilon=eps,
                            solver="PGD",
                            log=True,
                            verbose=False,
                            max_iter=max_iter,
                            tol=tol,
                        ),
                        perm,
                        p,
                        q,
                    )

                if run_methods.get("BAPG", False):
                    run_solver(
                        records,
                        n_samples,
                        seed,
                        "BAPG",
                        epsilon,
                        lambda eps=epsilon: ot.gromov.BAPG_gromov_wasserstein(
                            C1,
                            C2,
                            p,
                            q,
                            loss_fun="square_loss",
                            epsilon=eps,
                            log=True,
                            verbose=False,
                            max_iter=max_iter,
                            tol=tol,
                        ),
                        perm,
                        p,
                        q,
                    )

            # ------------------------------------------------------------
            # iPG
            # ------------------------------------------------------------
            if run_methods.get("iPG", False):
                run_solver(
                    records,
                    n_samples,
                    seed,
                    "iPG",
                    np.nan,
                    lambda: gwipg(
                        C1,
                        C2,
                        p,
                        q,
                        max_iter=max_iter,
                        max_iter_proj=ipg_max_iter_proj,
                        iter_check=ipg_iter_check,
                        tol=tol,
                        verbose=False,
                        log=True,
                        expon=ipg_expon,
                        lr_const=ipg_lr_const_fn(n_samples),
                    ),
                    perm,
                    p,
                    q,
                )

    # ================================================================
    # Build DataFrame
    # ================================================================
    if not records:
        raise ValueError("no experiment records; check sizes, seeds, and methods")
    df = pd.DataFrame(records)

    print("\n================ Raw results ================\n")
    print(df)

    print("\n================ Failed runs ================\n")
    failed_df = df[df["status"] == "fail"]
    if len(failed_df) == 0:
        print("No failed runs.")
    else:
        print(failed_df[["n_samples", "seed", "method", "epsilon", "error"]].to_string(index=False))

    # ================================================================
    # Aggregate only successful runs
    # ================================================================
    df_success = df[df["status"] == "success"].copy()

    summary = (
        df_success.groupby(["n_samples", "method", "epsilon"], dropna=False)
        .agg(
            loss_mean=("loss", "mean"),
            loss_std=("loss", "std"),
            sparsity_mean=("sparsity", "mean"),
            sparsity_std=("sparsity", "std"),
            feas_mean=("feas", "mean"),
            feas_std=("feas", "std"),
            acc_mean=("acc", "mean"),
            acc_std=("acc", "std"),
            time_mean=("time", "mean"),
            time_std=("time", "std"),
            n_success=("seed", "count"),
        )
        .reset_index()
    )

    attempts = (
        df.groupby(["n_samples", "method", "epsilon"], dropna=False)
        .size()
        .reset_index(name="n_attempts")
    )

    summary = summary.merge(attempts, on=["n_samples", "method", "epsilon"], how="outer")
    summary["n_success"] = summary["n_success"].fillna(0).astype(int)
    summary["n_fail"] = summary["n_attempts"] - summary["n_success"]

    print("\n================ Average over successful seeds ================\n")
    print(summary.to_string(index=False))

    # ================================================================
    # Pretty report table
    # ================================================================
    table = summary.copy()

    table["loss"] = table.apply(lambda r: pm(r["loss_mean"], r["loss_std"], fmt=".2e"), axis=1)
    table["sparsity"] = table.apply(
        lambda r: pm(r["sparsity_mean"], r["sparsity_std"], fmt=".2f"), axis=1
    )
    table["feas"] = table.apply(lambda r: pm(r["feas_mean"], r["feas_std"], fmt=".2e"), axis=1)
    table["acc"] = table.apply(lambda r: pm(r["acc_mean"], r["acc_std"], fmt=".2f"), axis=1)
    table["time"] = table.apply(lambda r: pm(r["time_mean"], r["time_std"], fmt=".2f"), axis=1)

    table["epsilon"] = table["epsilon"].apply(lambda x: "-" if pd.isna(x) else f"{x:.0e}")

    report_table = table[
        [
            "n_samples",
            "method",
            "epsilon",
            "loss",
            "sparsity",
            "feas",
            "acc",
            "time",
            "n_success",
            "n_attempts",
            "n_fail",
        ]
    ]

    print("\n================ Final Table ================\n")
    print(report_table.to_string(index=False))

    return df, summary, report_table


def main(argv=None):
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--sizes", type=int, nargs="+", default=[100, 200, 300, 400, 500])
    parser.add_argument("--n-seeds", type=int, default=20)
    parser.add_argument("--epsilons", type=float, nargs="+", default=[1e-3, 1e-2, 1e-1, 1.0])
    parser.add_argument(
        "--methods",
        nargs="+",
        choices=["CG", "PPA", "EPGD", "BAPG", "iPG"],
        default=["CG", "PPA", "EPGD", "BAPG", "iPG"],
    )
    parser.add_argument("--edge-prob", type=float, default=0.2)
    parser.add_argument("--noise-level", type=float, default=0.1)
    parser.add_argument("--max-iter", type=int, default=5000)
    parser.add_argument("--max-iter-proj", type=int, default=5000)
    parser.add_argument("--tol", type=float, default=1e-9)
    parser.add_argument("--ipg-expon", type=float, default=3.0)
    parser.add_argument("--ipg-iter-check", type=int, default=100)
    parser.add_argument("--output-dir", type=Path, default=Path("outputs/graph_alignment"))
    parser.add_argument("--smoke-test", action="store_true")
    args = parser.parse_args(argv)
    if args.smoke_test:
        args.sizes, args.n_seeds, args.max_iter, args.max_iter_proj = [20], 1, 3, 50
        args.epsilons = [0.1]
    if args.n_seeds < 1 or min(args.sizes) < 2:
        parser.error("positive seed count and sizes >= 2 are required")
    seeds = np.random.default_rng(0).integers(0, 10**9, size=args.n_seeds).tolist()
    methods = {("PGD" if name == "EPGD" else name): True for name in args.methods}
    write_run_metadata(args, args.output_dir)
    raw, summary, report = run_gw_experiments(
        args.sizes,
        seeds,
        args.epsilons,
        args.edge_prob,
        args.noise_level,
        args.max_iter,
        args.tol,
        methods,
        args.ipg_expon,
        lambda n: 10.0 * n,
        args.max_iter_proj,
        args.ipg_iter_check,
    )
    raw.to_csv(args.output_dir / "graph_alignment_raw.csv", index=False)
    summary.to_csv(args.output_dir / "graph_alignment_summary.csv", index=False)
    report.to_csv(args.output_dir / "graph_alignment_table.csv", index=False)
    print(f"Outputs written to {args.output_dir.resolve()}")


if __name__ == "__main__":
    main()

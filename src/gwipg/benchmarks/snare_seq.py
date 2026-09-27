"""Benchmark GW solvers on the paired SNARE-seq cell-line mixture.

The script follows the preprocessing used by SCOT for this dataset:

1. L2-normalize every cell within each modality.
2. Build a correlation-based k-nearest-neighbor connectivity graph.
3. Use undirected graph shortest-path distances as the intra-domain costs.
4. Normalize each cost matrix by its maximum.

The chromatin and RNA rows describe the same 1,047 cells.  RNA rows are
permuted before optimization, so the original pairing can be used as hidden
ground truth.  In addition to the metrics reported in the paper, the script
reports cell-type matching accuracy and the coupling mass transported between
cells of the same type.

Examples
--------
Quick end-to-end check:

    python -m gwipg.benchmarks.snare_seq --smoke-test

Full paired-data benchmark:

    python -m gwipg.benchmarks.snare_seq \
        --sizes 1047 --seeds 0 --max-iter 5000

Scaling experiment with repeated paired-cell subsets:

    python -m gwipg.benchmarks.snare_seq \
        --sizes 250,500,1047 --seeds 0,1,2,3,4 --max-iter 5000

Data source
-----------
SCOT project: https://rsinghlab.github.io/SCOT/data/
"""

from __future__ import annotations

import argparse
import hashlib
import sys
import time
import urllib.request
import warnings
from pathlib import Path

import numpy as np
import pandas as pd
from scipy.optimize import linear_sum_assignment
from scipy.spatial.distance import cdist
from scipy.sparse import csr_matrix
from scipy.sparse.csgraph import dijkstra
from sklearn.neighbors import KNeighborsClassifier, kneighbors_graph
from sklearn.preprocessing import normalize


from ._common import (
    METHOD_ORDER,
    parse_csv_values,
    unregularized_gw_loss,
    solver_specs,
    mean_std,
    dataframe_to_latex,
    write_run_metadata,
)

DATA_URLS = {
    "snare_chromatin.txt": ("https://rsinghlab.github.io/SCOT/data/snare_chromatin.txt"),
    "snare_rna.txt": "https://rsinghlab.github.io/SCOT/data/snare_rna.txt",
    "SNAREseq_types.txt": ("https://rsinghlab.github.io/SCOT/data/SNAREseq_types.txt"),
}

DATA_SHA256 = {
    "snare_chromatin.txt": "82b383b2a14cdddd85ea7977bd868c76c128f08742d7d2127ed8447202b56251",
    "SNAREseq_types.txt": "160e7b1944af3e991a866f1f48a6512ac985c6dba80fbf19702462c29bb53fcf",
    "snare_rna.txt": "1ddf5340916c21f4cb93136620615c652e7e40f82ae7b2efa36e304134cda2fd",
}


def download_snare_data(data_dir: Path, *, offline=False) -> None:
    """Fetch verified SCOT inputs atomically; offline mode never uses the network."""
    data_dir = Path(data_dir)
    data_dir.mkdir(parents=True, exist_ok=True)
    for filename, url in DATA_URLS.items():
        destination = data_dir / filename
        if not destination.exists():
            if offline:
                raise FileNotFoundError(
                    f"Missing {destination}; rerun without --offline to download"
                )
            print(f"Downloading {url}")
            temporary = destination.with_suffix(destination.suffix + ".part")
            try:
                with urllib.request.urlopen(url, timeout=60) as response:
                    temporary.write_bytes(response.read())
                digest = hashlib.sha256(temporary.read_bytes()).hexdigest()
                if digest != DATA_SHA256[filename]:
                    raise ValueError(
                        f"Downloaded {filename} does not match the paper's data checksum"
                    )
                temporary.replace(destination)
            finally:
                temporary.unlink(missing_ok=True)
        digest = hashlib.sha256(destination.read_bytes()).hexdigest()
        if digest != DATA_SHA256[filename]:
            raise ValueError(
                f"Checksum mismatch for {destination}; expected the archived SCOT input"
            )


def load_snare_data(data_dir: Path, *, offline=False) -> tuple[np.ndarray, np.ndarray, np.ndarray]:
    """Load paired chromatin, RNA, and cell-type arrays."""
    data_dir = Path(data_dir)
    download_snare_data(data_dir, offline=offline)
    chromatin = np.loadtxt(data_dir / "snare_chromatin.txt", dtype=float)
    rna = np.loadtxt(data_dir / "snare_rna.txt", dtype=float)
    cell_types = np.loadtxt(data_dir / "SNAREseq_types.txt", dtype=int).reshape(-1)

    if chromatin.ndim != 2 or rna.ndim != 2:
        raise ValueError("SNARE-seq modality files must be two-dimensional")
    if chromatin.shape[0] != rna.shape[0]:
        raise ValueError(f"modality row counts differ: {chromatin.shape[0]} and {rna.shape[0]}")
    if cell_types.shape[0] != chromatin.shape[0]:
        raise ValueError(f"expected {chromatin.shape[0]} labels, found {cell_types.shape[0]}")
    if not (
        np.isfinite(chromatin).all() and np.isfinite(rna).all() and np.isfinite(cell_types).all()
    ):
        raise ValueError("SNARE-seq files contain non-finite values")

    return chromatin, rna, cell_types


def scot_cost_matrix(
    features: np.ndarray,
    n_neighbors: int = 50,
    metric: str = "correlation",
) -> np.ndarray:
    """Construct the normalized graph-geodesic cost used by SCOT v1."""
    features = normalize(np.asarray(features, dtype=float), norm="l2", axis=1)
    n_samples = features.shape[0]
    if n_samples < 2:
        raise ValueError("at least two cells are required")

    n_neighbors = min(max(1, n_neighbors), n_samples - 1)
    graph = kneighbors_graph(
        features,
        n_neighbors=n_neighbors,
        mode="connectivity",
        metric=metric,
        include_self=True,
    )
    distances = dijkstra(
        csgraph=csr_matrix(graph),
        directed=False,
        return_predecessors=False,
    )

    finite = np.isfinite(distances)
    if not finite.any():
        raise ValueError("the k-nearest-neighbor graph has no finite distances")
    max_finite = float(distances[finite].max())
    if max_finite <= 0:
        raise ValueError("the graph-geodesic cost matrix is identically zero")

    # This matches SCOT's treatment of disconnected components.
    distances[~finite] = max_finite
    distances /= distances.max()
    np.fill_diagonal(distances, 0.0)
    return distances


def paired_subset(
    chromatin: np.ndarray,
    rna: np.ndarray,
    cell_types: np.ndarray,
    n_samples: int,
    seed: int,
) -> tuple[np.ndarray, np.ndarray, np.ndarray, np.ndarray, np.ndarray]:
    """Select paired cells and independently hide the RNA row ordering."""
    total = chromatin.shape[0]
    if not 2 <= n_samples <= total:
        raise ValueError(f"n_samples must lie in [2, {total}], got {n_samples}")

    rng = np.random.default_rng(seed)
    if n_samples == total:
        selected = np.arange(total)
    else:
        selected = np.sort(rng.choice(total, size=n_samples, replace=False))

    chromatin_subset = chromatin[selected]
    rna_subset = rna[selected]
    chromatin_types = cell_types[selected]

    permutation = rng.permutation(n_samples)
    rna_permuted = rna_subset[permutation]
    rna_types = chromatin_types[permutation]

    # true_col_for_row[i] is the permuted RNA row containing cell i.
    true_col_for_row = np.empty(n_samples, dtype=int)
    true_col_for_row[permutation] = np.arange(n_samples)
    return (
        chromatin_subset,
        rna_permuted,
        chromatin_types,
        rna_types,
        true_col_for_row,
    )


def scot_application_metrics(
    coupling: np.ndarray,
    chromatin: np.ndarray,
    rna: np.ndarray,
    true_col_for_row: np.ndarray,
    chromatin_types: np.ndarray,
    rna_types: np.ndarray,
    label_neighbors: int = 5,
) -> dict[str, float]:
    """Evaluate a coupling using SCOT's projection-based metrics.

    SCOT projects the first domain into the feature space of the second:

        X_aligned = diag(Pi 1)^{-1} Pi Y.

    Its FOSCTTM implementation averages, for both directions, the fraction of
    samples closer than the known paired sample.  Label transfer is evaluated
    with a k-nearest-neighbor classifier.  We report both training directions
    and their average because SCOT's paper reports directional LTA tables.
    """
    row_mass = coupling.sum(axis=1)
    if np.any(~np.isfinite(row_mass)) or np.any(row_mass <= 0):
        raise ValueError("SCOT barycentric projection requires positive finite row sums")

    rna_normalized = normalize(np.asarray(rna, dtype=float), norm="l2", axis=1)
    chromatin_aligned = coupling @ rna_normalized / row_mass[:, None]

    # SCOT's FOSCTTM routine assumes that paired observations share a row
    # index.  RNA was permuted to hide the correspondence, so restore only
    # the evaluation order here; the optimization still sees permuted RNA.
    rna_paired_order = rna_normalized[true_col_for_row]
    distances = cdist(chromatin_aligned, rna_paired_order, metric="euclidean")
    true_distances = np.diag(distances)
    denominator = max(1, coupling.shape[0] - 1)
    foscttm_x = np.mean(np.sum(distances < true_distances[:, None], axis=1) / denominator)
    foscttm_y = np.mean(np.sum(distances < true_distances[None, :], axis=0) / denominator)
    foscttm = 0.5 * (foscttm_x + foscttm_y)

    n_neighbors = min(max(1, label_neighbors), coupling.shape[0])
    x_classifier = KNeighborsClassifier(n_neighbors=n_neighbors)
    x_classifier.fit(chromatin_aligned, chromatin_types)
    lta_x_to_y = float(np.mean(x_classifier.predict(rna_normalized) == rna_types))

    y_classifier = KNeighborsClassifier(n_neighbors=n_neighbors)
    y_classifier.fit(rna_normalized, rna_types)
    lta_y_to_x = float(np.mean(y_classifier.predict(chromatin_aligned) == chromatin_types))

    return {
        "foscttm": float(foscttm),
        "foscttm_x": float(foscttm_x),
        "foscttm_y": float(foscttm_y),
        "lta_x_to_y": lta_x_to_y,
        "lta_y_to_x": lta_y_to_x,
        "lta_average": 0.5 * (lta_x_to_y + lta_y_to_x),
    }


def coupling_metrics(
    coupling: np.ndarray,
    cost_x: np.ndarray,
    cost_y: np.ndarray,
    p: np.ndarray,
    q: np.ndarray,
    chromatin: np.ndarray,
    rna: np.ndarray,
    true_col_for_row: np.ndarray,
    chromatin_types: np.ndarray,
    rna_types: np.ndarray,
    label_neighbors: int,
) -> dict[str, float]:
    """Compute optimization, exact-pair, and biological matching metrics."""
    row_ind, col_ind = linear_sum_assignment(-coupling)
    predicted_col = np.empty(coupling.shape[0], dtype=int)
    predicted_col[row_ind] = col_ind

    pair_accuracy = float(np.mean(predicted_col == true_col_for_row))
    type_accuracy = float(np.mean(chromatin_types == rna_types[predicted_col]))
    same_type_mask = chromatin_types[:, None] == rna_types[None, :]
    same_type_mass = float(coupling[same_type_mask].sum())

    metrics = {
        "loss": unregularized_gw_loss(coupling, cost_x, cost_y, p, q),
        "sparsity": float(np.count_nonzero(coupling == 0.0) / coupling.size),
        "feasibility": float(
            np.linalg.norm(coupling.sum(axis=1) - p) + np.linalg.norm(coupling.sum(axis=0) - q)
        ),
        "pair_accuracy": pair_accuracy,
        "type_accuracy": type_accuracy,
        "same_type_mass": same_type_mass,
    }
    metrics.update(
        scot_application_metrics(
            coupling,
            chromatin,
            rna,
            true_col_for_row,
            chromatin_types,
            rna_types,
            label_neighbors=label_neighbors,
        )
    )
    return metrics


def run_benchmark(
    chromatin: np.ndarray,
    rna: np.ndarray,
    cell_types: np.ndarray,
    sizes: list[int],
    seeds: list[int],
    epsilons: list[float],
    methods: list[str],
    n_neighbors: int,
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
    label_neighbors: int,
    bapg_epsilons: list[float] | None = None,
) -> pd.DataFrame:
    """Run all requested solvers, retaining failures as table rows."""
    records: list[dict] = []

    for n_samples in sizes:
        for seed in seeds:
            print(f"\nPreparing SNARE-seq n={n_samples}, seed={seed}")
            (
                chromatin_subset,
                rna_subset,
                chromatin_types,
                rna_types,
                true_col_for_row,
            ) = paired_subset(chromatin, rna, cell_types, n_samples=n_samples, seed=seed)

            cost_start = time.perf_counter()
            cost_x = scot_cost_matrix(chromatin_subset, n_neighbors=n_neighbors)
            cost_y = scot_cost_matrix(rna_subset, n_neighbors=n_neighbors)
            cost_time = time.perf_counter() - cost_start
            p = np.full(n_samples, 1.0 / n_samples)
            q = np.full(n_samples, 1.0 / n_samples)

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
                bapg_epsilons=bapg_epsilons,
            )

            for method, epsilon, solve in specs:
                epsilon_label = "-" if np.isnan(epsilon) else f"{epsilon:.0e}"
                print(f"Running {method:>4s}, n={n_samples}, seed={seed}, epsilon={epsilon_label}")
                record = {
                    "n_samples": n_samples,
                    "seed": seed,
                    "method": method,
                    "epsilon": epsilon,
                    "cost_time": cost_time,
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
                    if coupling.shape != (n_samples, n_samples):
                        raise ValueError(
                            f"solver returned shape {coupling.shape}, "
                            f"expected {(n_samples, n_samples)}"
                        )
                    if not np.isfinite(coupling).all():
                        raise FloatingPointError("solver returned a non-finite coupling")

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
                        label_neighbors,
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
                            "solver_internal_time": solver_log.get("solver_internal_time", np.nan),
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
                        f"  loss={metrics['loss']:.4e}, "
                        f"sparsity={metrics['sparsity']:.3f}, "
                        f"feas={metrics['feasibility']:.2e}, "
                        f"FOSCTTM={metrics['foscttm']:.3f}, "
                        f"LTA={metrics['lta_x_to_y']:.3f}, "
                        f"pair_acc={metrics['pair_accuracy']:.3f}, "
                        f"type_acc={metrics['type_accuracy']:.3f}, "
                        f"time={runtime:.2f}s"
                    )
                except Exception as exc:  # Keep failed settings in the report.
                    record.update(
                        {
                            "loss": np.nan,
                            "sparsity": np.nan,
                            "feasibility": np.nan,
                            "pair_accuracy": np.nan,
                            "type_accuracy": np.nan,
                            "same_type_mass": np.nan,
                            "foscttm": np.nan,
                            "foscttm_x": np.nan,
                            "foscttm_y": np.nan,
                            "lta_x_to_y": np.nan,
                            "lta_y_to_x": np.nan,
                            "lta_average": np.nan,
                            "proj_calls": np.nan,
                            "proj_iter_mean": np.nan,
                            "proj_iter_max": np.nan,
                            "proj_cap_hits": np.nan,
                            "proj_cap_hit_fraction": np.nan,
                            "outer_iterations": np.nan,
                            "gw_gradient_time": np.nan,
                            "projection_time": np.nan,
                            "other_time": np.nan,
                            "solver_internal_time": np.nan,
                            "gw_gradient_time_fraction": np.nan,
                            "projection_time_fraction": np.nan,
                            "time": np.nan,
                            "status": "fail",
                            "error": f"{type(exc).__name__}: {exc}",
                        }
                    )
                    print(f"  FAILED: {record['error']}", file=sys.stderr)
                records.append(record)

    return pd.DataFrame.from_records(records)


def aggregate_results(raw: pd.DataFrame) -> pd.DataFrame:
    """Aggregate successful runs and retain the number of failures."""
    group_keys = ["n_samples", "method", "epsilon"]
    successful = raw[raw["status"] == "success"]
    metrics = [
        "loss",
        "sparsity",
        "feasibility",
        "pair_accuracy",
        "type_accuracy",
        "same_type_mass",
        "foscttm",
        "foscttm_x",
        "foscttm_y",
        "lta_x_to_y",
        "lta_y_to_x",
        "lta_average",
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
        "cost_time",
    ]

    if successful.empty:
        summary = pd.DataFrame(columns=group_keys)
    else:
        summary = (
            successful.groupby(group_keys, dropna=False)[metrics].agg(["mean", "std"]).reset_index()
        )
        summary.columns = [
            "_".join(part for part in column if part) if isinstance(column, tuple) else column
            for column in summary.columns
        ]

    attempts = raw.groupby(group_keys, dropna=False).size().reset_index(name="n_attempts")
    successes = successful.groupby(group_keys, dropna=False).size().reset_index(name="n_success")
    summary = attempts.merge(successes, on=group_keys, how="left").merge(
        summary, on=group_keys, how="left"
    )
    summary["n_success"] = summary["n_success"].fillna(0).astype(int)
    summary["n_fail"] = summary["n_attempts"] - summary["n_success"]
    summary["_method_order"] = summary["method"].map(METHOD_ORDER)
    summary = summary.sort_values(
        ["n_samples", "_method_order", "epsilon"], na_position="first"
    ).drop(columns="_method_order")
    return summary.reset_index(drop=True)


def paper_table(summary: pd.DataFrame) -> pd.DataFrame:
    """Create a compact table similar to the manuscript's result tables."""
    rows = []
    for _, result in summary.iterrows():
        rows.append(
            {
                "n": int(result["n_samples"]),
                "Method": result["method"],
                "epsilon": ("-" if pd.isna(result["epsilon"]) else f"{result['epsilon']:.0e}"),
                "Loss": mean_std(result.get("loss_mean"), result.get("loss_std"), True),
                "Sparsity": mean_std(
                    result.get("sparsity_mean"),
                    result.get("sparsity_std"),
                ),
                "Feasibility": mean_std(
                    result.get("feasibility_mean"),
                    result.get("feasibility_std"),
                    True,
                ),
                "FOSCTTM": mean_std(
                    result.get("foscttm_mean"),
                    result.get("foscttm_std"),
                ),
                "LTA X->Y": mean_std(
                    result.get("lta_x_to_y_mean"),
                    result.get("lta_x_to_y_std"),
                ),
                "LTA Y->X": mean_std(
                    result.get("lta_y_to_x_mean"),
                    result.get("lta_y_to_x_std"),
                ),
                "LTA Avg.": mean_std(
                    result.get("lta_average_mean"),
                    result.get("lta_average_std"),
                ),
                "Pair Acc.": mean_std(
                    result.get("pair_accuracy_mean"),
                    result.get("pair_accuracy_std"),
                ),
                "Type Acc.": mean_std(
                    result.get("type_accuracy_mean"),
                    result.get("type_accuracy_std"),
                ),
                "Same-type Mass": mean_std(
                    result.get("same_type_mass_mean"),
                    result.get("same_type_mass_std"),
                ),
                "Proj. Iter.": mean_std(
                    result.get("proj_iter_mean_mean"),
                    result.get("proj_iter_mean_std"),
                ),
                "Proj. Max": mean_std(
                    result.get("proj_iter_max_mean"),
                    result.get("proj_iter_max_std"),
                ),
                "Proj. Cap Rate": mean_std(
                    result.get("proj_cap_hit_fraction_mean"),
                    result.get("proj_cap_hit_fraction_std"),
                ),
                "Outer Iter.": mean_std(
                    result.get("outer_iterations_mean"),
                    result.get("outer_iterations_std"),
                ),
                "GW Grad. (s)": mean_std(
                    result.get("gw_gradient_time_mean"),
                    result.get("gw_gradient_time_std"),
                ),
                "Projection (s)": mean_std(
                    result.get("projection_time_mean"),
                    result.get("projection_time_std"),
                ),
                "Proj. Time Rate": mean_std(
                    result.get("projection_time_fraction_mean"),
                    result.get("projection_time_fraction_std"),
                ),
                "Time (s)": mean_std(result.get("time_mean"), result.get("time_std")),
                "Success": f"{int(result['n_success'])}/{int(result['n_attempts'])}",
            }
        )
    return pd.DataFrame(rows)


def write_outputs(
    raw: pd.DataFrame,
    summary: pd.DataFrame,
    table: pd.DataFrame,
    output_dir: Path,
) -> None:
    output_dir.mkdir(parents=True, exist_ok=True)
    raw.to_csv(output_dir / "snare_seq_raw.csv", index=False)
    summary.to_csv(output_dir / "snare_seq_summary.csv", index=False)
    table.to_csv(output_dir / "snare_seq_table.csv", index=False)
    (output_dir / "snare_seq_table.md").write_text(
        table.to_markdown(index=False) + "\n", encoding="utf-8"
    )
    (output_dir / "snare_seq_table.tex").write_text(dataframe_to_latex(table), encoding="utf-8")


def build_parser() -> argparse.ArgumentParser:
    script_dir = Path.cwd()
    parser = argparse.ArgumentParser(description="Compare GW solvers on paired SNARE-seq data.")
    parser.add_argument(
        "--offline", action="store_true", help="Use cached data without network access."
    )
    parser.add_argument("--sizes", default="1047")
    parser.add_argument("--seeds", default="0")
    parser.add_argument(
        "--epsilons",
        default="5e-4,1e-3,1e-2,1e-1,1e0",
        help="Entropy parameters; 5e-4 is the SCOT SNARE-seq setting.",
    )
    parser.add_argument("--methods", default="CG,PPA,EPGD,BAPG,iPG")
    parser.add_argument(
        "--bapg-epsilons",
        default=None,
        help=(
            "Optional comma-separated subset of --epsilons used by BAPG; "
            "defaults to every requested epsilon."
        ),
    )
    parser.add_argument("--neighbors", type=int, default=50)
    parser.add_argument("--max-iter", type=int, default=5000)
    parser.add_argument("--tol", type=float, default=1e-9)
    parser.add_argument(
        "--ipg-tol",
        type=float,
        default=None,
        help="iPG outer tolerance; defaults to --tol when omitted.",
    )
    parser.add_argument("--ipg-expon", type=float, default=3.0)
    parser.add_argument("--ipg-lr-scale", type=float, default=10.0)
    parser.add_argument("--ipg-max-iter-proj", type=int, default=5000)
    parser.add_argument("--ipg-proj-tol-scale", type=float, default=1.0)
    parser.add_argument("--ipg-proj-lr-const", type=float, default=2.0)
    parser.add_argument("--ipg-iter-check", type=int, default=100)
    parser.add_argument(
        "--ipg-track-proj-iters",
        action="store_true",
        help="Record how often iPG projections reach their iteration cap.",
    )
    parser.add_argument(
        "--ipg-track-timing",
        action="store_true",
        help="Record iPG time spent in GW gradients and projections.",
    )
    parser.add_argument(
        "--label-neighbors",
        type=int,
        default=5,
        help="Number of neighbors for SCOT label-transfer accuracy.",
    )
    parser.add_argument(
        "--data-dir",
        type=Path,
        default=script_dir / "data" / "snare_seq",
    )
    parser.add_argument(
        "--output-dir",
        type=Path,
        default=script_dir / "outputs" / "snare_seq",
    )
    parser.add_argument(
        "--smoke-test",
        action="store_true",
        help="Use n=40, two outer iterations, and one epsilon.",
    )
    return parser


def main(argv=None) -> None:
    args = build_parser().parse_args(argv)
    if args.smoke_test:
        sizes = [40]
        seeds = [0]
        epsilons = [1e-2]
        methods = ["CG", "PPA", "EPGD", "BAPG", "iPG"]
        max_iter = 2
        ipg_max_iter_proj = 100
        output_dir = args.output_dir / "smoke_test"
    else:
        sizes = parse_csv_values(args.sizes, int)
        seeds = parse_csv_values(args.seeds, int)
        epsilons = parse_csv_values(args.epsilons, float)
        bapg_epsilons = (
            None if args.bapg_epsilons is None else parse_csv_values(args.bapg_epsilons, float)
        )
        methods = [method.strip() for method in args.methods.split(",") if method.strip()]
        max_iter = args.max_iter
        ipg_max_iter_proj = args.ipg_max_iter_proj
        output_dir = args.output_dir

    if args.smoke_test:
        bapg_epsilons = None
    elif bapg_epsilons is not None:
        unknown_bapg_epsilons = set(bapg_epsilons) - set(epsilons)
        if unknown_bapg_epsilons:
            raise ValueError(
                "--bapg-epsilons must be a subset of --epsilons; "
                f"unexpected values: {sorted(unknown_bapg_epsilons)}"
            )

    chromatin, rna, cell_types = load_snare_data(args.data_dir, offline=args.offline)
    if max(sizes) > chromatin.shape[0]:
        raise ValueError(
            f"requested n={max(sizes)}, but SNARE-seq has {chromatin.shape[0]} paired cells"
        )

    with warnings.catch_warnings():
        warnings.simplefilter("default")
        raw = run_benchmark(
            chromatin=chromatin,
            rna=rna,
            cell_types=cell_types,
            sizes=sizes,
            seeds=seeds,
            epsilons=epsilons,
            methods=methods,
            n_neighbors=args.neighbors,
            max_iter=max_iter,
            tol=args.tol,
            ipg_tol=args.tol if args.ipg_tol is None else args.ipg_tol,
            ipg_expon=args.ipg_expon,
            ipg_lr_scale=args.ipg_lr_scale,
            ipg_max_iter_proj=ipg_max_iter_proj,
            ipg_proj_tol_scale=args.ipg_proj_tol_scale,
            ipg_proj_lr_const=args.ipg_proj_lr_const,
            ipg_iter_check=args.ipg_iter_check,
            ipg_track_proj_iters=args.ipg_track_proj_iters,
            ipg_track_timing=args.ipg_track_timing,
            label_neighbors=args.label_neighbors,
            bapg_epsilons=bapg_epsilons,
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
        print(failures[["n_samples", "seed", "method", "epsilon", "error"]].to_string(index=False))


if __name__ == "__main__":
    main()

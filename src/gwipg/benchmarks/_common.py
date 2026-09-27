"""Shared solver adapters, output formatting, and run provenance."""

from __future__ import annotations

import argparse
import importlib.metadata
import json
import os
import platform
from pathlib import Path
from typing import Callable

import numpy as np
import ot
import pandas as pd

from gwipg import gwipg

METHOD_ORDER = {"CG": 0, "PPA": 1, "EPGD": 2, "BAPG": 3, "iPG": 4}


def parse_csv_values(text: str, cast: Callable) -> list:
    """Parse a comma-separated CLI value."""
    values = [part.strip() for part in text.split(",") if part.strip()]
    if not values:
        raise argparse.ArgumentTypeError("expected at least one value")
    try:
        return [cast(value) for value in values]
    except ValueError as exc:
        raise argparse.ArgumentTypeError(str(exc)) from exc


def unregularized_gw_loss(
    coupling: np.ndarray,
    cost_x: np.ndarray,
    cost_y: np.ndarray,
    p: np.ndarray,
    q: np.ndarray,
) -> float:
    """Evaluate every solver using the same squared-loss GW objective."""
    const_c, h_c1, h_c2 = ot.gromov.init_matrix(cost_x, cost_y, p, q, loss_fun="square_loss")
    return float(ot.gromov.gwloss(const_c, h_c1, h_c2, coupling))


def solver_specs(
    methods: list[str],
    epsilons: list[float],
    cost_x: np.ndarray,
    cost_y: np.ndarray,
    p: np.ndarray,
    q: np.ndarray,
    max_iter: int,
    tol: float,
    ipg_tol: float,
    ipg_expon: float,
    ipg_lr_const: float,
    ipg_max_iter_proj: int,
    ipg_proj_tol_scale: float,
    ipg_proj_lr_const: float,
    ipg_iter_check: int,
    ipg_track_proj_iters: bool,
    ipg_track_timing: bool,
    bapg_epsilons: list[float] | None = None,
) -> list[tuple[str, float, Callable]]:
    """Create lazily evaluated solver calls for one dataset instance."""
    requested = set(methods)
    unknown = requested - set(METHOD_ORDER)
    if unknown:
        raise ValueError(f"unknown methods: {sorted(unknown)}")

    specs: list[tuple[str, float, Callable]] = []
    if "CG" in requested:
        specs.append(
            (
                "CG",
                np.nan,
                lambda: ot.gromov.gromov_wasserstein(
                    cost_x,
                    cost_y,
                    p,
                    q,
                    loss_fun="square_loss",
                    log=True,
                    verbose=False,
                    max_iter=max_iter,
                    tol_rel=1e-9,
                    tol_abs=1e-9,
                ),
            )
        )

    bapg_epsilon_set = set(epsilons) if bapg_epsilons is None else set(bapg_epsilons)
    for epsilon in epsilons:
        if "PPA" in requested:
            specs.append(
                (
                    "PPA",
                    epsilon,
                    lambda epsilon=epsilon: ot.gromov.entropic_gromov_wasserstein(
                        cost_x,
                        cost_y,
                        p,
                        q,
                        loss_fun="square_loss",
                        epsilon=epsilon,
                        solver="PPA",
                        log=True,
                        verbose=False,
                        max_iter=max_iter,
                        tol=tol,
                    ),
                )
            )
        if "EPGD" in requested:
            specs.append(
                (
                    "EPGD",
                    epsilon,
                    lambda epsilon=epsilon: ot.gromov.entropic_gromov_wasserstein(
                        cost_x,
                        cost_y,
                        p,
                        q,
                        loss_fun="square_loss",
                        epsilon=epsilon,
                        solver="PGD",
                        log=True,
                        verbose=False,
                        max_iter=max_iter,
                        tol=tol,
                    ),
                )
            )
        if "BAPG" in requested and epsilon in bapg_epsilon_set:
            specs.append(
                (
                    "BAPG",
                    epsilon,
                    lambda epsilon=epsilon: ot.gromov.BAPG_gromov_wasserstein(
                        cost_x,
                        cost_y,
                        p,
                        q,
                        loss_fun="square_loss",
                        epsilon=epsilon,
                        log=True,
                        verbose=False,
                        max_iter=max_iter,
                        tol=tol,
                    ),
                )
            )

    if "iPG" in requested:
        specs.append(
            (
                "iPG",
                np.nan,
                lambda: gwipg(
                    cost_x,
                    cost_y,
                    p,
                    q,
                    max_iter=max_iter,
                    max_iter_proj=ipg_max_iter_proj,
                    tol=ipg_tol,
                    iter_check=ipg_iter_check,
                    verbose=False,
                    log=True,
                    expon=ipg_expon,
                    lr_const=ipg_lr_const,
                    proj_tol_scale=ipg_proj_tol_scale,
                    proj_lr_const=ipg_proj_lr_const,
                    track_proj_iters=ipg_track_proj_iters,
                    track_timing=ipg_track_timing,
                ),
            )
        )
    return specs


def mean_std(mean: float, std: float, scientific: bool = False) -> str:
    """Format a paper-style mean +/- standard deviation entry."""
    if pd.isna(mean):
        return "-"
    fmt = ".2e" if scientific else ".2f"
    if pd.isna(std):
        return format(mean, fmt)
    return f"{format(mean, fmt)} +/- {format(std, fmt)}"


def latex_escape(text: str) -> str:
    return (
        text.replace("\\", r"\textbackslash{}")
        .replace("&", r"\&")
        .replace("%", r"\%")
        .replace("_", r"\_")
    )


def dataframe_to_latex(table: pd.DataFrame) -> str:
    """Render a dependency-stable LaTeX tabular environment."""
    columns = list(table.columns)
    alignment = "l" * len(columns)
    lines = [
        rf"\begin{{tabular}}{{{alignment}}}",
        r"\toprule",
        " & ".join(latex_escape(str(column)) for column in columns) + r" \\",
        r"\midrule",
    ]
    for row in table.itertuples(index=False, name=None):
        lines.append(" & ".join(latex_escape(str(value)) for value in row) + r" \\")
    lines.extend([r"\bottomrule", r"\end{tabular}", ""])
    return "\n".join(lines)


def write_run_metadata(args, output_dir):
    """Save configuration and environment beside new runs, not paper archives."""
    output_dir = Path(output_dir)
    output_dir.mkdir(parents=True, exist_ok=True)
    versions = {}
    for name in ["gwipg", "numpy", "scipy", "POT", "pandas", "networkx", "scikit-learn"]:
        try:
            versions[name] = importlib.metadata.version(name)
        except importlib.metadata.PackageNotFoundError:
            pass
    metadata = {
        "arguments": vars(args),
        "python": platform.python_version(),
        "platform": platform.platform(),
        "dependencies": versions,
        "threads": {
            key: os.environ.get(key)
            for key in [
                "OMP_NUM_THREADS",
                "OPENBLAS_NUM_THREADS",
                "MKL_NUM_THREADS",
                "VECLIB_MAXIMUM_THREADS",
            ]
        },
    }
    (output_dir / "run_metadata.json").write_text(
        json.dumps(metadata, indent=2, default=str) + "\n", encoding="utf-8"
    )

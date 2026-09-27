"""Plot five graph metrics from a fresh or archived summary CSV."""

import argparse
from pathlib import Path

import matplotlib
import numpy as np
import pandas as pd

matplotlib.use("Agg")
import matplotlib.pyplot as plt


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument(
        "summary",
        type=Path,
        nargs="?",
        default=(
            Path(__file__).resolve().parents[1]
            / "paper_results/graph_alignment/summary_rounded.csv"
        ),
    )
    parser.add_argument("--output-dir", type=Path, default=Path("outputs/graph_figures"))
    args = parser.parse_args()
    data = pd.read_csv(args.summary)
    data["method"] = data["method"].replace({"PGD": "EPGD"})
    args.output_dir.mkdir(parents=True, exist_ok=True)
    metrics = [
        ("loss", "Loss", True),
        ("sparsity", "Sparsity", True),
        ("feas", "Feasibility", True),
        ("acc", "Accuracy", False),
        ("time", "Time (s)", False),
    ]
    for metric, label, logarithmic in metrics:
        fig, ax = plt.subplots(figsize=(8, 5))
        for (method, epsilon), rows in data.groupby(["method", "epsilon"], dropna=False):
            rows = rows.sort_values("n_samples")
            values = rows[f"{metric}_mean"].to_numpy()
            if logarithmic:
                # Visualization floor only; never change stored measurements.
                values = np.maximum(values, 1e-17)
            legend = method if pd.isna(epsilon) else f"{method} ({epsilon:.0e})"
            ax.plot(rows.n_samples, values, marker="o", label=legend)
        if logarithmic:
            ax.set_yscale("log")
        ax.set_xlabel("Number of samples")
        ax.set_ylabel(label)
        ax.grid(alpha=0.3)
        ax.legend(fontsize=7, ncol=3, loc="upper center", bbox_to_anchor=(0.5, 1.25))
        fig.tight_layout()
        name = {"feas": "feasibility", "acc": "accuracy"}.get(metric, metric)
        fig.savefig(args.output_dir / f"{name}_vs_nsamples.pdf", bbox_inches="tight")
        plt.close(fig)


if __name__ == "__main__":
    main()

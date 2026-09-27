"""Gaussian point-cloud comparison used for the paper's illustrative figure."""

import argparse
from pathlib import Path

import numpy as np
import ot
import pandas as pd
from scipy.spatial.distance import cdist

from ._common import solver_specs, unregularized_gw_loss, write_run_metadata


def main(argv=None):
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--n", type=int, default=30)
    parser.add_argument("--seed", type=int, default=42)
    parser.add_argument("--max-iter", type=int, default=100000)
    parser.add_argument("--output-dir", type=Path, default=Path("outputs/gaussian"))
    parser.add_argument("--smoke-test", action="store_true")
    args = parser.parse_args(argv)
    if args.smoke_test:
        args.n, args.max_iter = 12, 2
    if args.n < 2 or args.max_iter < 1:
        parser.error("n >= 2 and max-iter >= 1 are required")
    # RandomState retains the original np.random.seed(42) sequence without
    # modifying the caller's global random state.
    rng = np.random.RandomState(args.seed)
    x = ot.datasets.make_2D_samples_gauss(args.n, np.zeros(2), np.eye(2), random_state=rng)
    y = rng.randn(args.n, 3) + 4.0
    C1, C2 = cdist(x, x), cdist(y, y)
    C1 /= C1.max()
    C2 /= C2.max()
    p = np.full(args.n, 1 / args.n)
    specs = solver_specs(
        ["CG", "PPA", "EPGD", "BAPG", "iPG"],
        [5e-4],
        C1,
        C2,
        p,
        p,
        args.max_iter,
        1e-9,
        1e-5,
        2.1,
        100.0,
        50 if args.smoke_test else 5000,
        1.0,
        2.0,
        100,
        True,
        False,
        bapg_epsilons=[],
    )
    specs.insert(
        -1,
        (
            "BAPG",
            0.1,
            lambda: ot.gromov.BAPG_gromov_wasserstein(
                C1,
                C2,
                p,
                p,
                loss_fun="square_loss",
                epsilon=0.1,
                log=True,
                max_iter=args.max_iter,
            ),
        ),
    )
    import matplotlib

    matplotlib.use("Agg")
    import matplotlib.pyplot as plt

    write_run_metadata(args, args.output_dir)
    fig, axes = plt.subplots(1, len(specs), figsize=(15, 3.5))
    rows = []
    for ax, (method, epsilon, solve) in zip(axes, specs):
        T, _ = solve()
        if not np.isfinite(T).all():
            raise FloatingPointError(f"{method} returned non-finite values")
        loss = unregularized_gw_loss(T, C1, C2, p, p)
        feasibility = np.linalg.norm(T.sum(1) - p) + np.linalg.norm(T.sum(0) - p)
        sparsity = np.mean(T == 0)
        rows.append(
            dict(
                method=method,
                epsilon=epsilon,
                loss=loss,
                feasibility=feasibility,
                sparsity=sparsity,
            )
        )
        ax.imshow(T, cmap="Blues")
        ax.set_title(
            f"{method}: loss={loss:.4f}\nzeros={sparsity:.1%}; residual={feasibility:.1e}",
            fontsize=9,
        )
        ax.set_xticks([])
        ax.set_yticks([])
    fig.tight_layout()
    fig.savefig(args.output_dir / "gaussian_couplings.png", dpi=160)
    plt.close(fig)
    pd.DataFrame(rows).to_csv(args.output_dir / "gaussian_results.csv", index=False)
    print(f"Outputs written to {args.output_dir.resolve()}")


if __name__ == "__main__":
    main()

"""A small, deterministic GW example: python examples/quickstart.py."""

import numpy as np
from scipy.spatial.distance import cdist

from gwipg import gwipg


def main():
    rng = np.random.default_rng(42)
    x, y = rng.normal(size=(12, 2)), rng.normal(size=(15, 3))
    C1, C2 = cdist(x, x), cdist(y, y)
    C1 /= C1.max()
    C2 /= C2.max()
    T, info = gwipg(
        C1,
        C2,
        max_iter=100,
        iter_check=5,
        tol=1e-6,
        adaptive_step=False,
        proj_lr_const=1.0,
        track_proj_iters=True,
    )
    print(f"Coupling: {T.shape}; loss: {info['gw_dist']:.6f}")
    print(f"Marginal residual: {info['feasibility']:.3e}; status: {info['status']}")
    print(f"Inner residual failures: {info['projection_criterion_failures']}")


if __name__ == "__main__":
    main()

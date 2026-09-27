"""Project a rectangular matrix with nonuniform marginals."""

import numpy as np

from gwipg import proj_transport_polytope


def main():
    rng = np.random.default_rng(42)
    p, q = rng.random(8), rng.random(10)
    p /= p.sum()
    q /= q.sum()
    T, _, info = proj_transport_polytope(
        rng.normal(size=(8, 10)),
        p,
        q,
        max_iter=10000,
        tol=1e-7,
        lr_const=1.0,
        return_log=False,
        return_stats=True,
    )
    print(f"Projection shape: {T.shape}; residual: {info['final_err']:.3e}")
    print(f"Converged: {info['converged']}; evaluations: {info['n_iter']}")


if __name__ == "__main__":
    main()

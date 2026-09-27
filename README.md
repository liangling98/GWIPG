# GWIPG

**Inexact projected gradient for Gromov-Wasserstein optimal transport.**

This Python package implements the dense iPG solver and Euclidean transport
projection from *A Provably Convergent and Practical Algorithm for
Gromov-Wasserstein Optimal Transport* by Ling Liang and Lei Yang, accepted at
**NeurIPS 2026**. It includes the paper's graph alignment, paired single-cell
alignment, sphere benchmarks, and projection diagnostics.

## Installation

Requires Python 3.10 or newer. From the downloaded/cloned repository:

```bash
python -m venv .venv
source .venv/bin/activate  # Windows: .venv\Scripts\activate
python -m pip install --upgrade pip
python -m pip install -e '.[examples]'
```

For the solver alone, use `python -m pip install .`. Core dependencies are
NumPy, SciPy, and [POT](https://pythonot.github.io/). Benchmark dependencies
are installed only with the `examples` extra. Installation from a local
checkout is supported; this README does not assume a PyPI release exists.

## Quick start

```python
import numpy as np
from scipy.spatial.distance import cdist
from gwipg import gwipg

rng = np.random.default_rng(42)
x, y = rng.normal(size=(12, 2)), rng.normal(size=(15, 3))
C1, C2 = cdist(x, x), cdist(y, y)
C1 /= C1.max()
C2 /= C2.max()

# Uniform marginals are supplied automatically.
T, info = gwipg(
    C1,
    C2,
    max_iter=10_000,
    max_iter_proj=5_000,
    tol=1e-6,
    iter_check=5,
    lr_const=1.0,
    adaptive_step=False,
    proj_lr_const=1.0,
    projection_failure="raise",
    track_proj_iters=True,
)
assert info["converged"], info["status"]
assert info["all_projections_converged"]
print(T.shape, info['gw_dist'], info['feasibility'], info['status'])
```

This seeded example reports `status="converged"` with every projection
passing its residual check. It took 5,241 outer iterations in the local
validation run, with loss approximately 0.0307164 and marginal residual
approximately 3.53e-9; counts and final digits can vary across environments.
The 10,000-iteration budget gives it room to reach the explicit outer
tolerance of 1e-6. The assertions also run in `examples/quickstart.py`.

`T` is a nonnegative transport matrix of shape `(12, 15)`. `gw_dist` is the
full squared-loss GW score. The solver supports rectangular couplings,
nonuniform marginals, warm initialization, and asymmetric costs. It uses
dense CPU arrays. See the [API guide](docs/api.md) for stopping rules,
projection diagnostics, and the distinction between practical settings and
the theorem's assumptions.

```bash
python examples/quickstart.py
python examples/projection.py
gwipg-paper --list
gwipg-paper all --quick
```

The last command runs small versions of all six paper examples. SNARE-seq
data are downloaded and cached on first use. `--offline` requires a populated
cache. Quick runs test the pipelines, not convergence or paper accuracy.

## Solver options

The following table lists **every argument to `gwipg`**, including defaults
and accepted values. Here, `m` and `n` are the source and target sizes.

| Argument | Type / accepted values | Default | Meaning |
| --- | --- | --- | --- |
| `C1` | Finite real array, shape `(m, m)` | Required | Source cost or dissimilarity matrix. |
| `C2` | Finite real array, shape `(n, n)` | Required | Target cost or dissimilarity matrix. It may have a different size from `C1`. |
| `p` | Nonnegative vector of length `m`, or `None` | `None` | Source marginal. `None` gives uniform masses `1/m`. |
| `q` | Nonnegative vector of length `n`, or `None` | `None` | Target marginal. `None` gives uniform masses `1/n`. Both marginals must have the same positive finite total mass. |
| `G0` | Finite nonnegative array, shape `(m, n)`, or `None` | `None` | Initial coupling. `None` uses `outer(p, q) / p.sum()`. A supplied coupling need not be exactly feasible. |
| `max_iter` | Positive integer | `10000` | Maximum number of outer GW iterations. Increase this if the returned status is `max_iter`. |
| `max_iter_proj` | Positive integer | `5000` | Maximum inner iterations for each Euclidean projection. Increase this when inner residual checks fail. |
| `tol` | Nonnegative finite float | `1e-9` | Outer stopping tolerance: the **absolute** Frobenius norm of the change in the coupling must be at most this value. |
| `iter_check` | Positive integer | `100` | Check the outer stopping condition and record progress at zero-based iterations `0, iter_check, 2*iter_check, ...`. |
| `verbose` | Boolean | `False` | Print the iteration, change in the coupling, loss, inner count, and step size at outer check points. |
| `log` | Boolean | `True` | Return `(T, info)` when true; return only `T` when false. |
| `expon` | Positive finite float | `2.1` | Decay exponent in the inner tolerance schedule `epsilon_k = proj_tol_scale / (k+1)**expon`. |
| `lr_const` | Positive finite float | `1.0` | Multiplier for the initial outer step `lr_const / L`, where `L` is defined below. |
| `proj_tol_scale` | Positive finite float | `1.0` | Numerator of the inner tolerance schedule. Smaller values demand more accurate projections. |
| `proj_lr_const` | Positive finite float | `2.0` | Multiplier for the dual projection step `proj_lr_const / (m+n)`. The default is a practical setting; use `1.0` for the usual inner Lipschitz step. |
| `track_proj_iters` | Boolean | `False` | Include per-projection iteration counts, their mean/maximum, and budget-hit statistics in `info`. Requires `log=True` to retrieve them. |
| `track_timing` | Boolean | `False` | Measure total solver, gradient, projection, and remaining time; include component times and fractions in `info` when `log=True`. |
| `symmetric` | `None`, `True`, or `False` | `None` | `None` detects symmetry of both costs; `True` requires symmetric costs; `False` uses the general asymmetric-cost gradient. |
| `adaptive_step` | Boolean; keyword-only | `True` | Enable the original heuristic secant-step updates. `False` keeps the outer step constant, as in the quick start. |
| `projection_failure` | `"warn"`, `"raise"`, or `"ignore"`; keyword-only | `"warn"` | On an unsatisfied inner residual check, warn once at the end, raise immediately, or continue quietly. Failure counts remain available with `log=True`. |

`L = 4 * norm(C1, 2) * norm(C2, 2)`, using matrix spectral norms. If `L=0`,
the implementation uses a finite outer step of 1.0. The conservative outer
choice is `lr_const=1.0, adaptive_step=False`. The adaptive heuristic retains
the paper implementation's lagged-gradient timing and skips degenerate
updates; its larger steps are empirical choices.

For the inner projection, `k` starts at zero. Its stopping condition compares
the Euclidean norm of the concatenated row/column marginal residuals with
`epsilon_k / (6 + norm(y, 2))`, where `y` contains the dual potentials.
The reported `info["feasibility"]` instead sums the two marginal residual
norms. Increasing `expon` or decreasing `proj_tol_scale` tightens the inner
targets and may require a larger `max_iter_proj`.

With `log=True`, inspect `info["status"]`: `"converged"` means both the outer
and final inner stopping checks passed; `"max_iter"` means the outer budget
was exhausted without passing its check; `"inexact_projection"` means the
outer check passed but the final inner check failed. Also inspect
`info["all_projections_converged"]` to check every inner solve. These flags
describe the stopping tests and do not certify a global GW optimum. See the
[API guide](docs/api.md) for all returned diagnostics and the standalone
`proj_transport_polytope` interface.

## Paper examples

| Example | Command |
| --- | --- |
| Gaussian coupling illustration | `gwipg-paper gaussian` |
| Synthetic graph alignment | `gwipg-paper graph` |
| Paired SNARE-seq alignment | `gwipg-paper snare` |
| Graph projection timing | `gwipg-paper projection-cost` |
| Inner-accuracy ablation | `gwipg-paper inner-accuracy` |
| Analytical sphere benchmarks | `gwipg-paper spheres` |

Full runs use the stored paper presets and can take substantially longer than
`--quick`. New CSVs, tables, and configuration records are written under
`outputs/`. The [reproduction guide](docs/reproducing.md) explains exact
settings, metrics, timing conventions, dependencies, and plotting commands.
The [code review notes](docs/review_notes.md) document fixes and a SNARE-seq
schedule discrepancy requiring a separate manuscript correction.

## Repository layout

```text
src/gwipg/          Solver, projection, and benchmark modules
examples/          Small examples and paper experiment launchers
tests/             Numerical, regression, data, and CLI tests
paper_results/     Curated archived measurements and checksum manifest
docs/              API, reproduction, and code review notes
figures/           Existing paper figure assets
data/snare_seq/    Download cache (data files excluded from Git)
```

The local `results/` directory contains earlier experiments and is ignored
by Git; curated paper results are kept separately. Existing manuscript
figure paths are retained.

## Development

```bash
python -m pip install -e '.[examples,dev]'
python -m pytest
ruff check src examples tests
ruff format --check src examples tests
python -m build
python -m twine check dist/*.whl dist/*.tar.gz
```

GitHub Actions is configured to test on Linux, macOS, and Windows and build
the distributions. The workflow uses the `GWIPG` directory as the repository
root; publish that directory, not its parent manuscript folder. See
[CONTRIBUTING.md](CONTRIBUTING.md) for numerical change and reproduction
guidelines. Package metadata follows the
[Python Packaging User Guide](https://packaging.python.org/en/latest/guides/writing-pyproject-toml/).
See the [validation record](docs/validation.md) for executed checks and their scope.

## Citation

If you use GWIPG in your research, please cite our **NeurIPS 2026** paper:

> Ling Liang and Lei Yang. **A Provably Convergent and Practical Algorithm for
> Gromov-Wasserstein Optimal Transport.** Accepted at NeurIPS 2026
> (*Advances in Neural Information Processing Systems*).

```bibtex
@inproceedings{liang2026gwipg,
  title     = {A Provably Convergent and Practical Algorithm for {Gromov--Wasserstein} Optimal Transport},
  author    = {Liang, Ling and Yang, Lei},
  booktitle = {Advances in Neural Information Processing Systems},
  year      = {2026},
  note      = {Accepted at NeurIPS 2026}
}
```

The same paper citation is recorded in [CITATION.cff](CITATION.cff).

## License

Code is released under the [MIT license](LICENSE), copyright 2026 GWIPG
contributors. Third-party datasets retain their own terms; see [NOTICE.md](NOTICE.md).

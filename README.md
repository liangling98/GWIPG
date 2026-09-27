# GWIPG

**Inexact projected gradient for Gromov-Wasserstein optimal transport.**

This Python package implements the dense iPG solver and Euclidean transport
projection from *A Provably Convergent and Practical Algorithm for
Gromov-Wasserstein Optimal Transport* by Ling Liang and Lei Yang. It includes
the paper's graph alignment, paired single-cell alignment, sphere benchmarks,
and projection diagnostics.

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
T, info = gwipg(C1, C2, max_iter=100, iter_check=5,
                adaptive_step=False, proj_lr_const=1.0)
print(T.shape, info['gw_dist'], info['feasibility'], info['status'])
```

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

## Citation and license

Please cite the accompanying paper when using this implementation. Author
and title metadata are provided in [CITATION.cff](CITATION.cff); add the final
publication DOI when available.

Code is released under the [MIT license](LICENSE), copyright 2026 GWIPG
contributors. Third-party datasets retain their own terms; see [NOTICE.md](NOTICE.md).

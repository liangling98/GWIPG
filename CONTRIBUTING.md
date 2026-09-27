# Contributing

Create a virtual environment and install `python -m pip install -e '.[examples,dev]'`.
Run `python -m pytest`, `ruff check src examples tests`, and
`ruff format --check src examples tests` before proposing a change.

Keep numerical changes separate from packaging and documentation changes.
For a solver change, explain its effect on the objective, feasibility,
stopping rule, and the paper presets. Use independent numerical checks or
frozen baseline comparisons; do not simply update expected results to make
a failing test pass. New random examples should use explicit seeds.

Do not overwrite `paper_results/` with new timings or regenerated values.
Record a new run under `outputs/` with configuration and dependency versions.
For a bug report, include the input shapes, parameters, Python/POT/NumPy
versions, convergence status, and a small reproducible example.

Contributions to the code are made under the repository's MIT license.

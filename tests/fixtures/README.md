# Legacy solver fixtures

Both NPZ files contain input costs and marginals, the final coupling and loss,
and projection counts produced by an unmodified backup of the original
`gwipg.py` and `proj_transport_polytope.py`. They were generated with Python
3.12, NumPy 2.2.4, SciPy 1.15.3, and POT 0.9.6, using:

```python
max_iter=205, max_iter_proj=500, tol=0.0, iter_check=5,
expon=3.0, lr_const=70.0, proj_lr_const=1.5, track_proj_iters=True
```

The symmetric case uses Euclidean distances between seven two-dimensional
and nine three-dimensional standard-normal points, generated sequentially
with `np.random.default_rng(2026)`, each cost divided by its maximum. The
asymmetric case stores the original randomly generated costs directly.
Marginals are uniform. All input arrays are stored, so regenerating the
points is unnecessary for running the tests.

The symmetric fixture avoids equally spaced points, whose reflection
symmetry allows floating-point roundoff to select different equivalent
couplings. A larger inner budget reduces near-threshold variations between
numerical-library versions. The coupling check uses absolute tolerance 5e-8
and relative tolerance 1e-7; exact inner stopping counts are not required
across environments. Windows and Linux/Python 3.10 CI observed differences
below 2e-8 in the symmetric coupling; the loss check retains its absolute
tolerance of 1e-8. Original-versus-packaged comparisons in the same
environment also verified exact agreement before these cross-environment
tolerances were selected.

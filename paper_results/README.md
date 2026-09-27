# Archived paper results

The files listed in `manifest.json` are verbatim copies of the original
results. Their SHA-256 checksums are checked by the test suite. Fresh examples
write to `outputs/` and do not overwrite this archive.

- `snare_seq/`: all-solvers, five-seed sweep used for the camera-ready tables.
- `spheres/`: five-seed iPG sphere benchmarks.
- `projection_cost/`: 20-seed graph projection diagnostics.
- `inner_accuracy/`: fixed-instance four-exponent ablation.
- `graph_alignment/results.txt`: original rounded graph report; raw records
  were not present in the supplied directory.
- `graph_alignment/summary_rounded.csv`: derived from the original plotting
  notebook's embedded data for convenient plotting. It is a derived file,
  not a raw measurement and not included in the verbatim-copy manifest.

See `docs/reproducing.md` for the relationship between presets, tables,
metrics, and recorded timings. Timing values are historical measurements.

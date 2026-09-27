# SNARE-seq cache

The SNARE-seq examples download `snare_chromatin.txt`, `snare_rna.txt`, and
`SNAREseq_types.txt` from <https://rsinghlab.github.io/SCOT/data/> and verify
their SHA-256 checksums before use. These third-party data files are ignored
by Git and excluded from distributions. See `NOTICE.md` for attribution.

Use `--data-dir` to select a cache and `--offline` to prohibit network access.

"""Run documented paper settings or small checks of the same pipelines."""

import argparse
import importlib
import json
from importlib.resources import files
from pathlib import Path


def main(argv=None):
    presets = json.loads(files("gwipg.benchmarks").joinpath("paper_presets.json").read_text())
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("experiment", nargs="?", choices=[*presets, "all"])
    parser.add_argument("--list", action="store_true", help="List available paper examples.")
    parser.add_argument(
        "--quick", action="store_true", help="Small pipeline checks, not paper results."
    )
    parser.add_argument("--output-dir", type=Path, default=Path("outputs/paper"))
    parser.add_argument("--data-dir", type=Path, default=Path("data/snare_seq"))
    parser.add_argument("--offline", action="store_true", help="Require cached SNARE-seq inputs.")
    args = parser.parse_args(argv)
    if args.list:
        for name, preset in presets.items():
            print(f"{name:18} {preset['description']}")
        return
    if args.experiment is None:
        parser.error("choose an experiment or use --list")
    experiments = list(presets) if args.experiment == "all" else [args.experiment]
    for name in experiments:
        preset = presets[name]
        output = args.output_dir / ("quick" if args.quick else "full") / name
        command = list(preset["quick" if args.quick else "args"])
        if name == "inner-accuracy":
            command += ["--output", str(output / "exponent_sweep.csv")]
        else:
            command += ["--output-dir", str(output)]
        if name in {"snare", "inner-accuracy"}:
            command += ["--data-dir", str(args.data_dir)]
            if args.offline:
                command.append("--offline")
        print(f"\n{name}: {' '.join(command)}", flush=True)
        module = importlib.import_module(f"gwipg.benchmarks.{preset['module']}")
        module.main(command)


if __name__ == "__main__":
    main()

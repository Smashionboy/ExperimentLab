"""Run all six studies into a separate directory using the active interpreter."""

import argparse
import subprocess
import sys
from pathlib import Path


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument(
        "--quick",
        action="store_true",
        help="Smoke run, not published statistical results",
    )
    parser.add_argument("--output", type=Path, default=Path("local_data/reproduced"))
    parser.add_argument(
        "--studies",
        nargs="+",
        choices=[
            "core",
            "robustness",
            "sequential",
            "adjustment",
            "fieller",
            "ratio_sensitivity",
        ],
        default=[
            "core",
            "robustness",
            "sequential",
            "adjustment",
            "fieller",
            "ratio_sensitivity",
        ],
    )
    args = parser.parse_args()
    root = Path(__file__).resolve().parents[1]
    output = args.output.resolve()
    mode = "quick" if args.quick else "full"
    commands = {
        "core": [
            "experimentlab",
            "run",
            "--config",
            str(root / f"configs/{mode}.json"),
        ],
        "robustness": ["experimentlab.robustness"],
        "sequential": ["experimentlab.sequential_study"],
        "adjustment": [
            "experimentlab.adjustment_study",
            "--config",
            str(root / f"configs/adjustment_{mode}.json"),
        ],
        "fieller": ["experimentlab.fieller_study"],
        "ratio_sensitivity": ["experimentlab.ratio_sensitivity"],
    }
    if args.quick:
        commands["robustness"] += [
            "--replications",
            "30",
            "--bootstrap-replications",
            "10",
            "--resamples",
            "199",
        ]
        commands["sequential"] += ["--replications", "30", "--horizon", "100"]
        commands["fieller"] += ["--replications", "30"]
        commands["ratio_sensitivity"] += ["--replications", "30"]
    for study in args.studies:
        dest = output if study == "core" else output / study
        print(f"Running {study} ({mode}) -> {dest}", flush=True)
        subprocess.run(
            [sys.executable, "-m", *commands[study], "--output", str(dest)],
            cwd=root,
            check=True,
        )
    print(
        "Completed. Quick runs do not reproduce full-study rates."
        if args.quick
        else "Completed full studies."
    )


if __name__ == "__main__":
    main()

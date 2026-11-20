"""CSV entry points for ratio and cluster-level inference."""

import argparse
from pathlib import Path

import pandas as pd

from .clusters import cluster_welch
from .data import validate_frame
from .ratio import ratio_delta
from .reporting import dump_json


def main(argv=None):
    p = argparse.ArgumentParser(description="Ratio and cluster-randomized analysis")
    commands = p.add_subparsers(dest="kind", required=True)
    for name in ["ratio", "cluster"]:
        sub = commands.add_parser(name)
        sub.add_argument("csv")
        sub.add_argument("--alpha", type=float, default=0.05)
        sub.add_argument("--output", required=True)
        if name == "ratio":
            sub.add_argument("--numerator", required=True)
            sub.add_argument("--denominator", required=True)
        else:
            sub.add_argument("--metric", required=True)
    args = p.parse_args(argv)
    try:
        frame = pd.read_csv(
            args.csv, dtype={"user_id": "string", "cluster_id": "string"}
        )
        if args.kind == "ratio":
            a, b = validate_frame(frame, args.numerator)
            da, db = validate_frame(frame, args.denominator)
            result = ratio_delta(a, da, b, db, args.alpha)
        else:
            if "user_id" in frame and (
                frame.user_id.isna().any() or frame.user_id.duplicated().any()
            ):
                raise ValueError("user_id must be unique and nonmissing when provided")
            result = cluster_welch(frame, args.metric, args.alpha)
        dest = Path(args.output)
        dest.parent.mkdir(parents=True, exist_ok=True)
        dump_json(dest, result.to_dict())
        print(f"Saved: {dest}")
    except (ValueError, OSError, KeyError) as exc:
        raise SystemExit(str(exc)) from exc


if __name__ == "__main__":
    main()

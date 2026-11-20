"""Command-line interface; JSON output contains effects, uncertainty, and caveats."""

import argparse
from pathlib import Path

import pandas as pd

from .advanced_resampling import bca_mean
from .config import load_config
from .data import allocation_check, demo_data, validate_frame
from .inference import conversion, welch
from .planning import conversion_mde, conversion_sample_size, mean_sample_size
from .reporting import dump_json, write_report
from .resampling import bootstrap_mean, permutation_mean
from .studies import run_studies


def parser():
    p = argparse.ArgumentParser(
        description="ExperimentLab: two-arm independent-user experiments"
    )
    commands = p.add_subparsers(dest="command", required=True)
    run = commands.add_parser(
        "run", help="Run synthetic Monte Carlo studies and render figures"
    )
    run.add_argument("--config", default="configs/quick.json")
    run.add_argument("--output", default="reports/quick")
    demo = commands.add_parser(
        "demo", help="Generate explicitly synthetic example user data"
    )
    demo.add_argument("--output", default="local_data/demo.csv")
    demo.add_argument("--n", type=int, default=5000)
    demo.add_argument("--seed", type=int, default=42)
    analyze = commands.add_parser("analyze", help="Analyze one user-level CSV metric")
    analyze.add_argument("csv")
    analyze.add_argument("--metric", required=True)
    analyze.add_argument("--kind", choices=["binary", "continuous"], required=True)
    analyze.add_argument("--alpha", type=float, default=0.05)
    analyze.add_argument("--resamples", type=int, default=1999)
    analyze.add_argument("--seed", type=int, default=42)
    analyze.add_argument(
        "--permutation",
        action="store_true",
        help="Enable only if exchangeability is justified",
    )
    analyze.add_argument(
        "--bca", action="store_true", help="Add BCa interval for a continuous metric"
    )
    analyze.add_argument("--output", default="local_data/analysis.json")
    plan = commands.add_parser("plan", help="Sample sizes and positive conversion MDE")
    plan.add_argument("--output", default="local_data/planning.json")
    plan.add_argument("--baseline", type=float, default=0.10)
    plan.add_argument(
        "--uplift", type=float, default=0.02, help="Absolute probability difference"
    )
    plan.add_argument(
        "--effect", type=float, default=0.2, help="Mean difference in metric units"
    )
    plan.add_argument("--sigma", type=float, default=1.0)
    plan.add_argument("--power", type=float, default=0.8)
    plan.add_argument("--alpha", type=float, default=0.05)
    plan.add_argument("--n", type=int, default=5000, help="Per-arm sample size for MDE")
    return p


def main(argv=None):
    args = parser().parse_args(argv)
    try:
        output = Path(args.output)
        if args.command == "run":
            cfg = load_config(args.config)
            rates, cuped = run_studies(cfg)
            write_report(rates, cuped, cfg, output)
        elif args.command == "demo":
            frame = demo_data(args.n, args.seed)
            output.parent.mkdir(parents=True, exist_ok=True)
            frame.to_csv(output, index=False)
        elif args.command == "analyze":
            # Preserve string identifiers, including leading zeros.
            frame = pd.read_csv(args.csv, dtype={"user_id": "string"})
            a, b = validate_frame(frame, args.metric, args.kind == "binary")
            estimates = [
                (conversion if args.kind == "binary" else welch)(
                    a, b, args.alpha
                ).to_dict()
            ]
            if args.kind == "continuous":
                estimates.append(
                    bootstrap_mean(
                        a, b, alpha=args.alpha, resamples=args.resamples, seed=args.seed
                    ).to_dict()
                )
            if args.bca:
                if args.kind != "continuous":
                    raise ValueError("--bca is supported for continuous metrics only")
                estimates.append(
                    bca_mean(
                        a, b, alpha=args.alpha, resamples=args.resamples, seed=args.seed
                    ).to_dict()
                )
            if args.permutation:
                estimates.append(
                    permutation_mean(a, b, resamples=args.resamples, seed=args.seed)
                )
            result = {
                "metric": args.metric,
                "effect_direction": "treatment - control",
                "alpha": args.alpha,
                "srm_p_value_expected_50_50": allocation_check(a.size, b.size),
                "estimates": estimates,
                "assumptions": [
                    "independent users",
                    "pre-specified metric and fixed horizon",
                    "randomized assignment",
                ],
                "note": "CSV validity and SRM do not establish randomization. Input provenance is user supplied.",
            }
            output.parent.mkdir(parents=True, exist_ok=True)
            dump_json(output, result)
        else:
            output.parent.mkdir(parents=True, exist_ok=True)
            dump_json(
                output,
                {
                    "target_power": args.power,
                    "alpha": args.alpha,
                    "baseline": args.baseline,
                    "uplift": args.uplift,
                    "mean_effect": args.effect,
                    "sigma": args.sigma,
                    "mean_n_per_arm": mean_sample_size(
                        args.effect, args.sigma, args.power, args.alpha
                    ),
                    "conversion_n_per_arm": conversion_sample_size(
                        args.baseline, args.uplift, args.power, args.alpha
                    ),
                    "mde_n_per_arm": args.n,
                    "positive_conversion_mde": conversion_mde(
                        args.n, args.baseline, args.power, args.alpha
                    ),
                },
            )
    except (ValueError, OSError, KeyError) as exc:
        raise SystemExit(f"Error: {exc}") from exc
    print(f"Saved: {output}")


if __name__ == "__main__":
    main()

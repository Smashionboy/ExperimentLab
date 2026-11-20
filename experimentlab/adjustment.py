"""CSV command line for fixed-horizon randomized covariate adjustment."""

import argparse
from pathlib import Path

import pandas as pd

from .adjustment_data import balance_table, frame_arrays, generate_trial
from .crossfit import crossfit_aipw
from .data import allocation_check
from .inference import welch
from .regression import covariate_arrays, lin_adjusted
from .reporting import dump_json


def analyze(
    frame,
    metric,
    covariates,
    output,
    *,
    method="all",
    learner="ridge",
    propensity=0.5,
    folds=2,
    seed=42,
    alpha=0.05,
):
    """Write aggregate estimates and per-user OOF diagnostics when requested.

    All computation/validation precedes output writes. Files in output are
    replaced; choose a fresh directory for another analysis.
    """
    if method not in {"all", "lin", "aipw"}:
        raise ValueError("method must be all, lin or aipw")
    y, w, x = covariate_arrays(*frame_arrays(frame, metric, covariates))
    raw = welch(y[w == 0], y[w == 1], alpha)
    estimates = [raw.to_dict()]
    diagnostics, oof = {}, {}
    if method in {"all", "lin"}:
        result, info = lin_adjusted(y, w, x, alpha)
        estimates.append(result.to_dict())
        diagnostics[result.method] = info
    if method in {"all", "aipw"}:
        for kind in ["ridge", "forest"] if method == "all" else [learner]:
            result = crossfit_aipw(
                y,
                w,
                x,
                propensity=propensity,
                learner=kind,
                folds=folds,
                seed=seed,
                alpha=alpha,
            )
            name = result.estimate.method
            estimates.append(result.estimate.to_dict())
            diagnostics[name] = result.diagnostics
            oof[name] = pd.DataFrame(
                {
                    "user_id": frame.user_id.to_numpy(),
                    "fold": result.fold_id,
                    "prediction_control": result.predictions_control,
                    "prediction_treatment": result.predictions_treatment,
                    "influence_score": result.scores,
                }
            )
    payload = {
        "metric": metric,
        "covariates": covariates,
        "alpha": alpha,
        "known_assignment_probability": propensity,
        "design": "Independent-user randomized trial; fixed horizon; pretreatment features only",
        "srm": allocation_check(int((w == 0).sum()), int((w == 1).sum()), propensity),
        "estimates": estimates,
        "diagnostics": diagnostics,
        "interpretation": "Prespecify a primary method; do not select the smallest p-value. SRM is only a diagnostic.",
    }
    output = Path(output)
    output.mkdir(parents=True, exist_ok=True)
    dump_json(output / "summary.json", payload)
    balance_table(w, x, covariates).to_csv(output / "balance.csv", index=False)
    for name in ["aipw_ridge", "aipw_forest"]:
        if name not in oof:
            (output / f"{name}_oof.csv").unlink(missing_ok=True)
    for name, table in oof.items():
        table.to_csv(output / f"{name}_oof.csv", index=False)
    return payload


def main(argv=None):
    parser = argparse.ArgumentParser(description=__doc__)
    sub = parser.add_subparsers(dest="command", required=True)
    demo = sub.add_parser("demo", help="Generate a clearly synthetic randomized trial")
    demo.add_argument(
        "--n", type=int, default=800, help="Total users, not users per arm"
    )
    demo.add_argument("--effect", type=float, default=0.25)
    demo.add_argument(
        "--mechanism", choices=["linear", "nonlinear"], default="nonlinear"
    )
    demo.add_argument("--propensity", type=float, default=0.5)
    demo.add_argument("--seed", type=int, default=42)
    demo.add_argument("--output", type=Path, required=True)
    analyze_parser = sub.add_parser("analyze")
    analyze_parser.add_argument("csv", type=Path)
    analyze_parser.add_argument("--metric", required=True)
    analyze_parser.add_argument("--covariates", nargs="+", required=True)
    analyze_parser.add_argument(
        "--method", choices=["all", "lin", "aipw"], default="all"
    )
    analyze_parser.add_argument(
        "--learner", choices=["ridge", "forest"], default="ridge"
    )
    analyze_parser.add_argument(
        "--propensity",
        type=float,
        required=True,
        help="Known treatment probability from the randomization design",
    )
    analyze_parser.add_argument("--folds", type=int, default=2)
    analyze_parser.add_argument("--seed", type=int, default=42)
    analyze_parser.add_argument("--alpha", type=float, default=0.05)
    analyze_parser.add_argument("--output", type=Path, required=True)
    args = parser.parse_args(argv)
    try:
        if args.command == "demo":
            frame = generate_trial(
                args.n, args.effect, args.mechanism, args.seed, args.propensity
            )
            args.output.parent.mkdir(parents=True, exist_ok=True)
            frame.to_csv(args.output, index=False)
            dump_json(
                args.output.with_suffix(".metadata.json"),
                {
                    "data_origin": "synthetic",
                    "n_total": args.n,
                    "true_effect": args.effect,
                    "mechanism": args.mechanism,
                    "seed": args.seed,
                    "known_propensity": args.propensity,
                },
            )
        else:
            analyze(
                pd.read_csv(args.csv, dtype={"user_id": "string"}),
                args.metric,
                args.covariates,
                args.output,
                method=args.method,
                learner=args.learner,
                propensity=args.propensity,
                folds=args.folds,
                seed=args.seed,
                alpha=args.alpha,
            )
    except (ValueError, OSError) as exc:
        parser.exit(2, f"Error: {exc}\n")
    print(f"Saved {args.output}")


if __name__ == "__main__":
    main()

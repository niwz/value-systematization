from __future__ import annotations

import argparse
import json
from datetime import datetime
from pathlib import Path
import sys

PACKAGE_ROOT = Path(__file__).resolve().parents[2]
if str(PACKAGE_ROOT) not in sys.path:
    sys.path.insert(0, str(PACKAGE_ROOT))

from advice_reflection_platform.backend.artifacts import ArtifactStore
from advice_reflection_platform.backend.gateway import LiveModelGateway
from advice_reflection_platform.experiment_families import (
    FAMILY_SPECS,
    build_grid_jobs,
    condition_names_for_family,
)
from advice_reflection_platform.experiment_runner import (
    run_family_prior_probe,
    run_sampled_query,
)
from advice_reflection_platform.experiment_results import (
    summarize_sampled_tradeoff_grid,
    write_sampled_tradeoff_report,
)


def main() -> None:
    parser = argparse.ArgumentParser(description="Run the sampled multi-family tradeoff grid.")
    parser.add_argument("--model", type=str, default="claude-opus-4-6")
    parser.add_argument(
        "--families",
        type=str,
        default="admissions,performance_escalation,expense_reporting,ai_labor_displacement,defense_casualties",
    )
    parser.add_argument("--thinking-efforts", type=str, default="disabled,low,medium,high")
    parser.add_argument("--orders", type=str, default="AB,BA")
    parser.add_argument("--repeats", type=int, default=2)
    parser.add_argument("--request-timeout-seconds", type=float, default=120.0)
    parser.add_argument("--output-prefix", type=str, default="")
    args = parser.parse_args()

    family_keys = [item.strip() for item in args.families.split(",") if item.strip()]
    invalid_families = [family_key for family_key in family_keys if family_key not in FAMILY_SPECS]
    if invalid_families:
        raise ValueError(f"Unknown families: {invalid_families}")
    thinking_efforts = [item.strip().lower() for item in args.thinking_efforts.split(",") if item.strip()]
    invalid_efforts = [effort for effort in thinking_efforts if effort not in {"disabled", "low", "medium", "high"}]
    if invalid_efforts:
        raise ValueError(f"Unknown thinking efforts: {invalid_efforts}")
    orders = [item.strip().upper() for item in args.orders.split(",") if item.strip()]

    base_dir = Path(__file__).resolve().parents[1]
    stamp = datetime.now().strftime("%Y%m%d_%H%M%S")
    filename_prefix = args.output_prefix or f"sampled_tradeoff_grid_{stamp}"
    gateway = LiveModelGateway()
    store = ArtifactStore(base_dir)

    prior_artifacts: dict[tuple[str, str, str], dict[str, object]] = {}
    for family_key in family_keys:
        for thinking_effort in thinking_efforts:
            for condition in condition_names_for_family(family_key):
                if condition == "baseline":
                    continue
                prior_artifacts[(family_key, condition, thinking_effort)] = run_family_prior_probe(
                    gateway=gateway,
                    model_name=args.model,
                    family_key=family_key,
                    condition_name=condition,
                    thinking_effort=thinking_effort,
                    request_timeout_seconds=args.request_timeout_seconds,
                )

    records = []
    jobs = build_grid_jobs(
        family_keys=family_keys,
        thinking_efforts=thinking_efforts,
        orders=orders,
        repeats=args.repeats,
    )
    for job in jobs:
        prior_artifact = None
        if job["condition"] != "baseline":
            prior_artifact = prior_artifacts[(job["family_key"], job["condition"], job["thinking_effort"])]
        records.append(
            run_sampled_query(
                family_key=str(job["family_key"]),
                point_key=str(job["point_key"]),
                model_name=args.model,
                condition_name=str(job["condition"]),
                thinking_effort=str(job["thinking_effort"]),
                presentation_order=str(job["presentation_order"]),
                repeat_idx=int(job["repeat_idx"]),
                gateway=gateway,
                prior_artifact=prior_artifact,
                request_timeout_seconds=args.request_timeout_seconds,
            )
        )

    summary = summarize_sampled_tradeoff_grid(records)
    summary["artifacts"] = {
        f"{family_key}:{condition}:{thinking_effort}": artifact
        for (family_key, condition, thinking_effort), artifact in prior_artifacts.items()
    }
    summary["run_config"] = {
        "model": args.model,
        "families": family_keys,
        "thinking_efforts": thinking_efforts,
        "orders": orders,
        "repeats": args.repeats,
    }
    cross_family_csv_rows = [
        {key: value for key, value in row.items() if key != "curve_points"}
        for row in summary["cross_family_summary"]
    ]

    raw_path, flat_csv_path = store.write_records(records, filename_prefix)
    cross_family_csv_path = store.write_summary(cross_family_csv_rows, f"{filename_prefix}_cross_family.csv")
    point_csv_path = store.write_summary(summary["point_summary"], f"{filename_prefix}_point_summary.csv")
    aux_path = base_dir / "runs" / "raw" / f"{filename_prefix}_aux.json"
    aux_path.write_text(json.dumps(summary["artifacts"], indent=2) + "\n", encoding="utf-8")
    analysis_path = base_dir / "runs" / "summaries" / f"{filename_prefix}_analysis.json"
    analysis_path.write_text(json.dumps(summary, indent=2, sort_keys=True) + "\n", encoding="utf-8")
    report_path = write_sampled_tradeoff_report(
        summary,
        base_dir=base_dir,
        filename=f"{filename_prefix}.html",
        report_title=f"Sampled Tradeoff Grid: {args.model}",
    )

    print(f"raw_path={raw_path}")
    print(f"flat_csv_path={flat_csv_path}")
    print(f"cross_family_csv_path={cross_family_csv_path}")
    print(f"point_csv_path={point_csv_path}")
    print(f"aux_path={aux_path}")
    print(f"analysis_path={analysis_path}")
    print(f"report_path={report_path}")
    print(json.dumps(summary["run_config"], indent=2, sort_keys=True))


if __name__ == "__main__":
    main()

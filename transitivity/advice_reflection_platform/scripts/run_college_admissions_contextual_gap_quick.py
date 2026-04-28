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
from advice_reflection_platform.backend.college_admissions_contextual_gap_quick import (
    build_college_admissions_scenarios,
    run_family_reflection_probe,
    run_revealed_threshold_search,
    run_stated_level_probe,
    summarize_college_admissions_principle_gap,
    summary_to_json,
)
from advice_reflection_platform.backend.gateway import LiveModelGateway


def main() -> None:
    parser = argparse.ArgumentParser(description="Run the quick college admissions contextual-gap pilot.")
    parser.add_argument("--model", type=str, default="openai/gpt-5.4")
    parser.add_argument("--repeats", type=int, default=1)
    parser.add_argument("--surface-forms", type=str, default="p1")
    parser.add_argument("--orders", type=str, default="AB,BA")
    parser.add_argument(
        "--reflection-mode",
        choices=("general", "specific", "consistency", "error_tradeoff"),
        default="general",
    )
    parser.add_argument("--skip-baseline", action="store_true")
    parser.add_argument("--output-prefix", type=str, default="")
    args = parser.parse_args()

    base_dir = Path(__file__).resolve().parents[1]
    surface_forms = {item.strip() for item in args.surface_forms.split(",") if item.strip()}
    orders = {item.strip().upper() for item in args.orders.split(",") if item.strip()}

    scenarios = {
        scenario.scenario_id: scenario
        for scenario in build_college_admissions_scenarios()
        if scenario.surface_form in surface_forms
    }

    gateway = LiveModelGateway()
    reflection_artifact = run_family_reflection_probe(
        gateway=gateway,
        model_name=args.model,
        reflection_mode=args.reflection_mode,
    )

    revealed_records = []
    threshold_runs = []
    conditions: list[tuple[str, dict[str, object] | None]] = [("reflection", reflection_artifact)]
    if not args.skip_baseline:
        conditions.insert(0, ("baseline", None))

    for condition_name, artifact in conditions:
        for surface_form in sorted(surface_forms):
            for presentation_order in sorted(orders):
                for repeat_idx in range(1, args.repeats + 1):
                    run_records, threshold_run = run_revealed_threshold_search(
                        scenarios_by_id=scenarios,
                        model_name=args.model,
                        condition_name=condition_name,
                        surface_form=surface_form,
                        presentation_order=presentation_order,
                        repeat_idx=repeat_idx,
                        gateway=gateway,
                        reflection_artifact=artifact,
                    )
                    revealed_records.extend(run_records)
                    threshold_runs.append(threshold_run)

    stated_results = [
        run_stated_level_probe(
            gateway=gateway,
            model_name=args.model,
            condition_name="reflection",
            reflection_artifact=reflection_artifact,
        ),
    ]
    if not args.skip_baseline:
        stated_results.insert(
            0,
            run_stated_level_probe(
                gateway=gateway,
                model_name=args.model,
                condition_name="baseline",
            ),
        )

    summary = summarize_college_admissions_principle_gap(
        revealed_records=revealed_records,
        threshold_runs=threshold_runs,
        stated_results=stated_results,
    )

    stamp = datetime.now().strftime("%Y%m%d_%H%M%S")
    filename_prefix = args.output_prefix or f"college_admissions_contextual_gap_quick_{stamp}"
    store = ArtifactStore(base_dir)
    raw_path, summary_csv_path = store.write_records(revealed_records, filename_prefix)
    direct_path = base_dir / "runs" / "raw" / f"{filename_prefix}_stated_policy.json"
    direct_path.write_text(json.dumps(stated_results, indent=2) + "\n", encoding="utf-8")
    analysis_path = base_dir / "runs" / "summaries" / f"{filename_prefix}_analysis.json"
    analysis_path.write_text(summary_to_json(summary), encoding="utf-8")

    print(f"raw_path={raw_path}")
    print(f"summary_csv_path={summary_csv_path}")
    print(f"direct_path={direct_path}")
    print(f"analysis_path={analysis_path}")
    print(summary_to_json(summary))


if __name__ == "__main__":
    main()

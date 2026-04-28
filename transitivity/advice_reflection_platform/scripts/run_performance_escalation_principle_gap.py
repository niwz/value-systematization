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
from advice_reflection_platform.backend.gateway import HeuristicDemoGateway, LiveModelGateway
from advice_reflection_platform.backend.orchestrator import load_batch_jobs, run_batch
from advice_reflection_platform.backend.performance_escalation_principle_gap import (
    run_stated_policy_probe,
    summarize_performance_escalation_principle_gap,
    summary_to_json,
)
from advice_reflection_platform.backend.performance_escalation_threshold_pilot import (
    build_performance_escalation_jobs,
)
from advice_reflection_platform.backend.scenario_registry import ScenarioRegistry


def _build_gateway(name: str):
    if name == "demo":
        return HeuristicDemoGateway()
    if name == "live":
        return LiveModelGateway()
    raise ValueError(f"Unsupported gateway: {name}")


def _filter_jobs(
    jobs: list[dict[str, object]],
    *,
    surface_forms: set[str] | None,
    orders: set[str] | None,
) -> list[dict[str, object]]:
    filtered = jobs
    if surface_forms:
        filtered = [job for job in filtered if str(job.get("surface_form") or job["scenario_id"]).split("__")[-1] in surface_forms]
    if orders:
        filtered = [job for job in filtered if str(job.get("presentation_order", "AB")).upper() in orders]
    return filtered


def main() -> None:
    parser = argparse.ArgumentParser(description="Run the performance-escalation principle-gap pilot.")
    parser.add_argument("--gateway", choices=("demo", "live"), default="demo")
    parser.add_argument("--model", type=str, default="claude-haiku-4-5-20251001")
    parser.add_argument("--repeats", type=int, default=1)
    parser.add_argument(
        "--scenarios",
        type=str,
        default="advice_reflection_platform/data/scenarios/performance_escalation_threshold_pilot.json",
    )
    parser.add_argument("--jobs", type=str, default="")
    parser.add_argument("--surface-forms", type=str, default="p1")
    parser.add_argument("--orders", type=str, default="AB")
    parser.add_argument("--output-prefix", type=str, default="")
    args = parser.parse_args()

    base_dir = Path(__file__).resolve().parents[1]
    scenario_path = Path(args.scenarios)
    if not scenario_path.is_absolute():
        scenario_path = Path.cwd() / scenario_path

    registry = ScenarioRegistry(base_dir / "data" / "scenarios")
    scenarios = {scenario.scenario_id: scenario for scenario in registry.load_path(scenario_path)}
    jobs = (
        load_batch_jobs(args.jobs)
        if args.jobs
        else build_performance_escalation_jobs(model_name=args.model, repeats=args.repeats)
    )
    surface_forms = {item.strip() for item in args.surface_forms.split(",") if item.strip()}
    orders = {item.strip().upper() for item in args.orders.split(",") if item.strip()}
    jobs = _filter_jobs(jobs, surface_forms=surface_forms, orders=orders)

    gateway = _build_gateway(args.gateway)
    revealed_records = run_batch(scenarios_by_id=scenarios, jobs=jobs, gateway=gateway, default_model_name=args.model)
    stated_results = [
        run_stated_policy_probe(gateway=gateway, model_name=args.model, condition_name="baseline"),
        run_stated_policy_probe(gateway=gateway, model_name=args.model, condition_name="reflection"),
    ]
    summary = summarize_performance_escalation_principle_gap(
        revealed_records=revealed_records,
        stated_results=stated_results,
    )

    stamp = datetime.now().strftime("%Y%m%d_%H%M%S")
    filename_prefix = args.output_prefix or f"performance_escalation_principle_gap_{args.gateway}_{stamp}"
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

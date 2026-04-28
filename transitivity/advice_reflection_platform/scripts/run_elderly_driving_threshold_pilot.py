from __future__ import annotations

import argparse
from datetime import datetime
from pathlib import Path
import sys

PACKAGE_ROOT = Path(__file__).resolve().parents[2]
if str(PACKAGE_ROOT) not in sys.path:
    sys.path.insert(0, str(PACKAGE_ROOT))

from advice_reflection_platform.backend.artifacts import ArtifactStore
from advice_reflection_platform.backend.elderly_driving_threshold_pilot import (
    build_elderly_driving_jobs,
    summarize_elderly_driving_scan,
    summary_to_json,
)
from advice_reflection_platform.backend.orchestrator import load_batch_jobs, run_batch
from advice_reflection_platform.backend.scenario_registry import ScenarioRegistry


def _build_gateway(name: str):
    if name == "demo":
        from advice_reflection_platform.backend.gateway import HeuristicDemoGateway

        return HeuristicDemoGateway()
    if name == "live":
        from advice_reflection_platform.backend.gateway import LiveModelGateway

        return LiveModelGateway()
    raise ValueError(f"Unsupported gateway: {name}")


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument("--gateway", choices=("demo", "live"), default="demo")
    parser.add_argument("--model", type=str, default="claude-haiku-4-5-20251001")
    parser.add_argument("--repeats", type=int, default=1)
    parser.add_argument(
        "--scenarios",
        type=str,
        default="advice_reflection_platform/data/scenarios/elderly_driving_threshold_pilot.json",
    )
    parser.add_argument("--jobs", type=str, default="")
    parser.add_argument("--output-prefix", type=str, default="")
    args = parser.parse_args()

    base_dir = Path(__file__).resolve().parents[1]
    scenario_path = Path(args.scenarios)
    if not scenario_path.is_absolute():
        scenario_path = Path.cwd() / scenario_path

    registry = ScenarioRegistry(base_dir / "data" / "scenarios")
    scenarios = {scenario.scenario_id: scenario for scenario in registry.load_path(scenario_path)}
    jobs = load_batch_jobs(args.jobs) if args.jobs else build_elderly_driving_jobs(model_name=args.model, repeats=args.repeats)
    gateway = _build_gateway(args.gateway)

    records = run_batch(scenarios_by_id=scenarios, jobs=jobs, gateway=gateway, default_model_name=args.model)
    summary = summarize_elderly_driving_scan(records)

    stamp = datetime.now().strftime("%Y%m%d_%H%M%S")
    filename_prefix = args.output_prefix or f"elderly_driving_threshold_{args.gateway}_{stamp}"
    store = ArtifactStore(base_dir)
    raw_path, summary_csv_path = store.write_records(records, filename_prefix)
    analysis_path = base_dir / "runs" / "summaries" / f"{filename_prefix}_analysis.json"
    analysis_path.write_text(summary_to_json(summary), encoding="utf-8")

    print(f"raw_path={raw_path}")
    print(f"summary_csv_path={summary_csv_path}")
    print(f"analysis_path={analysis_path}")
    print(summary_to_json(summary))


if __name__ == "__main__":
    main()

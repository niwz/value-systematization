from __future__ import annotations

import argparse
import json
from datetime import datetime
from pathlib import Path
import sys

PACKAGE_ROOT = Path(__file__).resolve().parents[2]
if str(PACKAGE_ROOT) not in sys.path:
    sys.path.insert(0, str(PACKAGE_ROOT))

from advice_reflection_platform.backend.analysis import summarize_family_pilot
from advice_reflection_platform.backend.artifacts import ArtifactStore
from advice_reflection_platform.backend.family_pilot import load_family_pilot_jobs, run_family_pilot_batch
from advice_reflection_platform.backend.gateway import HeuristicDemoGateway, LiveModelGateway
from advice_reflection_platform.backend.scenario_registry import ScenarioRegistry


def main() -> None:
    parser = argparse.ArgumentParser(description="Run the one-family rule-reflection pilot.")
    parser.add_argument("--gateway", choices=["demo", "live"], default="demo")
    parser.add_argument("--scenarios", type=str, default="advice_reflection_platform/data/scenarios/mentee_family_pilot.json")
    parser.add_argument("--jobs", type=str, default="advice_reflection_platform/data/uploads/mentee_family_pilot_jobs.json")
    parser.add_argument("--output-prefix", type=str, default=None)
    args = parser.parse_args()

    base_dir = Path(__file__).resolve().parents[1]
    scenario_path = Path(args.scenarios)
    if not scenario_path.is_absolute():
        scenario_path = Path.cwd() / scenario_path
    jobs_path = Path(args.jobs)
    if not jobs_path.is_absolute():
        jobs_path = Path.cwd() / jobs_path

    registry = ScenarioRegistry(base_dir / "data" / "scenarios")
    scenarios_by_id = {scenario.scenario_id: scenario for scenario in registry.load_path(scenario_path)}
    jobs = load_family_pilot_jobs(jobs_path)

    gateway = HeuristicDemoGateway() if args.gateway == "demo" else LiveModelGateway()
    records = run_family_pilot_batch(scenarios_by_id=scenarios_by_id, jobs=jobs, gateway=gateway)
    summary = summarize_family_pilot(records)

    store = ArtifactStore(base_dir)
    stamp = datetime.now().strftime("%Y%m%d_%H%M%S")
    filename_prefix = args.output_prefix or f"mentee_family_pilot_{args.gateway}_{stamp}"
    raw_path, summary_path = store.write_records(records, filename_prefix)
    analysis_path = base_dir / "runs" / "summaries" / f"{filename_prefix}_analysis.json"
    analysis_path.write_text(json.dumps(summary, indent=2) + "\n", encoding="utf-8")

    print(json.dumps({"raw_path": str(raw_path), "summary_path": str(summary_path), "analysis_path": str(analysis_path)}, indent=2))


if __name__ == "__main__":
    main()

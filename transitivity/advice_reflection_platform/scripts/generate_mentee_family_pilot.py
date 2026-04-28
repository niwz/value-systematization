from __future__ import annotations

import json
from pathlib import Path
import sys

PACKAGE_ROOT = Path(__file__).resolve().parents[2]
if str(PACKAGE_ROOT) not in sys.path:
    sys.path.insert(0, str(PACKAGE_ROOT))

from advice_reflection_platform.backend.mentee_family_pilot import (
    build_mentee_family_pilot_jobs,
    build_mentee_family_pilot_scenarios,
    pilot_manifest,
)
from advice_reflection_platform.backend.scenario_registry import ScenarioRegistry


def main() -> None:
    base_dir = Path(__file__).resolve().parents[1]
    registry = ScenarioRegistry(base_dir / "data" / "scenarios")
    scenarios = build_mentee_family_pilot_scenarios()
    scenario_path = registry.save(scenarios, "mentee_family_pilot.json")

    jobs = [job.to_dict() for job in build_mentee_family_pilot_jobs()]
    jobs_path = base_dir / "data" / "uploads" / "mentee_family_pilot_jobs.json"
    jobs_path.write_text(json.dumps(jobs, indent=2) + "\n", encoding="utf-8")

    manifest_path = base_dir / "data" / "uploads" / "mentee_family_pilot_manifest.json"
    manifest_path.write_text(json.dumps(pilot_manifest(), indent=2) + "\n", encoding="utf-8")

    print(json.dumps({"scenario_path": str(scenario_path), "jobs_path": str(jobs_path), "manifest_path": str(manifest_path)}, indent=2))


if __name__ == "__main__":
    main()

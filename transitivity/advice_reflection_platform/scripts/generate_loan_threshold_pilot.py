from __future__ import annotations

import json
from pathlib import Path
import sys

PACKAGE_ROOT = Path(__file__).resolve().parents[2]
if str(PACKAGE_ROOT) not in sys.path:
    sys.path.insert(0, str(PACKAGE_ROOT))

from advice_reflection_platform.backend.loan_threshold_pilot import (
    AMOUNT_LEVELS_GBP,
    FAMILY_ID,
    PARAPHRASE_TEMPLATES,
    build_loan_threshold_jobs,
    build_loan_threshold_scenarios,
)
from advice_reflection_platform.backend.scenario_registry import ScenarioRegistry


def main() -> None:
    base_dir = Path(__file__).resolve().parents[1]
    registry = ScenarioRegistry(base_dir / "data" / "scenarios")
    scenarios = build_loan_threshold_scenarios()
    scenario_path = registry.save(scenarios, "loan_threshold_pilot.json")

    jobs_path = base_dir / "data" / "uploads" / "loan_threshold_pilot_jobs_haiku.json"
    jobs_path.write_text(
        json.dumps(build_loan_threshold_jobs(model_name="claude-haiku-4-5-20251001", repeats=1), indent=2),
        encoding="utf-8",
    )

    manifest = {
        "family_id": FAMILY_ID,
        "axis_name": "debt_amount_gbp",
        "axis_values": AMOUNT_LEVELS_GBP,
        "surface_forms": sorted(PARAPHRASE_TEMPLATES),
        "scenarios": len(scenarios),
        "jobs": len(build_loan_threshold_jobs(model_name="claude-haiku-4-5-20251001", repeats=1)),
        "scenario_path": str(scenario_path),
        "jobs_path": str(jobs_path),
    }
    manifest_path = base_dir / "data" / "uploads" / "loan_threshold_pilot_manifest.json"
    manifest_path.write_text(json.dumps(manifest, indent=2), encoding="utf-8")
    print(json.dumps(manifest, indent=2))


if __name__ == "__main__":
    main()

from __future__ import annotations

from pathlib import Path
import sys

PACKAGE_ROOT = Path(__file__).resolve().parents[2]
if str(PACKAGE_ROOT) not in sys.path:
    sys.path.insert(0, str(PACKAGE_ROOT))

from advice_reflection_platform.backend.scenario_factory import generate_scenarios_from_templates
from advice_reflection_platform.backend.scenario_registry import ScenarioRegistry


def main() -> None:
    base_dir = Path(__file__).resolve().parents[1]
    template_path = base_dir / "data" / "families" / "sample_family_templates.json"
    registry = ScenarioRegistry(base_dir / "data" / "scenarios")
    scenarios = generate_scenarios_from_templates(template_path)
    output_path = registry.save(scenarios, "sample_scenarios.json")
    print(output_path)


if __name__ == "__main__":
    main()

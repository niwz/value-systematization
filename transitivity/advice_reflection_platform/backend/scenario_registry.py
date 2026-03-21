from __future__ import annotations

import json
from pathlib import Path

from .schemas import ScenarioRecord


class ScenarioRegistry:
    def __init__(self, scenarios_dir: str | Path) -> None:
        self.scenarios_dir = Path(scenarios_dir)
        self.scenarios_dir.mkdir(parents=True, exist_ok=True)

    def list_paths(self) -> list[Path]:
        return sorted(self.scenarios_dir.glob("*.json"))

    def load_all(self) -> list[ScenarioRecord]:
        scenarios: list[ScenarioRecord] = []
        for path in self.list_paths():
            scenarios.extend(self.load_path(path))
        return scenarios

    def load_path(self, path: str | Path) -> list[ScenarioRecord]:
        with open(path, encoding="utf-8") as handle:
            payload = json.load(handle)
        items = payload if isinstance(payload, list) else [payload]
        return [ScenarioRecord.from_dict(item) for item in items]

    def get(self, scenario_id: str) -> ScenarioRecord:
        for scenario in self.load_all():
            if scenario.scenario_id == scenario_id:
                return scenario
        raise KeyError(f"Scenario not found: {scenario_id}")

    def save(self, scenarios: list[ScenarioRecord], filename: str) -> Path:
        path = self.scenarios_dir / filename
        with open(path, "w", encoding="utf-8") as handle:
            json.dump([item.to_dict() for item in scenarios], handle, indent=2, ensure_ascii=True)
            handle.write("\n")
        return path


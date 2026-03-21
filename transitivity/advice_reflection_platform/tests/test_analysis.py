from __future__ import annotations

import unittest

from advice_reflection_platform.backend.analysis import summarize_runs
from advice_reflection_platform.backend.gateway import ReplayGateway
from advice_reflection_platform.backend.orchestrator import run_batch
from advice_reflection_platform.backend.scenario_registry import ScenarioRegistry


class AnalysisTests(unittest.TestCase):
    def test_reflection_change_summary(self) -> None:
        registry = ScenarioRegistry("advice_reflection_platform/data/scenarios")
        scenarios = {scenario.scenario_id: scenario for scenario in registry.load_all()}
        jobs = [
            {"scenario_id": "manager_feedback_growth__direct_paraphrase", "condition": "baseline", "presentation_order": "AB"},
            {"scenario_id": "manager_feedback_growth__direct_paraphrase", "condition": "reflection", "presentation_order": "AB"},
        ]
        gateway = ReplayGateway(
            [
                '{"choice": "A", "reason": "Baseline."}',
                "Reflect.",
                '{"choice": "B", "reason": "Reflection."}',
            ]
        )
        records = run_batch(scenarios_by_id=scenarios, jobs=jobs, gateway=gateway)
        summary = summarize_runs(records)
        self.assertEqual(len(summary["reflection_summary"]), 1)
        self.assertTrue(summary["reflection_summary"][0]["reflection_changed"])


if __name__ == "__main__":
    unittest.main()

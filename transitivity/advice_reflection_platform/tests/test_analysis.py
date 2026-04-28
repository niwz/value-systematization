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

    def test_batch_thinking_flag_is_parsed_and_recorded(self) -> None:
        registry = ScenarioRegistry("advice_reflection_platform/data/scenarios")
        scenarios = {scenario.scenario_id: scenario for scenario in registry.load_all()}
        jobs = [
            {
                "scenario_id": "manager_feedback_growth__direct_paraphrase",
                "condition": "baseline",
                "presentation_order": "AB",
                "thinking": "true",
                "thinking_budget_tokens": "12000",
            }
        ]
        gateway = ReplayGateway(['{"choice": "A", "reason": "Baseline."}'])
        records = run_batch(scenarios_by_id=scenarios, jobs=jobs, gateway=gateway)
        self.assertEqual(len(records), 1)
        self.assertTrue(records[0].thinking)
        self.assertEqual(records[0].thinking_budget_tokens, 12000)

    def test_batch_thinking_effort_is_parsed_and_recorded(self) -> None:
        registry = ScenarioRegistry("advice_reflection_platform/data/scenarios")
        scenarios = {scenario.scenario_id: scenario for scenario in registry.load_all()}
        jobs = [
            {
                "scenario_id": "manager_feedback_growth__direct_paraphrase",
                "condition": "baseline",
                "presentation_order": "AB",
                "thinking_effort": "medium",
            }
        ]
        gateway = ReplayGateway(['{"choice": "A", "reason": "Baseline."}'])
        records = run_batch(scenarios_by_id=scenarios, jobs=jobs, gateway=gateway)
        self.assertEqual(len(records), 1)
        self.assertEqual(records[0].thinking_effort, "medium")

    def test_open_advice_batch_records_unclear_and_parser_fields(self) -> None:
        registry = ScenarioRegistry("advice_reflection_platform/data/scenarios")
        scenarios = {scenario.scenario_id: scenario for scenario in registry.load_all()}
        jobs = [
            {
                "scenario_id": "manager_feedback_growth__direct_paraphrase",
                "condition": "baseline",
                "run_mode": "open_advice",
                "parser_model_name": "claude-opus-4-6",
            }
        ]
        gateway = ReplayGateway(
            [
                "Advice text.",
                "Bottom-line recommendation.",
                '{"fit": "AMBIGUOUS", "primary_action_summary": "No privileged action.", "secondary_fit": "A", "mixed_or_conditional": true, "why_not_a_clean_fit": "Recommendation does not privilege one action."}',
            ]
        )
        records = run_batch(scenarios_by_id=scenarios, jobs=jobs, gateway=gateway)
        summary = summarize_runs(records)
        self.assertEqual(records[0].run_mode, "open_advice")
        self.assertEqual(records[0].parsed.final_choice, "AMBIGUOUS")
        self.assertEqual(summary["scenario_summary"][0]["ambiguous_rate"], 1.0)
        self.assertEqual(summary["scenario_summary"][0]["mixed_or_conditional_rate"], 1.0)


if __name__ == "__main__":
    unittest.main()

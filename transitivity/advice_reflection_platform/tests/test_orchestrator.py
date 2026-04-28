from __future__ import annotations

import unittest

from advice_reflection_platform.backend.gateway import ReplayGateway
from advice_reflection_platform.backend.orchestrator import run_single_scenario
from advice_reflection_platform.backend.schemas import AdviceOption, ScenarioRecord


class OrchestratorTests(unittest.TestCase):
    def test_single_run_tracks_reflection_change(self) -> None:
        scenario = ScenarioRecord(
            scenario_id="case_1",
            family_id="family_x",
            request_text="A manager asks what advice to give.",
            option_a=AdviceOption(title="Candor", text="Be blunt."),
            option_b=AdviceOption(title="Empathy", text="Be gentle."),
            domain="manager advising an employee",
            latent_dimensions={"candor_vs_empathy": "boundary"},
            paraphrase_group="group_1",
        )
        gateway = ReplayGateway(
            [
                '{"choice": "A", "reason": "Baseline."}',
                "Reflect on candor versus empathy.",
                '{"choice": "B", "reason": "Reflection changed the balance."}',
            ]
        )
        bundle = run_single_scenario(scenario=scenario, model_name="demo-model", gateway=gateway)
        self.assertEqual(bundle.baseline.canonical_choice, "A")
        self.assertEqual(bundle.reflection.canonical_choice, "B")
        self.assertTrue(bundle.changed)
        self.assertEqual(bundle.reflection.reflection_text, "Reflect on candor versus empathy.")

    def test_open_advice_run_stores_advice_recommendation_and_parser(self) -> None:
        scenario = ScenarioRecord(
            scenario_id="case_2",
            family_id="family_y",
            request_text="A friend asks for advice.",
            option_a=AdviceOption(title="Stay quiet", text="Do not intervene.", action_signature="Do not intervene directly."),
            option_b=AdviceOption(title="Speak up", text="Intervene directly.", action_signature="Intervene directly."),
            domain="friendship",
            latent_dimensions={},
            paraphrase_group="group_2",
        )
        gateway = ReplayGateway(
            [
                "Here is my practical advice in plain text.",
                "My bottom-line recommendation is to speak up.",
                '{"fit": "B", "primary_action_summary": "Recommend direct intervention.", "secondary_fit": null, "mixed_or_conditional": false, "why_not_a_clean_fit": ""}',
                "Reflection text for the case.",
                "Here is my reflected advice in plain text.",
                "My bottom-line recommendation is to stay quiet.",
                '{"fit": "A", "primary_action_summary": "Recommend staying quiet.", "secondary_fit": "B", "mixed_or_conditional": true, "why_not_a_clean_fit": "Conditional on more facts."}',
            ]
        )
        bundle = run_single_scenario(
            scenario=scenario,
            model_name="demo-model",
            gateway=gateway,
            run_mode="open_advice",
            parser_model_name="claude-opus-4-6",
        )
        self.assertEqual(bundle.baseline.run_mode, "open_advice")
        self.assertEqual(bundle.baseline.advice_text, "Here is my practical advice in plain text.")
        self.assertEqual(bundle.baseline.recommendation_text, "My bottom-line recommendation is to speak up.")
        self.assertEqual(bundle.baseline.parser_model_name, "claude-opus-4-6")
        self.assertEqual(bundle.baseline.parsed.final_choice, "B")
        self.assertEqual(bundle.baseline.parser_primary_action_summary, "Recommend direct intervention.")
        self.assertEqual(bundle.reflection.reflection_text, "Reflection text for the case.")
        self.assertEqual(bundle.reflection.parsed.final_choice, "A")
        self.assertTrue(bundle.reflection.mixed_or_conditional)
        self.assertEqual(bundle.reflection.parser_secondary_fit, "B")


if __name__ == "__main__":
    unittest.main()

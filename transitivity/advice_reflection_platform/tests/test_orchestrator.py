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


if __name__ == "__main__":
    unittest.main()


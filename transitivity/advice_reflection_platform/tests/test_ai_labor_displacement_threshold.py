from __future__ import annotations

import unittest

from advice_reflection_platform.backend.ai_labor_displacement_threshold import (
    ThresholdSearchFailure,
    build_savings_scenario,
    bisect_labor_threshold,
    render_family_constitution_prompt,
)


class AILaborDisplacementThresholdTests(unittest.TestCase):
    def test_build_savings_scenario_embeds_numeric_axis(self) -> None:
        scenario = build_savings_scenario(annual_net_savings_per_role=75_000, surface_form="p1")
        self.assertIn("$75,000", scenario.request_text)
        self.assertEqual(scenario.latent_values["annual_net_savings_per_role"], "75000")
        self.assertEqual(scenario.latent_values["total_annual_net_savings"], str(75_000 * 15))

    def test_constitution_prompt_requests_rules_not_threshold(self) -> None:
        prompt = render_family_constitution_prompt(min_savings=0, max_savings=250_000)
        self.assertIn("Write the shortest practical constitution", prompt)
        self.assertIn("Do not name a numeric threshold", prompt)

    def test_bisect_labor_threshold_returns_interval(self) -> None:
        def query_choice(savings: int) -> str:
            return "plan_a" if savings < 90_000 else "plan_b"

        payload, queried = bisect_labor_threshold(
            min_savings=0,
            max_savings=250_000,
            tolerance=10_000,
            step=5_000,
            query_choice=query_choice,
        )
        self.assertEqual(payload["position"], "within_range")
        self.assertLessEqual(
            payload["upper_savings_per_role"] - payload["lower_savings_per_role"],
            10_000,
        )
        self.assertGreaterEqual(len(queried), 2)

    def test_threshold_search_failure_carries_checkpoint_payload(self) -> None:
        failure = ThresholdSearchFailure(
            message="boom",
            partial_records=[],
            failure_payload={"annual_net_savings_per_role": 125_000, "raw_response": ""},
        )
        self.assertEqual(failure.failure_payload["annual_net_savings_per_role"], 125_000)
        self.assertEqual(failure.failure_payload["raw_response"], "")


if __name__ == "__main__":
    unittest.main()

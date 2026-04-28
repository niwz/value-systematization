from __future__ import annotations

import unittest

from advice_reflection_platform.backend.gateway import ReplayGateway
from advice_reflection_platform.backend.selection_contextual_gap_probe import (
    bisect_selection_gap_threshold,
    build_selection_scenario,
    render_family_constitution_prompt,
    run_constitution_prediction_query,
)


class SelectionContextualGapProbeTests(unittest.TestCase):
    def test_build_selection_scenario_fellowship_uses_family_specific_wording(self) -> None:
        scenario = build_selection_scenario(
            family_key="fellowship",
            score_gap_points=350,
            profile_variant="canonical",
        )
        self.assertIn("public-interest summer fellowship", scenario.request_text)
        self.assertIn("Candidate B's policy-analysis assessment score is 1150", scenario.request_text)
        self.assertEqual(scenario.latent_values["disadvantaged_candidate"], "B")

    def test_build_selection_scenario_hiring_swapped_flips_letters_only(self) -> None:
        scenario = build_selection_scenario(
            family_key="hiring",
            score_gap_points=550,
            profile_variant="swapped",
        )
        self.assertIn("Candidate A grew up in a low-income household", scenario.request_text)
        self.assertIn("Candidate A's structured skills assessment score is 950", scenario.request_text)
        self.assertIn("Candidate B's structured skills assessment score is 1500", scenario.request_text)
        self.assertEqual(scenario.latent_values["disadvantaged_candidate"], "A")

    def test_constitution_prompt_mentions_selected_family(self) -> None:
        prompt = render_family_constitution_prompt(family_key="hiring", min_gap=350, max_gap=750)
        self.assertIn("entry-level analyst hiring round", prompt)
        self.assertIn("Do not name a numeric threshold", prompt)

    def test_transfer_prediction_keeps_canonical_letter_mapping(self) -> None:
        gateway = ReplayGateway(['{"choice":"A","reason":"Apply the constitution."}'])
        row = run_constitution_prediction_query(
            family_key="fellowship",
            score_gap_points=550,
            model_name="demo-model",
            presentation_order="BA",
            repeat_idx=1,
            profile_variant="canonical",
            gateway=gateway,
            constitution_artifact={"prompt": "p", "prior_text": "- Rule"},
        )
        self.assertEqual(row["prediction_choice"], "A")
        self.assertEqual(row["prediction_canonical_choice"], "A")

    def test_bisect_selection_gap_threshold_returns_interval(self) -> None:
        def query_choice(gap: int) -> str:
            return "disadvantaged" if gap < 480 else "advantaged"

        payload, queried_gaps = bisect_selection_gap_threshold(
            min_gap=0,
            max_gap=1100,
            tolerance=50,
            step=10,
            query_choice=query_choice,
        )
        self.assertEqual(payload["position"], "within_range")
        self.assertLessEqual(payload["upper_gap_points"] - payload["lower_gap_points"], 50)
        self.assertGreaterEqual(len(queried_gaps), 2)


if __name__ == "__main__":
    unittest.main()

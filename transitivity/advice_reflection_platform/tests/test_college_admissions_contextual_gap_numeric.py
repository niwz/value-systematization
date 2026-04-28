from __future__ import annotations

import unittest

from advice_reflection_platform.backend.college_admissions_contextual_gap_numeric import (
    DEFAULT_GAP_STEP,
    DEFAULT_MAX_GAP,
    DEFAULT_MIN_GAP,
    DEFAULT_TOLERANCE,
    PROFILE_VARIANTS,
    _contextual_score_for_gap,
    bisect_numeric_gap_threshold,
    build_gap_scenario,
    parse_stated_gap_response,
    render_family_constitution_prompt,
    run_constitution_prediction_query,
)
from advice_reflection_platform.backend.gateway import ReplayGateway


class CollegeAdmissionsContextualGapNumericTests(unittest.TestCase):
    def test_contextual_score_tracks_gap(self) -> None:
        self.assertEqual(_contextual_score_for_gap(0), 1500)
        self.assertEqual(_contextual_score_for_gap(180), 1320)

    def test_build_gap_scenario_embeds_numeric_gap(self) -> None:
        scenario = build_gap_scenario(sat_gap_points=140, surface_form="p1", profile_variant="canonical")
        self.assertEqual(scenario.latent_values["sat_gap_points"], "140")
        self.assertIn("140 points", scenario.request_text)
        self.assertIn("educational and family circumstances", scenario.request_text)
        self.assertNotIn("contextual signal", scenario.request_text)
        self.assertEqual(scenario.latent_values["disadvantaged_applicant"], "B")

    def test_build_gap_scenario_swapped_variant_flips_letters_only(self) -> None:
        scenario = build_gap_scenario(sat_gap_points=140, surface_form="p2", profile_variant="swapped")
        self.assertIn("Applicant A attended an under-resourced public high school", scenario.request_text)
        self.assertIn("Applicant B attended a well-resourced suburban high school", scenario.request_text)
        self.assertIn("Applicant A's SAT score is 1360", scenario.request_text)
        self.assertIn("Applicant B's SAT score is 1500", scenario.request_text)
        self.assertEqual(scenario.latent_values["disadvantaged_applicant"], "A")
        self.assertIn("swapped", PROFILE_VARIANTS)

    def test_parse_stated_gap_response_json(self) -> None:
        parsed = parse_stated_gap_response(
            '{"threshold_sat_gap_points": 180, "position": "within_range", "reason": "That is the limit."}'
        )
        self.assertEqual(parsed["threshold_sat_gap_points"], 180)
        self.assertEqual(parsed["position"], "within_range")
        self.assertEqual(parsed["parse_provenance"], "json_last_valid")

    def test_bisect_numeric_gap_threshold_returns_interval(self) -> None:
        def query_choice(gap: int) -> str:
            return "disadvantaged" if gap < 170 else "advantaged"

        payload, queried_gaps = bisect_numeric_gap_threshold(
            min_gap=DEFAULT_MIN_GAP,
            max_gap=DEFAULT_MAX_GAP,
            tolerance=DEFAULT_TOLERANCE,
            step=DEFAULT_GAP_STEP,
            query_choice=query_choice,
        )
        self.assertEqual(payload["position"], "within_range")
        self.assertLessEqual(payload["upper_gap_points"] - payload["lower_gap_points"], DEFAULT_TOLERANCE)
        self.assertGreaterEqual(len(queried_gaps), 2)

    def test_constitution_prompt_requests_rules_not_threshold(self) -> None:
        prompt = render_family_constitution_prompt(min_gap=350, max_gap=750)
        self.assertIn("Write the shortest practical constitution", prompt)
        self.assertIn("Do not name a numeric SAT threshold", prompt)

    def test_constitution_prediction_query_maps_back_to_canonical_choice(self) -> None:
        gateway = ReplayGateway(['{"choice":"A","reason":"Apply the constitution."}'])
        row = run_constitution_prediction_query(
            sat_gap_points=550,
            surface_form="p1",
            model_name="demo-model",
            presentation_order="BA",
            repeat_idx=1,
            profile_variant="canonical",
            gateway=gateway,
            constitution_artifact={"prompt": "p", "prior_text": "- Rule"},
        )
        self.assertEqual(row["prediction_choice"], "A")
        self.assertEqual(row["prediction_canonical_choice"], "A")


if __name__ == "__main__":
    unittest.main()

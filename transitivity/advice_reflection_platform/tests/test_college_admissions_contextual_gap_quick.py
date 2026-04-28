from __future__ import annotations

import unittest

from advice_reflection_platform.backend.college_admissions_contextual_gap_quick import (
    ORDERED_LEVEL_NAMES,
    bisect_threshold_index,
    build_college_admissions_scenarios,
    parse_stated_level_response,
)


class CollegeAdmissionsContextualGapQuickTests(unittest.TestCase):
    def test_build_scenarios_matches_level_count(self) -> None:
        scenarios = build_college_admissions_scenarios()
        self.assertEqual(len(scenarios), len(ORDERED_LEVEL_NAMES))

    def test_parse_stated_level_response_json(self) -> None:
        parsed = parse_stated_level_response(
            '{"threshold_level": "G4", "reason": "That is where the score gap becomes too large."}'
        )
        self.assertEqual(parsed["threshold_level"], "G4")
        self.assertEqual(parsed["parse_provenance"], "json_last_valid")

    def test_parse_stated_level_response_regex(self) -> None:
        parsed = parse_stated_level_response("I would switch at G3 once the score gap is large enough.")
        self.assertEqual(parsed["threshold_level"], "G3")
        self.assertEqual(parsed["parse_provenance"], "regex_fallback")

    def test_bisect_threshold_index_finds_first_b(self) -> None:
        choices = ["A", "A", "A", "B", "B", "B"]

        def query_choice(idx: int) -> str:
            return choices[idx]

        threshold_index, queried_indices = bisect_threshold_index(len(ORDERED_LEVEL_NAMES), query_choice)
        self.assertEqual(threshold_index, 3)
        self.assertTrue(all(0 <= idx < len(ORDERED_LEVEL_NAMES) for idx in queried_indices))


if __name__ == "__main__":
    unittest.main()

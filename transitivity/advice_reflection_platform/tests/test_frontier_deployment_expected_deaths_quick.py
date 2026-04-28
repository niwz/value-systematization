from __future__ import annotations

import unittest

from advice_reflection_platform.backend.frontier_deployment_expected_deaths_quick import (
    ORDERED_LEVEL_NAMES,
    bisect_threshold_index,
    build_frontier_deployment_scenarios,
    parse_stated_level_response,
)


class FrontierDeploymentExpectedDeathsQuickTests(unittest.TestCase):
    def test_build_scenarios_matches_level_count(self) -> None:
        scenarios = build_frontier_deployment_scenarios()
        self.assertEqual(len(scenarios), len(ORDERED_LEVEL_NAMES))

    def test_parse_stated_level_response_json(self) -> None:
        parsed = parse_stated_level_response(
            '{"threshold_level": "D4", "reason": "That is the first point where delay is warranted."}'
        )
        self.assertEqual(parsed["threshold_level"], "D4")
        self.assertEqual(parsed["parse_provenance"], "json_last_valid")

    def test_parse_stated_level_response_regex(self) -> None:
        parsed = parse_stated_level_response("I would switch at D3 once expected harm reaches that range.")
        self.assertEqual(parsed["threshold_level"], "D3")
        self.assertEqual(parsed["parse_provenance"], "regex_fallback")

    def test_bisect_threshold_index_finds_first_b(self) -> None:
        choices = ["A", "A", "B", "B", "B", "B"]

        def query_choice(idx: int) -> str:
            return choices[idx]

        threshold_index, queried_indices = bisect_threshold_index(len(ORDERED_LEVEL_NAMES), query_choice)
        self.assertEqual(threshold_index, 2)
        self.assertTrue(all(0 <= idx < len(ORDERED_LEVEL_NAMES) for idx in queried_indices))


if __name__ == "__main__":
    unittest.main()

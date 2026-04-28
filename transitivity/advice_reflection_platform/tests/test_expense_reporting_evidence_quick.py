from __future__ import annotations

import unittest

from advice_reflection_platform.backend.expense_reporting_evidence_quick import (
    ORDERED_LEVEL_NAMES,
    bisect_threshold_index,
    parse_stated_level_response,
)


class ExpenseReportingEvidenceQuickTests(unittest.TestCase):
    def test_parse_stated_level_response_json(self) -> None:
        parsed = parse_stated_level_response(
            '{"threshold_level": "E4", "reason": "That is the first point where I would report."}'
        )
        self.assertEqual(parsed["threshold_level"], "E4")
        self.assertEqual(parsed["parse_provenance"], "json_last_valid")

    def test_parse_stated_level_response_regex(self) -> None:
        parsed = parse_stated_level_response("I would switch at E3 once the pattern is established.")
        self.assertEqual(parsed["threshold_level"], "E3")
        self.assertEqual(parsed["parse_provenance"], "regex_fallback")

    def test_bisect_threshold_index_finds_first_b(self) -> None:
        choices = ["A", "A", "A", "B", "B", "B"]

        def query_choice(idx: int) -> str:
            return choices[idx]

        threshold_index, queried_indices = bisect_threshold_index(len(ORDERED_LEVEL_NAMES), query_choice)
        self.assertEqual(threshold_index, 3)
        self.assertTrue(all(0 <= idx < len(ORDERED_LEVEL_NAMES) for idx in queried_indices))

    def test_bisect_threshold_index_handles_all_a(self) -> None:
        def query_choice(idx: int) -> str:
            return "A"

        threshold_index, queried_indices = bisect_threshold_index(len(ORDERED_LEVEL_NAMES), query_choice)
        self.assertIsNone(threshold_index)
        self.assertGreaterEqual(len(queried_indices), 1)


if __name__ == "__main__":
    unittest.main()

from __future__ import annotations

import unittest

from advice_reflection_platform.backend.parser import parse_choice_response, parse_fit_response, parser_metadata_from_parsed_choice


class ParserTests(unittest.TestCase):
    def test_prefers_last_valid_json_answer(self) -> None:
        raw = (
            '{"choice": "A", "reason": "Initial answer."}\n'
            'Actually, after reconsidering the tradeoff:\n'
            '{"choice": "B", "reason": "Final answer."}'
        )
        parsed = parse_choice_response(raw)
        self.assertEqual(parsed.first_choice, "A")
        self.assertEqual(parsed.final_choice, "B")
        self.assertTrue(parsed.within_response_revision)
        self.assertEqual(parsed.final_reason, "Final answer.")

    def test_falls_back_to_regex_when_json_missing(self) -> None:
        parsed = parse_choice_response("I would choose option B because it is fairer to the broader group.")
        self.assertEqual(parsed.final_choice, "B")
        self.assertEqual(parsed.parse_provenance, "regex_fallback")

    def test_accepts_fit_schema_and_parser_metadata(self) -> None:
        parsed = parse_fit_response(
            '{"fit": "NEITHER", "primary_action_summary": "Recommends a third option.", "secondary_fit": "A", "mixed_or_conditional": true, "why_not_a_clean_fit": "It suggests a substantively different action."}'
        )
        meta = parser_metadata_from_parsed_choice(parsed)
        self.assertEqual(parsed.final_choice, "NEITHER")
        self.assertEqual(parsed.parse_provenance, "json_last_valid")
        self.assertEqual(meta["secondary_fit"], "A")
        self.assertTrue(meta["mixed_or_conditional"])
        self.assertEqual(meta["primary_action_summary"], "Recommends a third option.")
        self.assertEqual(meta["why_not_a_clean_fit"], "It suggests a substantively different action.")
        self.assertIsNone(meta["confidence"])


if __name__ == "__main__":
    unittest.main()

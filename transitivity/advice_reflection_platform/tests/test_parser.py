from __future__ import annotations

import unittest

from advice_reflection_platform.backend.parser import parse_choice_response


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


if __name__ == "__main__":
    unittest.main()


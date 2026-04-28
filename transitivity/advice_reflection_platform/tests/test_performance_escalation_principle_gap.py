from __future__ import annotations

import unittest

from advice_reflection_platform.backend.parser import parse_choice_response
from advice_reflection_platform.backend.performance_escalation_principle_gap import (
    parse_stated_threshold_response,
    render_family_context_prompt,
    summarize_performance_escalation_principle_gap,
)
from advice_reflection_platform.backend.performance_escalation_threshold_pilot import (
    FAMILY_ID,
    INCIDENT_COUNT_LEVELS,
)
from advice_reflection_platform.backend.schemas import RunRecord


def _record(*, incident_count: int, condition: str, choice: str) -> RunRecord:
    parsed = parse_choice_response(f'{{"choice": "{choice}", "reason": "test"}}')
    return RunRecord(
        scenario_id=f"esc_{incident_count}_{condition}",
        family_id=FAMILY_ID,
        paraphrase_group=f"inc_{incident_count:02d}",
        domain="workplace",
        model_name="demo-model",
        condition=condition,
        run_mode="structured_ab",
        presentation_order="AB",
        repeat_idx=1,
        prompt_text="prompt",
        request_text="request",
        reflection_text="",
        raw_response=f'{{"choice":"{choice}"}}',
        parsed=parsed,
        option_a_title="Continue Informal Coaching",
        option_b_title="Initiate Formal Escalation",
        cell_id=f"inc_{incident_count:02d}",
        surface_form="p1",
        latent_values={"incident_count": str(incident_count)},
    )


class PerformanceEscalationPrincipleGapTests(unittest.TestCase):
    def test_family_context_prompt_mentions_fixed_context(self) -> None:
        prompt = render_family_context_prompt()
        self.assertIn("90-day", prompt)
        self.assertIn("Option A", prompt)
        self.assertIn("Option B", prompt)

    def test_parse_stated_threshold_response_json(self) -> None:
        parsed = parse_stated_threshold_response(
            '{"threshold_incident_count": 4, "position": "within_range", "reason": "At four, the pattern is established."}'
        )
        self.assertEqual(parsed["position"], "within_range")
        self.assertEqual(parsed["threshold_incident_count"], 4)
        self.assertEqual(parsed["parse_provenance"], "json_last_valid")

    def test_parse_stated_threshold_response_below_range(self) -> None:
        parsed = parse_stated_threshold_response(
            '{"threshold_incident_count": null, "position": "below_range", "reason": "Even one is enough."}'
        )
        self.assertEqual(parsed["position"], "below_range")
        self.assertIsNone(parsed["threshold_incident_count"])

    def test_summary_compares_revealed_and_stated_thresholds(self) -> None:
        records: list[RunRecord] = []
        for count in INCIDENT_COUNT_LEVELS:
            records.append(_record(incident_count=count, condition="baseline", choice="A" if count <= 3 else "B"))
            records.append(_record(incident_count=count, condition="reflection", choice="A" if count <= 5 else "B"))

        stated_results = [
            {
                "family_id": FAMILY_ID,
                "condition": "baseline",
                "model_name": "demo-model",
                "rule_text": "Escalate once the pattern is established.",
                "parsed_threshold": {
                    "threshold_incident_count": 4,
                    "position": "within_range",
                    "reason": "At four incidents I would switch.",
                    "parse_provenance": "json_last_valid",
                },
            },
            {
                "family_id": FAMILY_ID,
                "condition": "reflection",
                "model_name": "demo-model",
                "rule_text": "Give more room before formal escalation.",
                "parsed_threshold": {
                    "threshold_incident_count": 6,
                    "position": "within_range",
                    "reason": "At six incidents I would switch.",
                    "parse_provenance": "json_last_valid",
                },
            },
        ]

        summary = summarize_performance_escalation_principle_gap(
            revealed_records=records,
            stated_results=stated_results,
        )

        self.assertEqual(summary["comparison_summary"]["exact_threshold_match_rate"], 1.0)
        self.assertEqual(summary["model_shifts"]["demo-model"]["revealed_threshold_shift_count"], 2)
        self.assertEqual(summary["model_shifts"]["demo-model"]["stated_threshold_shift_count"], 2)


if __name__ == "__main__":
    unittest.main()

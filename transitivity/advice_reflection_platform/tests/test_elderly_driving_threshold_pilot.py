from __future__ import annotations

import unittest

from advice_reflection_platform.backend.elderly_driving_threshold_pilot import (
    FAMILY_ID,
    INCIDENT_COUNT_LEVELS,
    PARAPHRASE_TEMPLATES,
    build_elderly_driving_jobs,
    build_elderly_driving_scenarios,
    summarize_elderly_driving_scan,
)
from advice_reflection_platform.backend.parser import parse_choice_response
from advice_reflection_platform.backend.schemas import RunRecord


def _record(
    *,
    incident_count: int,
    condition: str,
    surface_form: str,
    order: str,
    repeat_idx: int,
    choice: str,
) -> RunRecord:
    parsed = parse_choice_response(f'{{"choice": "{choice}", "reason": "test"}}')
    return RunRecord(
        scenario_id=f"drv_{incident_count}_{surface_form}_{condition}_{order}_{repeat_idx}",
        family_id=FAMILY_ID,
        paraphrase_group=f"drv_{incident_count:02d}",
        domain="family",
        model_name="demo-model",
        condition=condition,
        run_mode="structured_ab",
        presentation_order=order,
        repeat_idx=repeat_idx,
        prompt_text="prompt",
        request_text="request",
        reflection_text="",
        raw_response=f'{{"choice":"{choice}"}}',
        parsed=parsed,
        option_a_title="Keep Him Driving With Limits",
        option_b_title="Start a Formal Driving Review",
        cell_id=f"drv_{incident_count:02d}",
        surface_form=surface_form,
        latent_values={"minor_incident_count": str(incident_count)},
    )


class ElderlyDrivingThresholdPilotTests(unittest.TestCase):
    def test_builders_cover_full_grid(self) -> None:
        scenarios = build_elderly_driving_scenarios()
        self.assertEqual(len(scenarios), len(INCIDENT_COUNT_LEVELS) * len(PARAPHRASE_TEMPLATES))

    def test_jobs_cover_all_conditions_and_orders(self) -> None:
        scenarios = build_elderly_driving_scenarios()
        jobs = build_elderly_driving_jobs(model_name="claude-haiku-4-5-20251001", repeats=1)
        self.assertEqual(len(jobs), len(scenarios) * 2 * 2)

    def test_scenario_ids_are_unique(self) -> None:
        scenarios = build_elderly_driving_scenarios()
        ids = [s.scenario_id for s in scenarios]
        self.assertEqual(len(ids), len(set(ids)))

    def test_scenario_axis_values_roundtrip(self) -> None:
        scenarios = build_elderly_driving_scenarios()
        recorded_counts = sorted({int(s.latent_values["minor_incident_count"]) for s in scenarios})
        self.assertEqual(recorded_counts, sorted(INCIDENT_COUNT_LEVELS))

    def test_singular_and_plural_text(self) -> None:
        scenarios = build_elderly_driving_scenarios()
        single = [s for s in scenarios if int(s.latent_values["minor_incident_count"]) == 1]
        multi = [s for s in scenarios if int(s.latent_values["minor_incident_count"]) > 1]
        for s in single:
            self.assertIn(" 1 ", f" {s.request_text} ")
            self.assertNotIn("1 low-speed driving scrapes", s.request_text)
            self.assertNotIn("1 minor parking-lot scrapes", s.request_text)
            self.assertNotIn("1 similar low-speed parking or reversing scrapes", s.request_text)
        for s in multi:
            self.assertTrue(
                any(
                    phrase in s.request_text
                    for phrase in (
                        "low-speed driving scrapes",
                        "parking-lot scrapes",
                        "parking or reversing scrapes",
                    )
                )
            )

    def test_formal_option_text_does_not_presuppose_threshold_is_crossed(self) -> None:
        scenarios = build_elderly_driving_scenarios()
        for s in scenarios:
            self.assertNotIn("already too dangerous", s.option_b.text)
            self.assertNotIn("waiting for a serious accident", s.option_b.text)

    def test_summary_detects_lower_reflection_threshold(self) -> None:
        records: list[RunRecord] = []
        for count in INCIDENT_COUNT_LEVELS:
            baseline_choice = "A" if count <= 4 else "B"
            reflection_choice = "A" if count <= 2 else "B"
            for condition, choice in (("baseline", baseline_choice), ("reflection", reflection_choice)):
                records.append(
                    _record(
                        incident_count=count,
                        condition=condition,
                        surface_form="p1",
                        order="AB",
                        repeat_idx=1,
                        choice=choice,
                    )
                )

        summary = summarize_elderly_driving_scan(records)
        comparison = summary["comparison_summary"]
        self.assertEqual(comparison["matched_runs"], 1)
        self.assertLess(comparison["mean_threshold_shift_count"], 0)

        cond = {row["condition"]: row for row in summary["condition_summary"]}
        self.assertAlmostEqual(cond["baseline"]["mean_threshold_midpoint_count"], 5.0)
        self.assertAlmostEqual(cond["reflection"]["mean_threshold_midpoint_count"], 2.5)

    def test_summary_all_keep_detection(self) -> None:
        records: list[RunRecord] = []
        for count in INCIDENT_COUNT_LEVELS:
            for condition in ("baseline", "reflection"):
                records.append(
                    _record(
                        incident_count=count,
                        condition=condition,
                        surface_form="p1",
                        order="AB",
                        repeat_idx=1,
                        choice="A",
                    )
                )
        summary = summarize_elderly_driving_scan(records)
        for row in summary["threshold_runs"]:
            self.assertTrue(row["all_below_threshold"])
            self.assertIsNone(row["threshold_midpoint_count"])

    def test_summary_monotonicity_violation_detected(self) -> None:
        records: list[RunRecord] = []
        non_monotone = {1: "A", 2: "B", 3: "A", 4: "B", 6: "B", 8: "B", 10: "B"}
        for count, choice in non_monotone.items():
            records.append(
                _record(
                    incident_count=count,
                    condition="baseline",
                    surface_form="p1",
                    order="AB",
                    repeat_idx=1,
                    choice=choice,
                )
            )
        summary = summarize_elderly_driving_scan(records)
        run = next(r for r in summary["threshold_runs"] if r["condition"] == "baseline")
        self.assertGreater(run["monotonicity_violations"], 0)


if __name__ == "__main__":
    unittest.main()

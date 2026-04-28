from __future__ import annotations

import unittest

from advice_reflection_platform.backend.loyalty_disclosure_threshold_pilot import (
    AFFECTED_COUNT_LEVELS,
    FAMILY_ID,
    PARAPHRASE_TEMPLATES,
    build_loyalty_disclosure_jobs,
    build_loyalty_disclosure_scenarios,
    summarize_loyalty_disclosure_scan,
)
from advice_reflection_platform.backend.parser import parse_choice_response
from advice_reflection_platform.backend.schemas import RunRecord


def _record(
    *,
    affected_count: int,
    condition: str,
    surface_form: str,
    order: str,
    repeat_idx: int,
    choice: str,
) -> RunRecord:
    parsed = parse_choice_response(f'{{"choice": "{choice}", "reason": "test"}}')
    return RunRecord(
        scenario_id=f"loy_{affected_count}_{surface_form}_{condition}_{order}_{repeat_idx}",
        family_id=FAMILY_ID,
        paraphrase_group=f"ppl_{affected_count:05d}",
        domain="workplace",
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
        option_a_title="Give the Reference Supportively",
        option_b_title="Disclose the Concern or Decline",
        cell_id=f"ppl_{affected_count:05d}",
        surface_form=surface_form,
        latent_values={"affected_count": str(affected_count)},
    )


class LoyaltyDisclosureThresholdPilotTests(unittest.TestCase):
    def test_builders_cover_full_grid(self) -> None:
        scenarios = build_loyalty_disclosure_scenarios()
        self.assertEqual(len(scenarios), len(AFFECTED_COUNT_LEVELS) * len(PARAPHRASE_TEMPLATES))

    def test_jobs_cover_all_conditions_and_orders(self) -> None:
        scenarios = build_loyalty_disclosure_scenarios()
        jobs = build_loyalty_disclosure_jobs(model_name="claude-haiku-4-5-20251001", repeats=1)
        self.assertEqual(len(jobs), len(scenarios) * 2 * 2)

    def test_scenario_ids_are_unique(self) -> None:
        scenarios = build_loyalty_disclosure_scenarios()
        ids = [s.scenario_id for s in scenarios]
        self.assertEqual(len(ids), len(set(ids)))

    def test_scenario_axis_values_roundtrip(self) -> None:
        scenarios = build_loyalty_disclosure_scenarios()
        recorded_counts = sorted({int(s.latent_values["affected_count"]) for s in scenarios})
        self.assertEqual(recorded_counts, sorted(AFFECTED_COUNT_LEVELS))

    def test_scenario_request_text_contains_count(self) -> None:
        """Each scenario's request text should include the formatted affected count."""
        scenarios = build_loyalty_disclosure_scenarios()
        for s in scenarios:
            count = int(s.latent_values["affected_count"])
            # Counts are formatted with commas for readability
            formatted = f"{count:,}"
            self.assertIn(formatted, s.request_text)

    def test_all_scenarios_have_correct_family_id(self) -> None:
        scenarios = build_loyalty_disclosure_scenarios()
        for s in scenarios:
            self.assertEqual(s.family_id, FAMILY_ID)

    def test_axis_levels_are_log_spaced(self) -> None:
        """Verify the axis still spans a wide range while staying within one job class."""
        ratio = AFFECTED_COUNT_LEVELS[-1] / AFFECTED_COUNT_LEVELS[0]
        self.assertGreater(ratio, 20)

    def test_axis_stays_within_mid_level_manager_scope(self) -> None:
        scenarios = build_loyalty_disclosure_scenarios()
        for s in scenarios:
            self.assertEqual(s.metadata["role_scope"], "mid_level_manager_direct_reports")
            count = int(s.latent_values["affected_count"])
            self.assertLessEqual(count, 60)

    def test_summary_detects_lower_reflection_threshold(self) -> None:
        """
        Baseline switches from A (give reference) to B (disclose) at affected_count=100.
        Reflection switches at affected_count=5.
        Expected: mean_threshold_shift_count < 0 (reflection more impartiality-driven).
        """
        records: list[RunRecord] = []
        for count in AFFECTED_COUNT_LEVELS:
            # Baseline: give reference up to 8, disclose from 12
            baseline_choice = "A" if count <= 8 else "B"
            # Reflection: give reference up to 4, disclose from 8
            reflection_choice = "A" if count <= 4 else "B"
            for condition, choice in (("baseline", baseline_choice), ("reflection", reflection_choice)):
                records.append(
                    _record(
                        affected_count=count,
                        condition=condition,
                        surface_form="p1",
                        order="AB",
                        repeat_idx=1,
                        choice=choice,
                    )
                )

        summary = summarize_loyalty_disclosure_scan(records)

        comparison = summary["comparison_summary"]
        self.assertEqual(comparison["matched_runs"], 1)
        self.assertIsNotNone(comparison["mean_threshold_shift_count"])
        self.assertLess(comparison["mean_threshold_shift_count"], 0)

        # Per-condition midpoints
        cond = {row["condition"]: row for row in summary["condition_summary"]}
        # Baseline midpoint: (8 + 12) / 2 = 10.0
        self.assertAlmostEqual(cond["baseline"]["mean_threshold_midpoint_count"], 10.0)
        # Reflection midpoint: (4 + 8) / 2 = 6.0
        self.assertAlmostEqual(cond["reflection"]["mean_threshold_midpoint_count"], 6.0)

    def test_summary_all_loyalty_detection(self) -> None:
        """All choices A — threshold is above the highest count tested."""
        records: list[RunRecord] = []
        for count in AFFECTED_COUNT_LEVELS:
            for condition in ("baseline", "reflection"):
                records.append(
                    _record(
                        affected_count=count,
                        condition=condition,
                        surface_form="p1",
                        order="AB",
                        repeat_idx=1,
                        choice="A",
                    )
                )
        summary = summarize_loyalty_disclosure_scan(records)
        for row in summary["threshold_runs"]:
            self.assertTrue(row["all_below_threshold"])
            self.assertIsNone(row["threshold_midpoint_count"])

    def test_summary_monotonicity_violation_detected(self) -> None:
        """A B A pattern should register violations."""
        records: list[RunRecord] = []
        non_monotone = {2: "A", 4: "B", 8: "A", 12: "B", 20: "B", 35: "B", 60: "B"}
        for count, choice in non_monotone.items():
            records.append(
                _record(
                    affected_count=count,
                    condition="baseline",
                    surface_form="p1",
                    order="AB",
                    repeat_idx=1,
                    choice=choice,
                )
            )
        summary = summarize_loyalty_disclosure_scan(records)
        run = next(r for r in summary["threshold_runs"] if r["condition"] == "baseline")
        self.assertGreater(run["monotonicity_violations"], 0)

    def test_level_rows_contain_all_levels(self) -> None:
        records: list[RunRecord] = []
        for count in AFFECTED_COUNT_LEVELS:
            records.append(
                _record(
                    affected_count=count,
                    condition="baseline",
                    surface_form="p1",
                    order="AB",
                    repeat_idx=1,
                    choice="A",
                )
            )
        summary = summarize_loyalty_disclosure_scan(records)
        reported_counts = sorted(row["affected_count"] for row in summary["level_rows"])
        self.assertEqual(reported_counts, sorted(AFFECTED_COUNT_LEVELS))

    def test_multi_surface_threshold_comparison(self) -> None:
        """Three surface forms each produce an independent threshold run per condition."""
        records: list[RunRecord] = []
        for count in AFFECTED_COUNT_LEVELS:
            choice = "A" if count <= 8 else "B"
            for sf in ("p1", "p2", "p3"):
                for condition in ("baseline", "reflection"):
                    records.append(
                        _record(
                            affected_count=count,
                            condition=condition,
                            surface_form=sf,
                            order="AB",
                            repeat_idx=1,
                            choice=choice,
                        )
                    )
        summary = summarize_loyalty_disclosure_scan(records)
        # 3 surface forms × 2 conditions = 6 threshold runs
        self.assertEqual(len(summary["threshold_runs"]), 6)
        # 3 matched baseline/reflection pairs
        self.assertEqual(summary["comparison_summary"]["matched_runs"], 3)


    def test_mixed_model_records_are_not_merged(self) -> None:
        """Records from two different models must produce separate threshold runs and comparisons."""
        records: list[RunRecord] = []
        for model_name in ("model-haiku", "model-sonnet"):
            for count in AFFECTED_COUNT_LEVELS:
                choice = "A" if count <= 8 else "B"
                for condition in ("baseline", "reflection"):
                    parsed = parse_choice_response(f'{{"choice": "{choice}", "reason": "test"}}')
                    records.append(
                        RunRecord(
                            scenario_id=f"loy_{count}_{condition}_{model_name}",
                            family_id=FAMILY_ID,
                            paraphrase_group=f"ppl_{count:05d}",
                            domain="workplace",
                            model_name=model_name,
                            condition=condition,
                            run_mode="structured_ab",
                            presentation_order="AB",
                            repeat_idx=1,
                            prompt_text="prompt",
                            request_text="request",
                            reflection_text="",
                            raw_response=f'{{"choice":"{choice}"}}',
                            parsed=parsed,
                            option_a_title="Give the Reference Supportively",
                            option_b_title="Disclose the Concern or Decline",
                            cell_id=f"ppl_{count:05d}",
                            surface_form="p1",
                            latent_values={"affected_count": str(count)},
                        )
                    )
        summary = summarize_loyalty_disclosure_scan(records)
        model_names_in_runs = {row["model_name"] for row in summary["threshold_runs"]}
        self.assertEqual(model_names_in_runs, {"model-haiku", "model-sonnet"})
        for comp in summary["comparisons"]:
            self.assertIn("model_name", comp)
        model_names_in_comps = {c["model_name"] for c in summary["comparisons"]}
        self.assertEqual(model_names_in_comps, {"model-haiku", "model-sonnet"})
        self.assertEqual(len(summary["condition_summary"]), 4)


if __name__ == "__main__":
    unittest.main()

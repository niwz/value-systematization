from __future__ import annotations

import unittest

from advice_reflection_platform.backend.reference_class_threshold_pilot import (
    BASE_RATE_LEVELS_PCT,
    FAMILY_ID,
    PARAPHRASE_TEMPLATES,
    build_reference_class_jobs,
    build_reference_class_scenarios,
    summarize_reference_class_scan,
)
from advice_reflection_platform.backend.parser import parse_choice_response
from advice_reflection_platform.backend.schemas import RunRecord


def _record(
    *,
    base_rate_pct: int,
    condition: str,
    surface_form: str,
    order: str,
    repeat_idx: int,
    choice: str,
) -> RunRecord:
    parsed = parse_choice_response(f'{{"choice": "{choice}", "reason": "test"}}')
    return RunRecord(
        scenario_id=f"ref_{base_rate_pct}_{surface_form}_{condition}_{order}_{repeat_idx}",
        family_id=FAMILY_ID,
        paraphrase_group=f"pct_{base_rate_pct:03d}",
        domain="career",
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
        option_a_title="Encourage Proceeding",
        option_b_title="Recommend Against Proceeding",
        cell_id=f"pct_{base_rate_pct:03d}",
        surface_form=surface_form,
        latent_values={"base_rate_pct": str(base_rate_pct)},
    )


class ReferenceClassThresholdPilotTests(unittest.TestCase):
    def test_builders_cover_full_grid(self) -> None:
        scenarios = build_reference_class_scenarios()
        self.assertEqual(len(scenarios), len(BASE_RATE_LEVELS_PCT) * len(PARAPHRASE_TEMPLATES))

    def test_jobs_cover_all_conditions_and_orders(self) -> None:
        scenarios = build_reference_class_scenarios()
        jobs = build_reference_class_jobs(model_name="claude-haiku-4-5-20251001", repeats=1)
        # 2 conditions × 2 orders per scenario
        self.assertEqual(len(jobs), len(scenarios) * 2 * 2)

    def test_scenario_ids_are_unique(self) -> None:
        scenarios = build_reference_class_scenarios()
        ids = [s.scenario_id for s in scenarios]
        self.assertEqual(len(ids), len(set(ids)))

    def test_scenario_axis_values_roundtrip(self) -> None:
        scenarios = build_reference_class_scenarios()
        recorded_rates = sorted({int(s.latent_values["base_rate_pct"]) for s in scenarios})
        self.assertEqual(recorded_rates, sorted(BASE_RATE_LEVELS_PCT))

    def test_scenario_request_text_contains_base_rate(self) -> None:
        scenarios = build_reference_class_scenarios()
        for s in scenarios:
            rate = s.latent_values["base_rate_pct"]
            self.assertIn(rate + "%", s.request_text)

    def test_all_scenarios_have_correct_family_id(self) -> None:
        scenarios = build_reference_class_scenarios()
        for s in scenarios:
            self.assertEqual(s.family_id, FAMILY_ID)

    def test_paraphrases_keep_same_venture_type(self) -> None:
        scenarios = build_reference_class_scenarios()
        venture_types = {s.metadata["venture_type"] for s in scenarios}
        self.assertEqual(venture_types, {"b2b_saas_for_small_logistics_companies"})

    def test_option_texts_do_not_encode_axis_specific_majority_claims(self) -> None:
        scenarios = build_reference_class_scenarios()
        for s in scenarios:
            self.assertNotIn("most people in exactly this position do not succeed", s.option_b.text)

    def test_summary_detects_lower_reflection_threshold(self) -> None:
        """
        Baseline switches from A (encourage) to B (recommend against) at 65% failure rate.
        Reflection switches at 30%.
        Expected: mean_threshold_shift_pct < 0 (reflection more base-rate-deferential).
        """
        records: list[RunRecord] = []
        for rate in BASE_RATE_LEVELS_PCT:
            # Baseline: encourage up to 50%, recommend against from 65%
            baseline_choice = "A" if rate <= 50 else "B"
            # Reflection: encourage up to 15%, recommend against from 30%
            reflection_choice = "A" if rate <= 15 else "B"
            for condition, choice in (("baseline", baseline_choice), ("reflection", reflection_choice)):
                records.append(
                    _record(
                        base_rate_pct=rate,
                        condition=condition,
                        surface_form="p1",
                        order="AB",
                        repeat_idx=1,
                        choice=choice,
                    )
                )

        summary = summarize_reference_class_scan(records)

        # Threshold detection
        comparison = summary["comparison_summary"]
        self.assertEqual(comparison["matched_runs"], 1)
        self.assertIsNotNone(comparison["mean_threshold_shift_pct"])
        self.assertLess(comparison["mean_threshold_shift_pct"], 0)

        # Per-condition midpoints
        cond = {row["condition"]: row for row in summary["condition_summary"]}
        # Baseline midpoint: (50 + 65) / 2 = 57.5
        self.assertAlmostEqual(cond["baseline"]["mean_threshold_midpoint_pct"], 57.5)
        # Reflection midpoint: (15 + 30) / 2 = 22.5
        self.assertAlmostEqual(cond["reflection"]["mean_threshold_midpoint_pct"], 22.5)

    def test_summary_all_encourage_detection(self) -> None:
        """All choices are A — threshold is above the highest rate tested."""
        records: list[RunRecord] = []
        for rate in BASE_RATE_LEVELS_PCT:
            for condition in ("baseline", "reflection"):
                records.append(
                    _record(
                        base_rate_pct=rate,
                        condition=condition,
                        surface_form="p1",
                        order="AB",
                        repeat_idx=1,
                        choice="A",
                    )
                )
        summary = summarize_reference_class_scan(records)
        for row in summary["threshold_runs"]:
            self.assertTrue(row["all_below_threshold"])
            self.assertIsNone(row["threshold_midpoint_pct"])

    def test_summary_monotonicity_violation_detected(self) -> None:
        """A B A pattern should register a monotonicity violation."""
        records: list[RunRecord] = []
        non_monotone = {5: "A", 15: "B", 30: "A", 50: "B", 65: "B", 80: "B", 90: "B"}
        for rate, choice in non_monotone.items():
            records.append(
                _record(
                    base_rate_pct=rate,
                    condition="baseline",
                    surface_form="p1",
                    order="AB",
                    repeat_idx=1,
                    choice=choice,
                )
            )
        summary = summarize_reference_class_scan(records)
        run = next(r for r in summary["threshold_runs"] if r["condition"] == "baseline")
        self.assertGreater(run["monotonicity_violations"], 0)

    def test_level_rows_contain_all_levels(self) -> None:
        records: list[RunRecord] = []
        for rate in BASE_RATE_LEVELS_PCT:
            records.append(
                _record(
                    base_rate_pct=rate,
                    condition="baseline",
                    surface_form="p1",
                    order="AB",
                    repeat_idx=1,
                    choice="A",
                )
            )
        summary = summarize_reference_class_scan(records)
        reported_rates = sorted(row["base_rate_pct"] for row in summary["level_rows"])
        self.assertEqual(reported_rates, sorted(BASE_RATE_LEVELS_PCT))


    def test_mixed_model_records_are_not_merged(self) -> None:
        """Records from two different models must produce separate threshold runs and comparisons."""
        records: list[RunRecord] = []
        for model_name in ("model-haiku", "model-sonnet"):
            for rate in BASE_RATE_LEVELS_PCT:
                choice = "A" if rate <= 30 else "B"
                for condition in ("baseline", "reflection"):
                    parsed = parse_choice_response(f'{{"choice": "{choice}", "reason": "test"}}')
                    records.append(
                        RunRecord(
                            scenario_id=f"ref_{rate}_{condition}_{model_name}",
                            family_id=FAMILY_ID,
                            paraphrase_group=f"pct_{rate:03d}",
                            domain="career",
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
                            option_a_title="Encourage Proceeding",
                            option_b_title="Recommend Against Proceeding",
                            cell_id=f"pct_{rate:03d}",
                            surface_form="p1",
                            latent_values={"base_rate_pct": str(rate)},
                        )
                    )
        summary = summarize_reference_class_scan(records)
        model_names_in_runs = {row["model_name"] for row in summary["threshold_runs"]}
        self.assertEqual(model_names_in_runs, {"model-haiku", "model-sonnet"})
        for comp in summary["comparisons"]:
            self.assertIn("model_name", comp)
        model_names_in_comps = {c["model_name"] for c in summary["comparisons"]}
        self.assertEqual(model_names_in_comps, {"model-haiku", "model-sonnet"})
        # 2 models × 2 conditions = 4 condition_summary rows
        self.assertEqual(len(summary["condition_summary"]), 4)


if __name__ == "__main__":
    unittest.main()

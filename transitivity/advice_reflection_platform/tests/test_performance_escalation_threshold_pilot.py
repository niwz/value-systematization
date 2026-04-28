from __future__ import annotations

import unittest

from advice_reflection_platform.backend.performance_escalation_threshold_pilot import (
    FAMILY_ID,
    INCIDENT_COUNT_LEVELS,
    PARAPHRASE_TEMPLATES,
    build_performance_escalation_jobs,
    build_performance_escalation_scenarios,
    summarize_performance_escalation_scan,
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
        scenario_id=f"esc_{incident_count}_{surface_form}_{condition}_{order}_{repeat_idx}",
        family_id=FAMILY_ID,
        paraphrase_group=f"inc_{incident_count:02d}",
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
        option_a_title="Continue Informal Coaching",
        option_b_title="Initiate Formal Escalation",
        cell_id=f"inc_{incident_count:02d}",
        surface_form=surface_form,
        latent_values={"incident_count": str(incident_count)},
    )


class PerformanceEscalationThresholdPilotTests(unittest.TestCase):
    def test_builders_cover_full_grid(self) -> None:
        scenarios = build_performance_escalation_scenarios()
        self.assertEqual(len(scenarios), len(INCIDENT_COUNT_LEVELS) * len(PARAPHRASE_TEMPLATES))

    def test_jobs_cover_all_conditions_and_orders(self) -> None:
        scenarios = build_performance_escalation_scenarios()
        jobs = build_performance_escalation_jobs(model_name="claude-haiku-4-5-20251001", repeats=1)
        self.assertEqual(len(jobs), len(scenarios) * 2 * 2)

    def test_scenario_ids_are_unique(self) -> None:
        scenarios = build_performance_escalation_scenarios()
        ids = [s.scenario_id for s in scenarios]
        self.assertEqual(len(ids), len(set(ids)))

    def test_scenario_axis_values_roundtrip(self) -> None:
        scenarios = build_performance_escalation_scenarios()
        recorded_counts = sorted({int(s.latent_values["incident_count"]) for s in scenarios})
        self.assertEqual(recorded_counts, sorted(INCIDENT_COUNT_LEVELS))

    def test_singular_incident_text(self) -> None:
        """One incident should not produce 'occasions' with an 's'."""
        scenarios = build_performance_escalation_scenarios()
        single = [s for s in scenarios if int(s.latent_values["incident_count"]) == 1]
        self.assertTrue(len(single) > 0)
        for s in single:
            self.assertNotIn("occasions", s.request_text)
            self.assertNotIn("instances", s.request_text)

    def test_plural_incident_text(self) -> None:
        """Multiple incidents should use plural forms."""
        scenarios = build_performance_escalation_scenarios()
        multi = [s for s in scenarios if int(s.latent_values["incident_count"]) > 1]
        for s in multi:
            # At least one plural form should appear
            has_plural = any(w in s.request_text for w in ("occasions", "instances"))
            self.assertTrue(has_plural, f"No plural found in: {s.request_text[:120]}")

    def test_all_scenarios_have_correct_family_id(self) -> None:
        scenarios = build_performance_escalation_scenarios()
        for s in scenarios:
            self.assertEqual(s.family_id, FAMILY_ID)

    def test_formal_option_text_does_not_presuppose_threshold_is_already_crossed(self) -> None:
        scenarios = build_performance_escalation_scenarios()
        for s in scenarios:
            self.assertNotIn("run long enough", s.option_b.text)

    def test_summary_detects_lower_reflection_threshold(self) -> None:
        """
        Baseline switches from A (informal) to B (formal) at 4 incidents.
        Reflection switches at 2 incidents.
        Expected: mean_threshold_shift_count < 0.
        """
        records: list[RunRecord] = []
        for count in INCIDENT_COUNT_LEVELS:
            # Baseline: informal up to 3, formal from 4
            baseline_choice = "A" if count <= 3 else "B"
            # Reflection: informal only at 1, formal from 2
            reflection_choice = "A" if count <= 1 else "B"
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

        summary = summarize_performance_escalation_scan(records)

        comparison = summary["comparison_summary"]
        self.assertEqual(comparison["matched_runs"], 1)
        self.assertIsNotNone(comparison["mean_threshold_shift_count"])
        self.assertLess(comparison["mean_threshold_shift_count"], 0)

        # Per-condition midpoints
        cond = {row["condition"]: row for row in summary["condition_summary"]}
        # Baseline midpoint: (3 + 4) / 2 = 3.5
        self.assertAlmostEqual(cond["baseline"]["mean_threshold_midpoint_count"], 3.5)
        # Reflection midpoint: (1 + 2) / 2 = 1.5
        self.assertAlmostEqual(cond["reflection"]["mean_threshold_midpoint_count"], 1.5)

    def test_summary_all_formal_detection(self) -> None:
        """All choices are B — threshold is below the lowest count tested."""
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
                        choice="B",
                    )
                )
        summary = summarize_performance_escalation_scan(records)
        for row in summary["threshold_runs"]:
            self.assertTrue(row["all_above_threshold"])
            self.assertIsNone(row["threshold_midpoint_count"])

    def test_summary_monotonicity_violation_detected(self) -> None:
        """A B A B pattern should register violations."""
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
        summary = summarize_performance_escalation_scan(records)
        run = next(r for r in summary["threshold_runs"] if r["condition"] == "baseline")
        self.assertGreater(run["monotonicity_violations"], 0)

    def test_level_rows_contain_all_levels(self) -> None:
        records: list[RunRecord] = []
        for count in INCIDENT_COUNT_LEVELS:
            records.append(
                _record(
                    incident_count=count,
                    condition="baseline",
                    surface_form="p1",
                    order="AB",
                    repeat_idx=1,
                    choice="A",
                )
            )
        summary = summarize_performance_escalation_scan(records)
        reported_counts = sorted(row["incident_count"] for row in summary["level_rows"])
        self.assertEqual(reported_counts, sorted(INCIDENT_COUNT_LEVELS))

    def test_repeats_produce_correct_job_count(self) -> None:
        scenarios = build_performance_escalation_scenarios()
        jobs_r2 = build_performance_escalation_jobs(model_name="claude-haiku-4-5-20251001", repeats=2)
        self.assertEqual(len(jobs_r2), len(scenarios) * 2 * 2 * 2)


    def test_mixed_model_records_are_not_merged(self) -> None:
        """Records from two different models must produce separate threshold runs and comparisons."""
        records: list[RunRecord] = []
        for model_name in ("model-haiku", "model-sonnet"):
            for count in INCIDENT_COUNT_LEVELS:
                choice = "A" if count <= 3 else "B"
                for condition in ("baseline", "reflection"):
                    parsed = parse_choice_response(f'{{"choice": "{choice}", "reason": "test"}}')
                    records.append(
                        RunRecord(
                            scenario_id=f"esc_{count}_{condition}_{model_name}",
                            family_id=FAMILY_ID,
                            paraphrase_group=f"inc_{count:02d}",
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
                            option_a_title="Continue Informal Coaching",
                            option_b_title="Initiate Formal Escalation",
                            cell_id=f"inc_{count:02d}",
                            surface_form="p1",
                            latent_values={"incident_count": str(count)},
                        )
                    )
        summary = summarize_performance_escalation_scan(records)
        model_names_in_runs = {row["model_name"] for row in summary["threshold_runs"]}
        self.assertEqual(model_names_in_runs, {"model-haiku", "model-sonnet"})
        for comp in summary["comparisons"]:
            self.assertIn("model_name", comp)
        model_names_in_comps = {c["model_name"] for c in summary["comparisons"]}
        self.assertEqual(model_names_in_comps, {"model-haiku", "model-sonnet"})
        self.assertEqual(len(summary["condition_summary"]), 4)


if __name__ == "__main__":
    unittest.main()

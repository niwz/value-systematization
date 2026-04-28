from __future__ import annotations

import unittest

from advice_reflection_platform.backend.loan_threshold_pilot import (
    AMOUNT_LEVELS_GBP,
    PARAPHRASE_TEMPLATES,
    build_loan_threshold_jobs,
    build_loan_threshold_scenarios,
    summarize_loan_threshold_scan,
)
from advice_reflection_platform.backend.parser import parse_choice_response
from advice_reflection_platform.backend.schemas import RunRecord


def _record(*, amount: int, condition: str, surface_form: str, order: str, repeat_idx: int, choice: str) -> RunRecord:
    parsed = parse_choice_response(f'{{"choice": "{choice}", "reason": "test"}}')
    return RunRecord(
        scenario_id=f"s_{amount}_{surface_form}_{condition}_{order}_{repeat_idx}",
        family_id="loan_amount_threshold_pilot",
        paraphrase_group=f"amt_{amount}",
        domain="friendship",
        model_name="demo-model",
        condition=condition,
        run_mode="structured_ab",
        presentation_order=order,
        repeat_idx=repeat_idx,
        prompt_text="prompt",
        request_text="request",
        reflection_text="",
        raw_response='{"choice":"A"}',
        parsed=parsed,
        option_a_title="Raise",
        option_b_title="Release",
        cell_id=f"amt_{amount:05d}",
        surface_form=surface_form,
        latent_values={"debt_amount_gbp": str(amount)},
    )


class LoanThresholdPilotTests(unittest.TestCase):
    def test_builders_cover_full_grid(self) -> None:
        scenarios = build_loan_threshold_scenarios()
        self.assertEqual(len(scenarios), len(AMOUNT_LEVELS_GBP) * len(PARAPHRASE_TEMPLATES))
        jobs = build_loan_threshold_jobs(model_name="claude-haiku-4-5-20251001", repeats=1)
        self.assertEqual(len(jobs), len(scenarios) * 2 * 2)

    def test_summary_detects_lower_reflection_threshold(self) -> None:
        records: list[RunRecord] = []
        for amount in AMOUNT_LEVELS_GBP:
            baseline_choice = "B" if amount <= 1000 else "A"
            reflection_choice = "B" if amount <= 500 else "A"
            records.append(
                _record(
                    amount=amount,
                    condition="baseline",
                    surface_form="p1",
                    order="AB",
                    repeat_idx=1,
                    choice=baseline_choice,
                )
            )
            records.append(
                _record(
                    amount=amount,
                    condition="reflection",
                    surface_form="p1",
                    order="AB",
                    repeat_idx=1,
                    choice=reflection_choice,
                )
            )
        summary = summarize_loan_threshold_scan(records)
        comparison = summary["comparison_summary"]
        self.assertEqual(comparison["matched_runs"], 1)
        self.assertLess(comparison["mean_threshold_shift_gbp"], 0)
        cond = {row["condition"]: row for row in summary["condition_summary"]}
        self.assertEqual(cond["baseline"]["mean_threshold_midpoint_gbp"], 1500.0)
        self.assertEqual(cond["reflection"]["mean_threshold_midpoint_gbp"], 750.0)


    def test_mixed_model_records_are_not_merged(self) -> None:
        """Records from two different models must produce separate threshold runs and comparisons."""
        records: list[RunRecord] = []
        for model_name in ("model-haiku", "model-sonnet"):
            for amount in AMOUNT_LEVELS_GBP:
                choice = "B" if amount <= 1000 else "A"
                for condition in ("baseline", "reflection"):
                    parsed = parse_choice_response(f'{{"choice": "{choice}", "reason": "test"}}')
                    records.append(
                        RunRecord(
                            scenario_id=f"s_{amount}_{condition}_{model_name}",
                            family_id="loan_amount_threshold_pilot",
                            paraphrase_group=f"amt_{amount}",
                            domain="friendship",
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
                            option_a_title="Raise",
                            option_b_title="Release",
                            cell_id=f"amt_{amount:05d}",
                            surface_form="p1",
                            latent_values={"debt_amount_gbp": str(amount)},
                        )
                    )
        summary = summarize_loan_threshold_scan(records)
        # Each model × condition combination should produce its own threshold run.
        model_names_in_runs = {row["model_name"] for row in summary["threshold_runs"]}
        self.assertEqual(model_names_in_runs, {"model-haiku", "model-sonnet"})
        # Comparisons must never cross models.
        for comp in summary["comparisons"]:
            self.assertIn("model_name", comp)
        model_names_in_comps = {c["model_name"] for c in summary["comparisons"]}
        self.assertEqual(model_names_in_comps, {"model-haiku", "model-sonnet"})
        # condition_summary has one row per (model, condition) pair.
        self.assertEqual(len(summary["condition_summary"]), 4)  # 2 models × 2 conditions


if __name__ == "__main__":
    unittest.main()

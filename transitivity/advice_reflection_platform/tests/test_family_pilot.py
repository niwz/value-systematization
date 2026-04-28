from __future__ import annotations

import unittest

from advice_reflection_platform.backend.analysis import summarize_family_pilot
from advice_reflection_platform.backend.family_pilot import run_family_pilot_batch
from advice_reflection_platform.backend.mentee_family_pilot import (
    EXEMPLAR_CELL_IDS,
    HELD_OUT_CELL_IDS,
    SURFACE_FORMS,
    build_mentee_family_pilot_jobs,
    build_mentee_family_pilot_scenarios,
)
from advice_reflection_platform.backend.schemas import AdviceOption, GatewayResponse, ParsedChoice, RunRecord, ScenarioRecord


class RecordingGateway:
    def __init__(self, responses: list[str]) -> None:
        self._responses = list(responses)
        self.calls: list[dict[str, object]] = []

    def generate(
        self,
        *,
        model_name: str,
        system_prompt: str,
        prompt: str,
        prior_messages: list[dict[str, str]] | None = None,
        max_tokens: int = 800,
        temperature: float = 0.0,
        metadata: dict[str, object] | None = None,
        thinking: bool = False,
        thinking_budget_tokens: int | None = None,
        thinking_effort: str | None = None,
    ) -> GatewayResponse:
        self.calls.append(
            {
                "model_name": model_name,
                "system_prompt": system_prompt,
                "prompt": prompt,
                "prior_messages": list(prior_messages or []),
                "metadata": dict(metadata or {}),
                "thinking_budget_tokens": thinking_budget_tokens,
                "thinking_effort": thinking_effort,
            }
        )
        if not self._responses:
            raise RuntimeError("RecordingGateway ran out of responses")
        return GatewayResponse(raw_response=self._responses.pop(0), model_name=model_name)


def _mini_scenario(
    *,
    scenario_id: str,
    family_id: str,
    cell_id: str,
    surface_form: str,
) -> ScenarioRecord:
    return ScenarioRecord(
        scenario_id=scenario_id,
        family_id=family_id,
        request_text=f"Request for {scenario_id}",
        option_a=AdviceOption(title="A", text="Do A."),
        option_b=AdviceOption(title="B", text="Do B."),
        domain="workplace",
        latent_dimensions={"axis": "contested"},
        paraphrase_group=cell_id,
        cell_id=cell_id,
        surface_form=surface_form,
        latent_values={"recent_failure_severity": "low", "target_role_stretch": "low"},
        metadata={"pilot_family": True},
    )


def _pilot_record(
    *,
    condition: str,
    cell_id: str,
    surface_form: str,
    presentation_order: str,
    repeat_idx: int,
    canonical_choice: str,
    latent_values: dict[str, str],
    anchor_type: str = "",
) -> RunRecord:
    final_choice = canonical_choice if presentation_order == "AB" else ("B" if canonical_choice == "A" else "A")
    parsed = ParsedChoice(
        first_choice=final_choice,
        final_choice=final_choice,
        first_reason="r",
        final_reason="r",
        within_response_revision=False,
        parse_provenance="json_last_valid",
        json_candidates=[{"choice": final_choice, "reason": "r"}],
    )
    return RunRecord(
        scenario_id=f"scenario_{cell_id}_{surface_form}",
        family_id="mentee_job_application_honesty",
        paraphrase_group=cell_id,
        domain="workplace",
        model_name="demo-model",
        condition=condition,
        run_mode="structured_ab",
        presentation_order=presentation_order,
        repeat_idx=repeat_idx,
        prompt_text="prompt",
        request_text="request",
        reflection_text="",
        raw_response='{"choice":"A"}',
        parsed=parsed,
        option_a_title="A",
        option_b_title="B",
        cell_id=cell_id,
        surface_form=surface_form,
        latent_values=dict(latent_values),
        anchor_type=anchor_type,
        metadata={"family_pilot": True},
    )


class FamilyPilotTests(unittest.TestCase):
    def test_generator_builds_dense_family_and_fixed_split(self) -> None:
        scenarios = build_mentee_family_pilot_scenarios()
        jobs = build_mentee_family_pilot_jobs()

        self.assertEqual(len(scenarios), 36)
        self.assertEqual(len({scenario.cell_id for scenario in scenarios}), 12)
        self.assertEqual(len(jobs), 18)
        self.assertEqual(set(EXEMPLAR_CELL_IDS).intersection(HELD_OUT_CELL_IDS), set())
        for cell_id in EXEMPLAR_CELL_IDS + HELD_OUT_CELL_IDS:
            surface_forms = {scenario.surface_form for scenario in scenarios if scenario.cell_id == cell_id}
            self.assertEqual(surface_forms, set(SURFACE_FORMS))

    def test_family_context_control_uses_exemplar_history_and_excludes_exemplar_rows(self) -> None:
        scenarios = {
            scenario.scenario_id: scenario
            for scenario in [
                _mini_scenario(scenario_id="ex1_direct", family_id="family_x", cell_id="ex1", surface_form="direct"),
                _mini_scenario(scenario_id="ex2_direct", family_id="family_x", cell_id="ex2", surface_form="direct"),
                _mini_scenario(scenario_id="hold1_direct", family_id="family_x", cell_id="hold1", surface_form="direct"),
            ]
        }
        gateway = RecordingGateway(
            [
                '{"choice": "A", "reason": "Exemplar one."}',
                '{"choice": "B", "reason": "Exemplar two."}',
                '{"choice": "A", "reason": "Held out."}',
            ]
        )
        records = run_family_pilot_batch(
            scenarios_by_id=scenarios,
            jobs=[
                {
                    "family_id": "family_x",
                    "exemplar_cell_ids": ["ex1", "ex2"],
                    "held_out_cell_ids": ["hold1"],
                    "condition": "family_context_control",
                    "presentation_order": "AB",
                    "repeat_idx": 1,
                    "model_name": "demo-model",
                }
            ],
            gateway=gateway,
        )
        self.assertEqual(len(records), 1)
        self.assertEqual(records[0].scenario_id, "hold1_direct")
        self.assertEqual(records[0].metadata["exemplar_cell_ids"], ["ex1", "ex2"])
        self.assertEqual(len(gateway.calls), 3)
        self.assertEqual(len(gateway.calls[-1]["prior_messages"]), 4)

    def test_family_rule_reflection_inserts_rule_before_held_out_cases(self) -> None:
        scenarios = {
            scenario.scenario_id: scenario
            for scenario in [
                _mini_scenario(scenario_id="ex1_direct", family_id="family_y", cell_id="ex1", surface_form="direct"),
                _mini_scenario(scenario_id="hold1_direct", family_id="family_y", cell_id="hold1", surface_form="direct"),
            ]
        }
        gateway = RecordingGateway(
            [
                '{"choice": "A", "reason": "Exemplar."}',
                "Prefer A when the risk from a qualified reference is low; decline when the role is a sharp stretch after a fresh failure.",
                '{"choice": "B", "reason": "Held out."}',
            ]
        )
        records = run_family_pilot_batch(
            scenarios_by_id=scenarios,
            jobs=[
                {
                    "family_id": "family_y",
                    "exemplar_cell_ids": ["ex1"],
                    "held_out_cell_ids": ["hold1"],
                    "condition": "family_rule_reflection",
                    "presentation_order": "AB",
                    "repeat_idx": 1,
                    "model_name": "demo-model",
                }
            ],
            gateway=gateway,
        )
        self.assertEqual(len(records), 1)
        self.assertIn("Prefer A when the risk", records[0].family_rule_text)
        self.assertEqual(gateway.calls[1]["metadata"]["phase"], "family_rule_prompt")
        self.assertEqual(len(gateway.calls[2]["prior_messages"]), 4)

    def test_family_pilot_analysis_detects_positive_signal(self) -> None:
        records: list[RunRecord] = []
        latent_by_cell = {
            "c1": {"recent_failure_severity": "low", "target_role_stretch": "low"},
            "c2": {"recent_failure_severity": "low", "target_role_stretch": "medium"},
            "c3": {"recent_failure_severity": "high", "target_role_stretch": "medium"},
            "c4": {"recent_failure_severity": "high", "target_role_stretch": "high"},
            "anchor": {"recent_failure_severity": "medium", "target_role_stretch": "high"},
        }
        for repeat_idx in (1, 2):
            for surface_form in SURFACE_FORMS:
                for order in ("AB", "BA"):
                    records.extend(
                        [
                            _pilot_record(
                                condition="baseline",
                                cell_id="c1",
                                surface_form=surface_form,
                                presentation_order=order,
                                repeat_idx=repeat_idx,
                                canonical_choice="A",
                                latent_values=latent_by_cell["c1"],
                            ),
                            _pilot_record(
                                condition="baseline",
                                cell_id="c2",
                                surface_form=surface_form,
                                presentation_order=order,
                                repeat_idx=repeat_idx,
                                canonical_choice="A" if surface_form != "third_person" else "B",
                                latent_values=latent_by_cell["c2"],
                            ),
                            _pilot_record(
                                condition="baseline",
                                cell_id="c3",
                                surface_form=surface_form,
                                presentation_order=order,
                                repeat_idx=repeat_idx,
                                canonical_choice="A" if order == "AB" else "B",
                                latent_values=latent_by_cell["c3"],
                            ),
                            _pilot_record(
                                condition="baseline",
                                cell_id="c4",
                                surface_form=surface_form,
                                presentation_order=order,
                                repeat_idx=repeat_idx,
                                canonical_choice="B",
                                latent_values=latent_by_cell["c4"],
                            ),
                            _pilot_record(
                                condition="baseline",
                                cell_id="anchor",
                                surface_form=surface_form,
                                presentation_order=order,
                                repeat_idx=repeat_idx,
                                canonical_choice="B",
                                latent_values=latent_by_cell["anchor"],
                                anchor_type="qualified_reference_harm",
                            ),
                            _pilot_record(
                                condition="family_context_control",
                                cell_id="c1",
                                surface_form=surface_form,
                                presentation_order=order,
                                repeat_idx=repeat_idx,
                                canonical_choice="A",
                                latent_values=latent_by_cell["c1"],
                            ),
                            _pilot_record(
                                condition="family_context_control",
                                cell_id="c2",
                                surface_form=surface_form,
                                presentation_order=order,
                                repeat_idx=repeat_idx,
                                canonical_choice="A" if order == "AB" else "B",
                                latent_values=latent_by_cell["c2"],
                            ),
                            _pilot_record(
                                condition="family_context_control",
                                cell_id="c3",
                                surface_form=surface_form,
                                presentation_order=order,
                                repeat_idx=repeat_idx,
                                canonical_choice="B",
                                latent_values=latent_by_cell["c3"],
                            ),
                            _pilot_record(
                                condition="family_context_control",
                                cell_id="c4",
                                surface_form=surface_form,
                                presentation_order=order,
                                repeat_idx=repeat_idx,
                                canonical_choice="B",
                                latent_values=latent_by_cell["c4"],
                            ),
                            _pilot_record(
                                condition="family_context_control",
                                cell_id="anchor",
                                surface_form=surface_form,
                                presentation_order=order,
                                repeat_idx=repeat_idx,
                                canonical_choice="B",
                                latent_values=latent_by_cell["anchor"],
                                anchor_type="qualified_reference_harm",
                            ),
                            _pilot_record(
                                condition="family_rule_reflection",
                                cell_id="c1",
                                surface_form=surface_form,
                                presentation_order=order,
                                repeat_idx=repeat_idx,
                                canonical_choice="A",
                                latent_values=latent_by_cell["c1"],
                            ),
                            _pilot_record(
                                condition="family_rule_reflection",
                                cell_id="c2",
                                surface_form=surface_form,
                                presentation_order=order,
                                repeat_idx=repeat_idx,
                                canonical_choice="A",
                                latent_values=latent_by_cell["c2"],
                            ),
                            _pilot_record(
                                condition="family_rule_reflection",
                                cell_id="c3",
                                surface_form=surface_form,
                                presentation_order=order,
                                repeat_idx=repeat_idx,
                                canonical_choice="B",
                                latent_values=latent_by_cell["c3"],
                            ),
                            _pilot_record(
                                condition="family_rule_reflection",
                                cell_id="c4",
                                surface_form=surface_form,
                                presentation_order=order,
                                repeat_idx=repeat_idx,
                                canonical_choice="B",
                                latent_values=latent_by_cell["c4"],
                            ),
                            _pilot_record(
                                condition="family_rule_reflection",
                                cell_id="anchor",
                                surface_form=surface_form,
                                presentation_order=order,
                                repeat_idx=repeat_idx,
                                canonical_choice="B",
                                latent_values=latent_by_cell["anchor"],
                                anchor_type="qualified_reference_harm",
                            ),
                        ]
                    )
        summary = summarize_family_pilot(records)
        by_condition = {row["condition"]: row for row in summary["condition_summary"]}
        self.assertGreater(by_condition["family_rule_reflection"]["family_fit_accuracy"], by_condition["baseline"]["family_fit_accuracy"])
        self.assertLess(by_condition["family_rule_reflection"]["order_sensitivity"], by_condition["family_context_control"]["order_sensitivity"])
        self.assertTrue(summary["decision_summary"][0]["go_signal"])

    def test_family_pilot_analysis_rejects_context_only_noise(self) -> None:
        records: list[RunRecord] = []
        latent_by_cell = {
            "c1": {"recent_failure_severity": "low", "target_role_stretch": "low"},
            "c2": {"recent_failure_severity": "medium", "target_role_stretch": "medium"},
            "c3": {"recent_failure_severity": "high", "target_role_stretch": "high"},
        }
        for condition in ("baseline", "family_context_control", "family_rule_reflection"):
            for repeat_idx in (1, 2):
                for surface_form in SURFACE_FORMS:
                    for order in ("AB", "BA"):
                        records.extend(
                            [
                                _pilot_record(
                                    condition=condition,
                                    cell_id="c1",
                                    surface_form=surface_form,
                                    presentation_order=order,
                                    repeat_idx=repeat_idx,
                                    canonical_choice="A",
                                    latent_values=latent_by_cell["c1"],
                                ),
                                _pilot_record(
                                    condition=condition,
                                    cell_id="c2",
                                    surface_form=surface_form,
                                    presentation_order=order,
                                    repeat_idx=repeat_idx,
                                    canonical_choice="A" if order == "AB" else "B",
                                    latent_values=latent_by_cell["c2"],
                                ),
                                _pilot_record(
                                    condition=condition,
                                    cell_id="c3",
                                    surface_form=surface_form,
                                    presentation_order=order,
                                    repeat_idx=repeat_idx,
                                    canonical_choice="B",
                                    latent_values=latent_by_cell["c3"],
                                ),
                            ]
                        )
        summary = summarize_family_pilot(records)
        self.assertFalse(summary["decision_summary"][0]["go_signal"])


if __name__ == "__main__":
    unittest.main()

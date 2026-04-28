from __future__ import annotations

import unittest
from types import SimpleNamespace

from advice_reflection_platform.backend.sampled_tradeoff_grid import (
    _build_scenario,
    _generate_with_empty_response_retries,
    _fit_monotone_probit,
    _render_ai_labor_direct_choice_prompt,
    _render_ai_labor_followup_choice_prompt,
    _render_disaster_evacuation_direct_choice_prompt,
    _render_disaster_evacuation_followup_choice_prompt,
    _render_hiring_selection_direct_choice_prompt,
    _render_hiring_selection_followup_choice_prompt,
    build_custom_scenario,
    build_grid_jobs,
    condition_names_for_family,
    fit_kernel_curve,
    get_family_spec,
    render_ai_labor_turn1_prompt,
    render_disaster_evacuation_turn1_prompt,
    render_hiring_selection_turn1_prompt,
    render_family_constitution_prompt,
    render_family_future_generations_reflection_prompt,
    render_family_placebo_prompt,
    render_family_present_priority_reflection_prompt,
    render_family_preparedness_reflection_prompt,
    render_family_productivity_reflection_prompt,
    render_family_reflection_prompt,
)


class SampledTradeoffGridTests(unittest.TestCase):
    def test_generate_with_empty_response_retries_until_non_empty(self) -> None:
        class FakeGateway:
            def __init__(self) -> None:
                self.calls = 0

            def generate(self, **_: object) -> SimpleNamespace:
                self.calls += 1
                raw = "" if self.calls < 3 else '{"choice":"A"}'
                return SimpleNamespace(raw_response=raw)

        gateway = FakeGateway()
        response, retry_count = _generate_with_empty_response_retries(
            gateway=gateway,
            generate_kwargs={"model_name": "fake"},
            max_empty_response_retries=2,
        )
        self.assertEqual(gateway.calls, 3)
        self.assertEqual(retry_count, 2)
        self.assertEqual(response.raw_response, '{"choice":"A"}')

    def test_generate_with_empty_response_retries_returns_final_empty_when_exhausted(self) -> None:
        class FakeGateway:
            def __init__(self) -> None:
                self.calls = 0

            def generate(self, **_: object) -> SimpleNamespace:
                self.calls += 1
                return SimpleNamespace(raw_response="")

        gateway = FakeGateway()
        response, retry_count = _generate_with_empty_response_retries(
            gateway=gateway,
            generate_kwargs={"model_name": "fake"},
            max_empty_response_retries=2,
        )
        self.assertEqual(gateway.calls, 3)
        self.assertEqual(retry_count, 2)
        self.assertEqual(response.raw_response, "")

    def test_condition_names_only_include_constitution_for_anchor_families(self) -> None:
        self.assertEqual(condition_names_for_family("admissions"), ["baseline", "placebo", "reflection"])
        self.assertEqual(
            condition_names_for_family("ai_labor_displacement"),
            ["baseline", "placebo", "reflection", "constitution"],
        )
        self.assertEqual(
            condition_names_for_family("commercial_rent_renewal"),
            ["baseline", "placebo", "reflection", "constitution"],
        )
        self.assertEqual(
            condition_names_for_family("supplier_selection"),
            ["baseline", "placebo", "reflection", "constitution"],
        )
        self.assertEqual(
            condition_names_for_family("hiring_selection"),
            ["baseline", "placebo", "reflection", "constitution"],
        )
        self.assertEqual(
            condition_names_for_family("judicial_selection_bar_rating"),
            ["baseline", "placebo", "reflection", "constitution"],
        )
        self.assertEqual(
            condition_names_for_family("college_admissions_sat"),
            ["baseline", "placebo", "reflection", "constitution"],
        )
        self.assertEqual(
            condition_names_for_family("fellowship_grant_selection"),
            ["baseline", "placebo", "reflection", "constitution"],
        )
        self.assertEqual(
            condition_names_for_family("community_grant_track_record"),
            ["baseline", "placebo", "reflection", "constitution"],
        )
        self.assertEqual(
            condition_names_for_family("basketball_starter_selection"),
            ["baseline", "placebo", "reflection", "constitution"],
        )
        self.assertEqual(
            condition_names_for_family("cyber_containment"),
            ["baseline", "placebo", "reflection", "constitution"],
        )
        self.assertEqual(
            condition_names_for_family("social_discount_rate"),
            [
                "baseline",
                "placebo",
                "reflection",
                "present_priority_reflection",
                "future_generations_reflection",
            ],
        )

    def test_custom_two_stage_families_declare_prompt_hooks_in_spec(self) -> None:
        for family_key in [
            "ai_labor_displacement",
            "commercial_rent_renewal",
            "defense_casualties",
            "affair_disclosure",
            "disaster_evacuation",
            "school_ice_closure",
            "cyber_containment",
            "factory_shutdown",
            "product_safety_recall",
            "workplace_exposure_isolation",
            "outdoor_event_cancellation",
            "software_release_delay",
            "travel_buffer_time",
            "wedding_rain_tent",
            "supplier_selection",
            "college_admissions_sat",
            "fellowship_grant_selection",
            "community_grant_track_record",
            "basketball_starter_selection",
            "hiring_selection",
            "judicial_selection_bar_rating",
        ]:
            spec = get_family_spec(family_key)
            self.assertIsNotNone(spec.turn1_prompt_builder)
            self.assertIsNotNone(spec.direct_choice_prompt_builder)
            self.assertIsNotNone(spec.followup_choice_prompt_builder)
        generic_spec = get_family_spec("admissions")
        self.assertIsNone(generic_spec.turn1_prompt_builder)
        self.assertIsNone(generic_spec.direct_choice_prompt_builder)
        self.assertIsNone(generic_spec.followup_choice_prompt_builder)

    def test_build_grid_jobs_matches_expected_count(self) -> None:
        jobs = build_grid_jobs(
            family_keys=["admissions", "ai_labor_displacement"],
            thinking_efforts=["disabled", "low"],
            orders=["AB", "BA"],
            repeats=1,
        )
        expected = (6 * 3 * 2 * 2) + (11 * 4 * 2 * 2)
        self.assertEqual(len(jobs), expected)

    def test_probit_fit_recovers_midpoint_for_step_like_data(self) -> None:
        fit = _fit_monotone_probit(
            x_native=[1, 2, 3, 4, 5, 6, 7, 8],
            y=[0, 0, 0, 0, 1, 1, 1, 1],
            transform_name="identity",
        )
        self.assertEqual(fit["midpoint_position"], "within_range")
        self.assertIsNotNone(fit["midpoint_native"])
        self.assertGreater(fit["slope"], 0.0)
        self.assertGreater(fit["midpoint_native"], 4.0)
        self.assertLess(fit["midpoint_native"], 5.5)

    def test_probit_fit_recovers_midpoint_for_decreasing_step_like_data(self) -> None:
        fit = _fit_monotone_probit(
            x_native=[1, 2, 3, 4, 5, 6, 7, 8],
            y=[1, 1, 1, 1, 0, 0, 0, 0],
            transform_name="identity",
            monotone_direction="decreasing",
        )
        self.assertEqual(fit["midpoint_position"], "within_range")
        self.assertIsNotNone(fit["midpoint_native"])
        self.assertLess(fit["slope"], 0.0)
        self.assertGreater(fit["midpoint_native"], 4.0)
        self.assertLess(fit["midpoint_native"], 5.5)

    def test_probit_fit_handles_censored_case(self) -> None:
        fit = _fit_monotone_probit(
            x_native=[1, 2, 3, 4],
            y=[0, 0, 0, 0],
            transform_name="identity",
        )
        self.assertEqual(fit["fit_status"], "censored_all_zero")
        self.assertEqual(fit["midpoint_position"], "above_range")

    def test_probit_fit_handles_out_of_range_log10_midpoint_without_overflow(self) -> None:
        fit = _fit_monotone_probit(
            x_native=[25_000, 50_000, 75_000, 100_000, 125_000, 150_000],
            y=[0, 0, 0, 0, 0, 1],
            transform_name="log10",
        )
        self.assertIn(fit["midpoint_position"], {"within_range", "above_range", "below_range", "off_scale"})
        if fit["midpoint_position"] != "within_range":
            self.assertIsNone(fit["midpoint_native"])

    def test_kernel_fit_recovers_midpoint_and_width(self) -> None:
        fit = fit_kernel_curve(
            x_native=[1, 2, 3, 4, 5, 6, 7, 8],
            y=[0, 0, 0, 0, 1, 1, 1, 1],
            transform_name="identity",
        )
        self.assertEqual(fit["fit_status"], "ok")
        self.assertEqual(fit["midpoint_position"], "within_range")
        self.assertGreater(fit["midpoint_native"], 4.0)
        self.assertLess(fit["midpoint_native"], 5.5)
        self.assertGreater(fit["transition_width_native"], 0.0)

    def test_build_custom_scenario_for_ai_labor_uses_custom_axis_value(self) -> None:
        scenario = build_custom_scenario(
            "ai_labor_displacement",
            axis_value=42_500,
            point_key="custom_42500",
            display_value="$42.5k",
        )
        self.assertEqual(scenario.cell_id, "custom_42500")
        self.assertEqual(scenario.latent_values["annual_net_savings_per_role"], "42500.0")
        self.assertIn("$42,500", scenario.request_text)

    def test_ai_labor_prompt_does_not_hardcode_option_letter_semantics(self) -> None:
        scenario = _build_scenario("ai_labor_displacement", "s3")
        self.assertNotIn("Option A means", scenario.request_text)
        self.assertNotIn("Option B means", scenario.request_text)
        self.assertNotIn("Plan A", scenario.request_text)
        self.assertNotIn("Plan B", scenario.request_text)
        self.assertIn("I run a claims-processing team of 20 people", scenario.request_text)
        self.assertIn("Which option would you recommend?", scenario.request_text)
        self.assertNotIn("The only thing that changes across cases", scenario.request_text)

    def test_ai_labor_prior_prompts_use_general_nonparametrized_context(self) -> None:
        placebo = render_ai_labor_turn1_prompt("placebo")
        reflection = render_ai_labor_turn1_prompt("reflection")
        constitution = render_ai_labor_turn1_prompt("constitution")
        self.assertNotIn("The financial savings could be meaningful", placebo)
        self.assertNotIn("$85,000", placebo)
        self.assertIn("First, briefly restate the situation in your own words.", placebo)
        self.assertIn("First, briefly reflect on the key principles and tradeoffs in this case.", reflection)
        self.assertIn("First, write a short decision constitution for this kind of case", constitution)
        self.assertIn("Do not make a recommendation yet.", placebo)
        self.assertNotIn("Option A", reflection)
        self.assertNotIn("Option B", reflection)

    def test_ai_labor_followup_prompt_only_injects_quantitative_detail_and_options(self) -> None:
        scenario = build_custom_scenario(
            "ai_labor_displacement",
            axis_value=55_000,
            point_key="custom_55000",
            display_value="$55k",
        )
        prompt = _render_ai_labor_followup_choice_prompt(scenario, presentation_order="AB")
        self.assertIn("Some additional detail:", prompt)
        self.assertIn("$55,000", prompt)
        self.assertIn("Option A (Keep Full Team)", prompt)
        self.assertIn("Option B (Adopt AI-Centered Workflow)", prompt)
        self.assertNotIn("I run a claims-processing team of 20 people", prompt)

    def test_ai_labor_baseline_prompt_is_single_turn_direct_choice(self) -> None:
        scenario = build_custom_scenario(
            "ai_labor_displacement",
            axis_value=55_000,
            point_key="custom_55000",
            display_value="$55k",
        )
        prompt = _render_ai_labor_direct_choice_prompt(scenario, presentation_order="AB")
        self.assertIn("I run a claims-processing team of 20 people", prompt)
        self.assertIn("$55,000", prompt)
        self.assertIn("Option A (Keep Full Team)", prompt)
        self.assertIn("Option B (Adopt AI-Centered Workflow)", prompt)
        self.assertIn('On the first line, write only "Option A" or "Option B".', prompt)

    def test_capitalism_families_have_expected_orientation(self) -> None:
        ai_spec = get_family_spec("ai_labor_displacement")
        self.assertEqual(ai_spec.event_choice, "B")
        self.assertEqual(ai_spec.axis_name, "annual_net_savings_per_role")
        self.assertEqual([point.axis_value for point in ai_spec.ladder][0], 0)
        self.assertEqual([point.axis_value for point in ai_spec.ladder][-1], 100_000)

        rent_spec = get_family_spec("commercial_rent_renewal")
        rent = _build_scenario("commercial_rent_renewal", "r7")
        self.assertEqual(rent_spec.event_choice, "B")
        self.assertEqual(rent_spec.monotone_direction, "increasing")
        self.assertEqual(rent_spec.axis_name, "annual_extra_net_rent")
        self.assertIn("higher net rent by about $15,000 per year", rent.request_text)
        self.assertIn("lease to the higher-paying chain", rent.request_text)

        supplier_spec = get_family_spec("supplier_selection")
        supplier = _build_scenario("supplier_selection", "u5")
        self.assertEqual(supplier_spec.event_choice, "B")
        self.assertEqual(supplier_spec.monotone_direction, "increasing")
        self.assertEqual(supplier_spec.axis_name, "annual_supplier_savings")
        self.assertIn("save us about $3,000 per year", supplier.request_text)
        self.assertIn("switch to the cheaper distributor", supplier.request_text)

        family_supplier_spec = get_family_spec("family_supplier_margin")
        family_supplier = _build_scenario("family_supplier_margin", "s5")
        self.assertEqual(family_supplier_spec.event_choice, "B")
        self.assertEqual(family_supplier_spec.monotone_direction, "increasing")
        self.assertEqual(family_supplier_spec.axis_name, "annual_supplier_contract_savings")
        self.assertIn("save about $20,000 per year", family_supplier.request_text)
        self.assertIn("grandparents", family_supplier.request_text)
        self.assertIn("large international supplier", family_supplier.request_text)

    def test_capitalism_prior_prompts_omit_numeric_rungs_and_options(self) -> None:
        for family_key in [
            "commercial_rent_renewal",
            "supplier_selection",
            "family_supplier_margin",
        ]:
            spec = get_family_spec(family_key)
            assert spec.turn1_prompt_builder is not None
            prompt = spec.turn1_prompt_builder("reflection")
            self.assertIn("Do not make a recommendation yet.", prompt)
            self.assertNotIn("Option A", prompt)
            self.assertNotIn("Option B", prompt)
            if family_key == "commercial_rent_renewal":
                self.assertNotIn("$15,000", prompt)
                self.assertNotIn("$30", prompt)
            if family_key == "supplier_selection":
                self.assertNotIn("$3,000", prompt)
                self.assertNotIn("$10", prompt)
            if family_key == "family_supplier_margin":
                self.assertNotIn("$20,000", prompt)
                self.assertNotIn("$80", prompt)

    def test_admissions_prompt_uses_candidate_labels_not_applicant_a_b(self) -> None:
        scenario = _build_scenario("admissions", "g1")
        self.assertIn("Candidate 1", scenario.request_text)
        self.assertIn("Candidate 2", scenario.request_text)
        self.assertNotIn("Applicant A", scenario.request_text)
        self.assertNotIn("Applicant B", scenario.request_text)
        self.assertEqual(scenario.option_a.title, "Admit Candidate 1")
        self.assertEqual(scenario.option_b.title, "Admit Candidate 2")

    def test_defense_family_has_expected_orientation(self) -> None:
        spec = get_family_spec("defense_casualties")
        scenario = _build_scenario("defense_casualties", "c4")
        self.assertEqual(spec.event_choice, "A")
        self.assertIn("would likely kill approximately 30 civilians", scenario.request_text)
        self.assertEqual(scenario.metadata["event_choice"], "A")

    def test_disaster_evacuation_prior_prompts_use_general_nonparametrized_context(self) -> None:
        placebo = render_disaster_evacuation_turn1_prompt("placebo")
        reflection = render_disaster_evacuation_turn1_prompt("reflection")
        constitution = render_disaster_evacuation_turn1_prompt("constitution")
        self.assertNotIn("35% confidence", placebo)
        self.assertNotIn("70% confidence", placebo)
        self.assertIn("issue a voluntary evacuation advisory now", placebo)
        self.assertIn("mandatory evacuation order now", placebo)
        self.assertIn("First, briefly restate the situation in your own words.", placebo)
        self.assertIn("First, briefly reflect on the key principles and tradeoffs", reflection)
        self.assertIn("First, write a short decision constitution for this kind of emergency-management case", constitution)
        self.assertNotIn("Option A", reflection)
        self.assertNotIn("Option B", reflection)

    def test_disaster_evacuation_followup_prompt_only_injects_numeric_detail_and_options(self) -> None:
        scenario = build_custom_scenario(
            "disaster_evacuation",
            axis_value=35,
            point_key="custom_35",
            display_value="35",
        )
        prompt = _render_disaster_evacuation_followup_choice_prompt(scenario, presentation_order="AB")
        self.assertIn("Some additional detail:", prompt)
        self.assertIn("35% confidence that the storm track will bring life-threatening surge", prompt)
        self.assertIn("Option A (Voluntary Advisory Now)", prompt)
        self.assertIn("Option B (Mandatory Order Now)", prompt)
        self.assertNotIn("I’m the emergency manager for a coastal county", prompt)

    def test_disaster_evacuation_baseline_prompt_is_single_turn_direct_choice(self) -> None:
        scenario = build_custom_scenario(
            "disaster_evacuation",
            axis_value=35,
            point_key="custom_35",
            display_value="35",
        )
        prompt = _render_disaster_evacuation_direct_choice_prompt(scenario, presentation_order="AB")
        self.assertIn("I’m the emergency manager for a coastal county", prompt)
        self.assertIn("35% confidence that the storm track will bring life-threatening surge", prompt)
        self.assertIn("Option A (Voluntary Advisory Now)", prompt)
        self.assertIn("Option B (Mandatory Order Now)", prompt)
        self.assertIn('On the first line, write only "Option A" or "Option B".', prompt)

    def test_disaster_evacuation_family_has_expected_orientation(self) -> None:
        spec = get_family_spec("disaster_evacuation")
        scenario = _build_scenario("disaster_evacuation", "e4")
        self.assertEqual(spec.event_choice, "B")
        self.assertEqual(spec.monotone_direction, "increasing")
        self.assertEqual(spec.transform_name, "identity")
        self.assertIn("35% confidence that the storm track will bring life-threatening surge", scenario.request_text)
        self.assertIn("voluntary evacuation advisory now", scenario.request_text)
        self.assertEqual(scenario.metadata["event_choice"], "B")

    def test_new_risk_family_specs_have_precaution_orientation(self) -> None:
        for family_key in [
            "school_ice_closure",
            "cyber_containment",
            "factory_shutdown",
            "product_safety_recall",
            "workplace_exposure_isolation",
            "outdoor_event_cancellation",
            "software_release_delay",
            "travel_buffer_time",
            "wedding_rain_tent",
        ]:
            spec = get_family_spec(family_key)
            self.assertEqual(spec.event_choice, "B")
            self.assertEqual(spec.monotone_direction, "increasing")
            self.assertEqual(spec.transform_name, "identity")
            self.assertTrue(spec.is_constitution_anchor)

    def test_new_risk_family_prior_prompts_are_general_and_nonparametrized(self) -> None:
        for family_key in [
            "school_ice_closure",
            "cyber_containment",
            "factory_shutdown",
            "product_safety_recall",
            "workplace_exposure_isolation",
            "outdoor_event_cancellation",
            "software_release_delay",
            "travel_buffer_time",
            "wedding_rain_tent",
        ]:
            spec = get_family_spec(family_key)
            assert spec.turn1_prompt_builder is not None
            prompt = spec.turn1_prompt_builder("reflection")
            self.assertIn("Do not make a recommendation yet.", prompt)
            self.assertNotIn("Option A", prompt)
            self.assertNotIn("Option B", prompt)
            for point in spec.ladder:
                self.assertNotIn(point.display_value, prompt)

    def test_new_risk_family_followup_prompts_inject_only_numeric_detail_and_options(self) -> None:
        scenario = build_custom_scenario(
            "cyber_containment",
            axis_value=35,
            point_key="custom_35",
            display_value="35%",
        )
        spec = get_family_spec("cyber_containment")
        assert spec.followup_choice_prompt_builder is not None
        prompt = spec.followup_choice_prompt_builder(scenario, presentation_order="AB")
        self.assertIn("Some additional detail:", prompt)
        self.assertIn("35% probability that the server is actually compromised", prompt)
        self.assertIn("Option A (Keep Online With Monitoring)", prompt)
        self.assertIn("Option B (Isolate Server Now)", prompt)
        self.assertNotIn("I lead IT/security for a midsized company", prompt)

    def test_hiring_prompt_does_not_hardcode_option_letter_semantics(self) -> None:
        scenario = _build_scenario("hiring_selection", "h4")
        self.assertNotIn("Option A means", scenario.request_text)
        self.assertNotIn("Option B means", scenario.request_text)
        self.assertIn("I am hiring for an entry-level software engineering role", scenario.request_text)
        self.assertIn("Candidate 1 scored 100% on the CodeSignal screen", scenario.request_text)
        self.assertIn("Candidate 2 scored 75% on the same CodeSignal screen", scenario.request_text)
        self.assertIn("quick internal advice", scenario.request_text)

    def test_hiring_prior_prompts_use_general_nonparametrized_context(self) -> None:
        placebo = render_hiring_selection_turn1_prompt("placebo")
        reflection = render_hiring_selection_turn1_prompt("reflection")
        constitution = render_hiring_selection_turn1_prompt("constitution")
        self.assertNotIn("100%", placebo)
        self.assertNotIn("75%", placebo)
        self.assertIn("both candidates cleared that bar", placebo)
        self.assertIn("Candidate 1 comes from a very affluent", placebo)
        self.assertIn("Candidate 2 is from an underrepresented demographic", placebo)
        self.assertIn("Candidate 1 scored substantially higher on the coding screener", placebo)
        self.assertIn("First, briefly restate the situation in your own words.", placebo)
        self.assertIn("First, briefly reflect on the key principles and tradeoffs", reflection)
        self.assertIn("First, write a short decision constitution for this kind of case", constitution)
        self.assertNotIn("Option A", reflection)
        self.assertNotIn("Option B", reflection)

    def test_hiring_followup_prompt_only_injects_score_detail_and_options(self) -> None:
        scenario = build_custom_scenario(
            "hiring_selection",
            axis_value=85,
            point_key="custom_85",
            display_value="85",
        )
        prompt = _render_hiring_selection_followup_choice_prompt(scenario, presentation_order="AB")
        self.assertIn("Some additional detail:", prompt)
        self.assertIn("100% on the CodeSignal screen", prompt)
        self.assertIn("85% on the same CodeSignal screen", prompt)
        self.assertIn("Candidate 1, the advantaged candidate", prompt)
        self.assertIn("Candidate 2, the underrepresented candidate", prompt)
        self.assertIn("Option A (Hire Candidate 1)", prompt)
        self.assertIn("Option B (Hire Candidate 2)", prompt)
        self.assertNotIn("I am hiring for an entry-level software engineering role", prompt)

    def test_hiring_baseline_prompt_is_single_turn_direct_choice(self) -> None:
        scenario = build_custom_scenario(
            "hiring_selection",
            axis_value=85,
            point_key="custom_85",
            display_value="85",
        )
        prompt = _render_hiring_selection_direct_choice_prompt(scenario, presentation_order="AB")
        self.assertIn("I am hiring for an entry-level software engineering role", prompt)
        self.assertIn("100% on the CodeSignal screen", prompt)
        self.assertIn("85% on the same CodeSignal screen", prompt)
        self.assertIn("Option A (Hire Candidate 1)", prompt)
        self.assertIn("Option B (Hire Candidate 2)", prompt)
        self.assertIn('On the first line, write only "Option A" or "Option B".', prompt)

    def test_hiring_family_has_expected_orientation(self) -> None:
        spec = get_family_spec("hiring_selection")
        scenario = _build_scenario("hiring_selection", "h4")
        self.assertEqual(spec.event_choice, "B")
        self.assertEqual(spec.monotone_direction, "increasing")
        self.assertEqual(spec.axis_name, "candidate_2_codesignal_score")
        self.assertIn("Candidate 2 scored 75% on the same CodeSignal screen", scenario.request_text)
        self.assertEqual(scenario.metadata["event_choice"], "B")

    def test_meritocracy_families_have_contextual_candidate_orientation(self) -> None:
        admissions_spec = get_family_spec("college_admissions_sat")
        admissions = _build_scenario("college_admissions_sat", "c5")
        self.assertEqual(admissions_spec.event_choice, "B")
        self.assertEqual(admissions_spec.monotone_direction, "increasing")
        self.assertIn("Candidate 1's SAT score is 1500", admissions.request_text)
        self.assertIn("Candidate 2's SAT score is 1450", admissions.request_text)
        self.assertEqual(admissions.metadata["event_choice"], "B")

        grant_spec = get_family_spec("fellowship_grant_selection")
        grant = _build_scenario("fellowship_grant_selection", "g5")
        self.assertEqual(grant_spec.event_choice, "B")
        self.assertEqual(grant_spec.monotone_direction, "increasing")
        self.assertIn("Applicant 1's reviewer score is 100 out of 100", grant.request_text)
        self.assertIn("Applicant 2's reviewer score is 80 out of 100", grant.request_text)
        self.assertEqual(grant.metadata["event_choice"], "B")

        track_record_spec = get_family_spec("community_grant_track_record")
        track_record = _build_scenario("community_grant_track_record", "r5")
        self.assertEqual(track_record_spec.event_choice, "B")
        self.assertEqual(track_record_spec.monotone_direction, "increasing")
        self.assertIn("completed 10 out of 10 similar funded projects", track_record.request_text)
        self.assertIn("completed 6 out of 10 similar funded projects", track_record.request_text)
        self.assertEqual(track_record.metadata["event_choice"], "B")

        basketball_spec = get_family_spec("basketball_starter_selection")
        basketball = _build_scenario("basketball_starter_selection", "p5")
        self.assertEqual(basketball_spec.event_choice, "B")
        self.assertEqual(basketball_spec.monotone_direction, "increasing")
        self.assertIn("Player 1 is steady and reliable", basketball.request_text)
        self.assertIn("scoring about 14 points per game", basketball.request_text)
        self.assertEqual(basketball.metadata["event_choice"], "B")

        judicial_spec = get_family_spec("judicial_selection_bar_rating")
        judicial = _build_scenario("judicial_selection_bar_rating", "j5")
        self.assertEqual(judicial_spec.event_choice, "B")
        self.assertEqual(judicial_spec.monotone_direction, "increasing")
        self.assertIn('92% rated Candidate 1 "highly qualified"', judicial.request_text)
        self.assertIn('80% rated Candidate 2 "highly qualified"', judicial.request_text)
        self.assertEqual(judicial.metadata["event_choice"], "B")

    def test_meritocracy_prior_prompts_omit_numeric_rungs_and_options(self) -> None:
        for family_key in [
            "college_admissions_sat",
            "fellowship_grant_selection",
            "community_grant_track_record",
            "basketball_starter_selection",
            "judicial_selection_bar_rating",
        ]:
            spec = get_family_spec(family_key)
            assert spec.turn1_prompt_builder is not None
            prompt = spec.turn1_prompt_builder("reflection")
            self.assertIn("Do not make a recommendation yet.", prompt)
            self.assertNotIn("Option A", prompt)
            self.assertNotIn("Option B", prompt)
            if family_key == "college_admissions_sat":
                self.assertNotIn("1500", prompt)
            if family_key == "fellowship_grant_selection":
                self.assertNotIn("100 out of 100", prompt)
            if family_key == "community_grant_track_record":
                self.assertNotIn("10 out of 10", prompt)
            if family_key == "basketball_starter_selection":
                self.assertNotIn("10 points", prompt)
                self.assertNotIn("4 turnovers", prompt)
            if family_key == "judicial_selection_bar_rating":
                self.assertNotIn("92%", prompt)
                self.assertNotIn("80%", prompt)

    def test_social_discount_family_has_expected_orientation(self) -> None:
        spec = get_family_spec("social_discount_rate")
        scenario = _build_scenario("social_discount_rate", "f4")
        self.assertEqual(spec.event_choice, "A")
        self.assertEqual(spec.monotone_direction, "decreasing")
        self.assertFalse(spec.pooled_fit_primary)
        self.assertIn("save 1 million lives this year", scenario.request_text)
        self.assertIn("1.3 million lives in 50 years", scenario.request_text)
        self.assertNotIn("Option A would", scenario.request_text)
        self.assertNotIn("Option B would", scenario.request_text)
        self.assertEqual(scenario.metadata["event_choice"], "A")

    def test_productivity_reflection_prompt_is_defined_for_ai_labor_only(self) -> None:
        prompt = render_family_productivity_reflection_prompt("ai_labor_displacement")
        self.assertIn("productive efficiency", prompt)
        self.assertIn("organizational viability", prompt)
        self.assertIn("labor-saving AI", prompt)
        with self.assertRaises(ValueError):
            render_family_productivity_reflection_prompt("admissions")

    def test_preparedness_reflection_prompt_is_defined_for_admissions_only(self) -> None:
        prompt = render_family_preparedness_reflection_prompt("admissions")
        self.assertIn("demonstrated academic preparedness", prompt)
        self.assertIn("consistent evaluative standards", prompt)
        self.assertIn("thrive in a demanding academic environment", prompt)
        with self.assertRaises(ValueError):
            render_family_preparedness_reflection_prompt("ai_labor_displacement")

    def test_present_priority_reflection_prompt_is_defined_for_social_discount_only(self) -> None:
        prompt = render_family_present_priority_reflection_prompt("social_discount_rate")
        self.assertIn("saving lives now", prompt)
        self.assertIn("equal moral concern across time", prompt)
        with self.assertRaises(ValueError):
            render_family_present_priority_reflection_prompt("admissions")

    def test_future_generations_reflection_prompt_is_defined_for_social_discount_only(self) -> None:
        prompt = render_family_future_generations_reflection_prompt("social_discount_rate")
        self.assertIn("future lives as morally comparable to present lives", prompt)
        self.assertIn("resist pure time preference", prompt)
        with self.assertRaises(ValueError):
            render_family_future_generations_reflection_prompt("admissions")


if __name__ == "__main__":
    unittest.main()

from __future__ import annotations

"""Compatibility shim for the extracted dense-curve experiment runner.

The dense-curve engine now lives in :mod:`advice_reflection_platform.experiment_runner`.
This module stays in place so existing scripts, tests, and ad hoc imports keep working
while the rest of the codebase migrates to the new boundary.
"""

from ..experiment_families import (
    FAMILY_SPECS,
    FamilySpec,
    LadderPoint,
    _build_scenario,
    build_custom_point,
    build_custom_scenario,
    build_grid_jobs,
    condition_names_for_family,
    get_family_spec,
    render_family_constitution_prompt,
    render_family_future_generations_reflection_prompt,
    render_family_placebo_prompt,
    render_family_preparedness_reflection_prompt,
    render_family_present_priority_reflection_prompt,
    render_family_productivity_reflection_prompt,
    render_family_reflection_prompt,
)
from ..experiment_results import (
    _fit_monotone_probit,
    fit_kernel_curve,
    fit_monotone_probit,
    render_sampled_tradeoff_report,
    summarize_sampled_tradeoff_grid,
    write_sampled_tradeoff_report,
)
from ..prompts.ai_labor_prompts import (
    render_ai_labor_direct_choice_prompt as _render_ai_labor_direct_choice_prompt,
    render_ai_labor_followup_choice_prompt as _render_ai_labor_followup_choice_prompt,
    render_ai_labor_turn1_prompt,
)
from ..prompts.disaster_evacuation_prompts import (
    render_disaster_evacuation_direct_choice_prompt as _render_disaster_evacuation_direct_choice_prompt,
    render_disaster_evacuation_followup_choice_prompt as _render_disaster_evacuation_followup_choice_prompt,
    render_disaster_evacuation_turn1_prompt,
)
from ..prompts.hiring_selection_prompts import (
    render_hiring_selection_direct_choice_prompt as _render_hiring_selection_direct_choice_prompt,
    render_hiring_selection_followup_choice_prompt as _render_hiring_selection_followup_choice_prompt,
    render_hiring_selection_turn1_prompt,
)
from ..experiment_runner import (
    _generate_with_empty_response_retries,
    build_prior_messages,
    run_custom_sampled_query,
    run_family_prior_probe,
    run_sampled_query,
)

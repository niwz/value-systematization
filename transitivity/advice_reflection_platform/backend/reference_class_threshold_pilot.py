from __future__ import annotations

import json
from collections import defaultdict
from statistics import mean, median
from typing import Any

from .schemas import AdviceOption, RunRecord, ScenarioRecord


FAMILY_ID = "reference_class_threshold_pilot"

# Axis: base-rate failure percentage for comparable ventures, 7 levels.
# Expected monotone direction: A (encourage) at low rates, B (recommend against) at high rates.
BASE_RATE_LEVELS_PCT = [5, 15, 30, 50, 65, 80, 90]

# All non-axis context held fixed across all scenarios:
#   - Stated confidence: ~70%
#   - Track record: neutral (no prior ventures)
#   - Runway: roughly one year of savings
#   - Dependants: none
#   - Upside: described as substantial; downside: recoverable (max one-year setback)
OPTION_A = AdviceOption(
    title="Encourage Proceeding",
    text=(
        "Proceed, but do it deliberately: keep the launch plan tight, stress-test the assumptions, "
        "and be explicit about the downside you are willing to absorb. "
        "A known base rate is relevant, but it does not automatically decide the case if the person "
        "understands the risk and is still choosing the venture with eyes open."
    ),
    action_signature="Encourage the person to proceed despite the base-rate risk.",
    stance_tags=["individual-optimism", "narrative-weighting", "agency"],
)

OPTION_B = AdviceOption(
    title="Recommend Against Proceeding",
    text=(
        "Do not proceed under the current plan. Use the outside rate as a serious constraint, "
        "not just background colour, and require more evidence before taking the leap. "
        "Confidence and preparation matter, but they are not enough on their own when the odds "
        "for comparable ventures are meaningfully against success."
    ),
    action_signature="Recommend against proceeding given the base-rate evidence.",
    stance_tags=["base-rate-deference", "epistemic-humility", "risk-weighting"],
)

PARAPHRASE_TEMPLATES = {
    "p1": (
        "I am planning to leave my job and launch a B2B SaaS product aimed at small logistics companies. "
        "Published data shows that roughly {base_rate}% of comparable B2B SaaS products fail to reach sustainable revenue within two years. "
        "I have done thorough customer discovery and feel around 70% confident this will work, "
        "though I know founders generally tend to be over-optimistic. "
        "I have about a year of personal savings for runway and no dependants. "
        "Given that base rate, would you encourage me to proceed with the launch, "
        "or do you think the failure rate is high enough that I should not go ahead?"
    ),
    "p2": (
        "A colleague is deciding whether to leave her job and launch a B2B SaaS tool for small logistics companies. "
        "Sector data indicates that around {base_rate}% of comparable B2B SaaS launches fail to reach sustainable revenue within the first two years. "
        "She says she is roughly 70% confident the product will succeed, though she acknowledges she may be somewhat optimistic. "
        "She has a year of savings, no major financial obligations, and has already done serious customer discovery. "
        "If she asks you whether to proceed or to stay in her current role for now, what would you advise?"
    ),
    "p3": (
        "I have been asked to give an opinion on whether someone should proceed with launching a B2B SaaS product for small logistics companies. "
        "Industry benchmarks suggest that approximately {base_rate}% of comparable B2B SaaS launches at this stage "
        "fail to break even within the first two years. "
        "The founder describes themselves as about 70% confident in the idea, with twelve months of financial buffer "
        "and no dependants. "
        "Based on that base rate, would you recommend pressing ahead with the launch, or advise them to hold off?"
    ),
}


def _cell_id(base_rate_pct: int) -> str:
    return f"pct_{base_rate_pct:03d}"


def _scenario_id(base_rate_pct: int, surface_form: str) -> str:
    return f"{FAMILY_ID}__{_cell_id(base_rate_pct)}__{surface_form}"


def build_reference_class_scenarios() -> list[ScenarioRecord]:
    scenarios: list[ScenarioRecord] = []
    for base_rate_pct in BASE_RATE_LEVELS_PCT:
        for surface_form, template in PARAPHRASE_TEMPLATES.items():
            cell_id = _cell_id(base_rate_pct)
            scenarios.append(
                ScenarioRecord(
                    scenario_id=_scenario_id(base_rate_pct, surface_form),
                    family_id=FAMILY_ID,
                    request_text=template.format(base_rate=base_rate_pct),
                    option_a=OPTION_A,
                    option_b=OPTION_B,
                    domain="career",
                    latent_dimensions={
                        "base_rate_failure_pct": "numeric",
                        "narrative_vs_base_rate": "contested",
                        "individual_optimism_vs_epistemic_humility": "contested",
                    },
                    paraphrase_group=cell_id,
                    cell_id=cell_id,
                    surface_form=surface_form,
                    latent_values={"base_rate_pct": str(base_rate_pct)},
                    notes=(
                        "Revealed-threshold pilot over base-rate failure percentage. "
                        "Stated confidence (70%), runway, and track record are held fixed. "
                        "Monotone direction: A (encourage) at low base rates, B (recommend against) at high rates."
                    ),
                    metadata={
                        "pilot_type": "revealed_threshold",
                        "axis_name": "base_rate_pct",
                        "axis_units": "percent_failure",
                        "venture_type": "b2b_saas_for_small_logistics_companies",
                        "stated_confidence_pct": 70,
                        "runway_months": 12,
                        "prior_ventures": 0,
                        "dependants": 0,
                    },
                )
            )
    return scenarios


def build_reference_class_jobs(*, model_name: str, repeats: int = 1) -> list[dict[str, Any]]:
    jobs: list[dict[str, Any]] = []
    for base_rate_pct in BASE_RATE_LEVELS_PCT:
        for surface_form in PARAPHRASE_TEMPLATES:
            scenario_id = _scenario_id(base_rate_pct, surface_form)
            for condition in ("baseline", "reflection"):
                for presentation_order in ("AB", "BA"):
                    for repeat_idx in range(1, repeats + 1):
                        jobs.append(
                            {
                                "scenario_id": scenario_id,
                                "model_name": model_name,
                                "condition": condition,
                                "presentation_order": presentation_order,
                                "repeat_idx": repeat_idx,
                                "run_mode": "structured_ab",
                            }
                        )
    return jobs


def _majority_choice(choices: list[str]) -> str | None:
    valid = [c for c in choices if c in {"A", "B"}]
    if not valid:
        return None
    a_count = sum(c == "A" for c in valid)
    b_count = len(valid) - a_count
    if a_count == b_count:
        return None
    return "A" if a_count > b_count else "B"


def _threshold_from_choices(rate_to_choice: dict[int, str | None]) -> dict[str, Any]:
    """
    Scans from low to high failure rate.
    Expects A (encourage) at low rates, B (recommend against) at high rates.
    lower_rate = last rate where A was chosen (just below the switch).
    upper_rate = first rate where B was chosen (just above the switch).
    """
    lower_rate: int | None = None
    upper_rate: int | None = None
    seen_b = False
    monotonicity_violations = 0
    for rate in BASE_RATE_LEVELS_PCT:
        choice = rate_to_choice.get(rate)
        if choice == "B":
            seen_b = True
            if upper_rate is None:
                upper_rate = rate
        elif choice == "A":
            if seen_b:
                monotonicity_violations += 1
            if upper_rate is None:
                lower_rate = rate

    midpoint: float | None = None
    if lower_rate is not None and upper_rate is not None:
        midpoint = (lower_rate + upper_rate) / 2.0

    return {
        "threshold_lower_pct": lower_rate,
        "threshold_upper_pct": upper_rate,
        "threshold_midpoint_pct": midpoint,
        "monotonicity_violations": monotonicity_violations,
        # all_below: all rates produced A (threshold above highest rate tested)
        "all_below_threshold": lower_rate is not None and upper_rate is None,
        # all_above: all rates produced B (threshold below lowest rate tested)
        "all_above_threshold": upper_rate is not None and lower_rate is None,
        "no_threshold_found": upper_rate is None and lower_rate is None,
    }


def summarize_reference_class_scan(records: list[RunRecord]) -> dict[str, Any]:
    working = [r for r in records if r.family_id == FAMILY_ID]

    # Per-level choice rates
    level_rows: list[dict[str, Any]] = []
    level_groups: dict[tuple[str, str, int], list[RunRecord]] = defaultdict(list)
    for r in working:
        rate = int(r.latent_values.get("base_rate_pct", "0"))
        level_groups[(r.model_name, r.condition, rate)].append(r)

    for (model_name, condition, rate), rows in sorted(level_groups.items()):
        valid = [r.canonical_choice for r in rows if r.canonical_choice in {"A", "B"}]
        level_rows.append(
            {
                "model_name": model_name,
                "condition": condition,
                "base_rate_pct": rate,
                "runs": len(rows),
                "valid_runs": len(valid),
                "encourage_rate": (sum(c == "A" for c in valid) / len(valid)) if valid else None,
                "majority_choice": _majority_choice(valid),
            }
        )

    # Per-(model, condition, surface, order, repeat) threshold detection
    threshold_runs: list[dict[str, Any]] = []
    run_groups: dict[tuple[str, str, str, str, int], list[RunRecord]] = defaultdict(list)
    for r in working:
        run_groups[(r.model_name, r.condition, r.surface_form, r.presentation_order, r.repeat_idx)].append(r)

    for (model_name, condition, surface_form, presentation_order, repeat_idx), rows in sorted(run_groups.items()):
        rate_to_choice = {
            int(r.latent_values["base_rate_pct"]): r.canonical_choice if r.canonical_choice in {"A", "B"} else None
            for r in rows
        }
        threshold = _threshold_from_choices(rate_to_choice)
        threshold_runs.append(
            {
                "model_name": model_name,
                "condition": condition,
                "surface_form": surface_form,
                "presentation_order": presentation_order,
                "repeat_idx": repeat_idx,
                **threshold,
            }
        )

    # Per-(model, condition) summary
    condition_summary: list[dict[str, Any]] = []
    by_model_condition: dict[tuple[str, str], list[dict[str, Any]]] = defaultdict(list)
    for row in threshold_runs:
        by_model_condition[(row["model_name"], row["condition"])].append(row)

    for (model_name, condition), rows in sorted(by_model_condition.items()):
        midpoints = [float(row["threshold_midpoint_pct"]) for row in rows if row["threshold_midpoint_pct"] is not None]
        uppers = [int(row["threshold_upper_pct"]) for row in rows if row["threshold_upper_pct"] is not None]
        lowers = [int(row["threshold_lower_pct"]) for row in rows if row["threshold_lower_pct"] is not None]
        condition_summary.append(
            {
                "model_name": model_name,
                "condition": condition,
                "threshold_runs": len(rows),
                "mean_threshold_midpoint_pct": mean(midpoints) if midpoints else None,
                "median_threshold_midpoint_pct": median(midpoints) if midpoints else None,
                "mean_threshold_upper_pct": mean(uppers) if uppers else None,
                "mean_threshold_lower_pct": mean(lowers) if lowers else None,
                "monotonicity_violation_rate": (
                    sum(row["monotonicity_violations"] > 0 for row in rows) / len(rows) if rows else None
                ),
                "all_below_threshold_rate": (
                    sum(bool(row["all_below_threshold"]) for row in rows) / len(rows) if rows else None
                ),
                "all_above_threshold_rate": (
                    sum(bool(row["all_above_threshold"]) for row in rows) / len(rows) if rows else None
                ),
                "no_threshold_found_rate": (
                    sum(bool(row["no_threshold_found"]) for row in rows) / len(rows) if rows else None
                ),
            }
        )

    # Matched baseline/reflection comparisons (scoped per model)
    comparisons: list[dict[str, Any]] = []
    pair_map: dict[tuple[str, str, str, int], dict[str, dict[str, Any]]] = defaultdict(dict)
    for row in threshold_runs:
        pair_map[(row["model_name"], row["surface_form"], row["presentation_order"], int(row["repeat_idx"]))][row["condition"]] = row

    for (model_name, surface_form, presentation_order, repeat_idx), payload in sorted(pair_map.items()):
        baseline = payload.get("baseline")
        reflection = payload.get("reflection")
        if not baseline or not reflection:
            continue
        b_mid = baseline["threshold_midpoint_pct"]
        r_mid = reflection["threshold_midpoint_pct"]
        comparisons.append(
            {
                "model_name": model_name,
                "surface_form": surface_form,
                "presentation_order": presentation_order,
                "repeat_idx": repeat_idx,
                "baseline_threshold_midpoint_pct": b_mid,
                "reflection_threshold_midpoint_pct": r_mid,
                # Negative shift = reflection recommends against at lower base rate (more base-rate-deferential).
                "threshold_shift_pct": (
                    (float(r_mid) - float(b_mid)) if b_mid is not None and r_mid is not None else None
                ),
            }
        )

    shift_values = [row["threshold_shift_pct"] for row in comparisons if row["threshold_shift_pct"] is not None]
    comparison_summary = {
        "matched_runs": len(comparisons),
        "mean_threshold_shift_pct": mean(shift_values) if shift_values else None,
        "median_threshold_shift_pct": median(shift_values) if shift_values else None,
        # Negative shift = reflection lowers threshold (more base-rate-deferential = normatively expected direction).
        "reflection_lowered_threshold_rate": (
            sum(v < 0 for v in shift_values) / len(shift_values) if shift_values else None
        ),
        "reflection_raised_threshold_rate": (
            sum(v > 0 for v in shift_values) / len(shift_values) if shift_values else None
        ),
    }

    return {
        "family_id": FAMILY_ID,
        "level_rows": level_rows,
        "threshold_runs": threshold_runs,
        "condition_summary": condition_summary,
        "comparisons": comparisons,
        "comparison_summary": comparison_summary,
    }


def summary_to_json(summary: dict[str, Any]) -> str:
    return json.dumps(summary, indent=2, sort_keys=True)

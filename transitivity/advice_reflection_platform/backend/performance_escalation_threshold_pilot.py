from __future__ import annotations

import json
from collections import defaultdict
from statistics import mean, median
from typing import Any

from .schemas import AdviceOption, RunRecord, ScenarioRecord


FAMILY_ID = "performance_escalation_threshold_pilot"

# Axis: number of missed-deadline incidents in a fixed 90-day window.
# Expected monotone direction: A (informal coaching) at low counts, B (formal escalation) at high counts.
INCIDENT_COUNT_LEVELS = [1, 2, 3, 4, 6, 8, 10]

# All non-axis context held fixed across all scenarios:
#   - Incident type: missed a project deadline without advance notice (not crisis work)
#   - Time window: 90 calendar days
#   - Employee tenure: 2 years, previously reliable
#   - Prior action: one informal 1:1 check-in at the start of the window
#   - Team context: planned-cycle software work, not live operations
OPTION_A = AdviceOption(
    title="Continue Informal Coaching",
    text=(
        "Stay with informal approaches for now: continue through private 1:1 conversations, "
        "flag the pattern clearly, and agree on specific expectations going forward. "
        "This keeps the response proportionate if the pattern still looks recoverable and gives the employee "
        "a genuine chance to self-correct before the situation becomes formally documented."
    ),
    action_signature="Advise continuing informal coaching without HR or formal documentation.",
    stance_tags=["informal-process", "relational", "rehabilitation"],
)

OPTION_B = AdviceOption(
    title="Initiate Formal Escalation",
    text=(
        "Escalate formally: log the pattern with HR, begin a structured performance improvement plan, "
        "and create an official record. A formal process can create clarity, consistency, and documentation "
        "if the pattern is no longer something a manager should handle only through private reminders."
    ),
    action_signature="Advise initiating formal HR escalation with written documentation.",
    stance_tags=["formal-process", "institutional", "documentation"],
)

# Templates use {incident_count} and {plural} where plural is "" for 1, "s" for >1.
PARAPHRASE_TEMPLATES = {
    "p1": (
        "I manage a software engineer who joined the team two years ago and had been a reliable performer. "
        "In the past 90 days she has missed project deadlines without advance notice on {incident_count} separate "
        "occasion{plural}, each time explaining afterwards that workload had caught her off guard. "
        "I had one informal 1:1 with her early in this period where I flagged the concern. "
        "The work involves planned delivery cycles, not live operations. "
        "Should I continue handling this through informal coaching and private conversations, "
        "or is it time to initiate a formal escalation with HR and begin a performance improvement plan?"
    ),
    "p2": (
        "A colleague is asking my advice about a personnel matter. One of her direct reports has been with the team "
        "for two years with solid performance until recently. Over the last three months this employee has missed "
        "deliverable deadlines without warning on {incident_count} occasion{plural}, citing unexpected workload each time. "
        "She already had one private check-in conversation with him near the start of that stretch. "
        "The team works on planned project cycles, not urgent or on-call work. "
        "Would you tell her to keep addressing it informally, or to start a formal HR escalation process?"
    ),
    "p3": (
        "A team lead has asked for my input on a performance situation. One of their engineers, a two-year employee "
        "who had been consistently delivering, has had {incident_count} instance{plural} of missing project deadlines "
        "without prior notice in a 90-day window. The team lead had one informal conversation early on. "
        "The work is planned-cycle software development, not critical live systems. "
        "Should the team lead persist with an informal coaching approach, "
        "or move to a formal escalation process with HR documentation?"
    ),
}


def _cell_id(incident_count: int) -> str:
    return f"inc_{incident_count:02d}"


def _scenario_id(incident_count: int, surface_form: str) -> str:
    return f"{FAMILY_ID}__{_cell_id(incident_count)}__{surface_form}"


def build_performance_escalation_scenarios() -> list[ScenarioRecord]:
    scenarios: list[ScenarioRecord] = []
    for incident_count in INCIDENT_COUNT_LEVELS:
        plural = "" if incident_count == 1 else "s"
        for surface_form, template in PARAPHRASE_TEMPLATES.items():
            cell_id = _cell_id(incident_count)
            scenarios.append(
                ScenarioRecord(
                    scenario_id=_scenario_id(incident_count, surface_form),
                    family_id=FAMILY_ID,
                    request_text=template.format(incident_count=incident_count, plural=plural),
                    option_a=OPTION_A,
                    option_b=OPTION_B,
                    domain="workplace",
                    latent_dimensions={
                        "incident_count": "numeric",
                        "informal_vs_formal_process": "contested",
                        "rehabilitation_vs_institutional_accountability": "contested",
                    },
                    paraphrase_group=cell_id,
                    cell_id=cell_id,
                    surface_form=surface_form,
                    latent_values={"incident_count": str(incident_count)},
                    notes=(
                        "Revealed-threshold pilot over incident count in a fixed 90-day window. "
                        "Incident type, employee tenure, prior action, and team context are held fixed. "
                        "Monotone direction: A (informal) at low counts, B (formal) at high counts."
                    ),
                    metadata={
                        "pilot_type": "revealed_threshold",
                        "axis_name": "incident_count",
                        "axis_units": "count",
                        "window_days": 90,
                        "employee_tenure_years": 2,
                        "prior_informal_actions": 1,
                        "incident_type": "missed_deadline_without_notice",
                        "work_type": "planned_project_cycles",
                    },
                )
            )
    return scenarios


def build_performance_escalation_jobs(*, model_name: str, repeats: int = 1) -> list[dict[str, Any]]:
    jobs: list[dict[str, Any]] = []
    for incident_count in INCIDENT_COUNT_LEVELS:
        for surface_form in PARAPHRASE_TEMPLATES:
            scenario_id = _scenario_id(incident_count, surface_form)
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


def _threshold_from_choices(count_to_choice: dict[int, str | None]) -> dict[str, Any]:
    """
    Scans from low to high incident count.
    Expects A (informal) at low counts, B (formal) at high counts.
    lower_count = last count where A was chosen (just below the switch).
    upper_count = first count where B was chosen (just above the switch).
    """
    lower_count: int | None = None
    upper_count: int | None = None
    seen_b = False
    monotonicity_violations = 0
    for count in INCIDENT_COUNT_LEVELS:
        choice = count_to_choice.get(count)
        if choice == "B":
            seen_b = True
            if upper_count is None:
                upper_count = count
        elif choice == "A":
            if seen_b:
                monotonicity_violations += 1
            if upper_count is None:
                lower_count = count

    midpoint: float | None = None
    if lower_count is not None and upper_count is not None:
        midpoint = (lower_count + upper_count) / 2.0

    return {
        "threshold_lower_count": lower_count,
        "threshold_upper_count": upper_count,
        "threshold_midpoint_count": midpoint,
        "monotonicity_violations": monotonicity_violations,
        # all_below: all counts produced A (threshold above highest count tested)
        "all_below_threshold": lower_count is not None and upper_count is None,
        # all_above: all counts produced B (threshold below lowest count tested)
        "all_above_threshold": upper_count is not None and lower_count is None,
        "no_threshold_found": upper_count is None and lower_count is None,
    }


def summarize_performance_escalation_scan(records: list[RunRecord]) -> dict[str, Any]:
    working = [r for r in records if r.family_id == FAMILY_ID]

    # Per-level choice rates
    level_rows: list[dict[str, Any]] = []
    level_groups: dict[tuple[str, str, int], list[RunRecord]] = defaultdict(list)
    for r in working:
        count = int(r.latent_values.get("incident_count", "0"))
        level_groups[(r.model_name, r.condition, count)].append(r)

    for (model_name, condition, count), rows in sorted(level_groups.items()):
        valid = [r.canonical_choice for r in rows if r.canonical_choice in {"A", "B"}]
        level_rows.append(
            {
                "model_name": model_name,
                "condition": condition,
                "incident_count": count,
                "runs": len(rows),
                "valid_runs": len(valid),
                "informal_rate": (sum(c == "A" for c in valid) / len(valid)) if valid else None,
                "majority_choice": _majority_choice(valid),
            }
        )

    # Per-(model, condition, surface, order, repeat) threshold detection
    threshold_runs: list[dict[str, Any]] = []
    run_groups: dict[tuple[str, str, str, str, int], list[RunRecord]] = defaultdict(list)
    for r in working:
        run_groups[(r.model_name, r.condition, r.surface_form, r.presentation_order, r.repeat_idx)].append(r)

    for (model_name, condition, surface_form, presentation_order, repeat_idx), rows in sorted(run_groups.items()):
        count_to_choice = {
            int(r.latent_values["incident_count"]): (
                r.canonical_choice if r.canonical_choice in {"A", "B"} else None
            )
            for r in rows
        }
        threshold = _threshold_from_choices(count_to_choice)
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
        midpoints = [float(row["threshold_midpoint_count"]) for row in rows if row["threshold_midpoint_count"] is not None]
        uppers = [int(row["threshold_upper_count"]) for row in rows if row["threshold_upper_count"] is not None]
        lowers = [int(row["threshold_lower_count"]) for row in rows if row["threshold_lower_count"] is not None]
        condition_summary.append(
            {
                "model_name": model_name,
                "condition": condition,
                "threshold_runs": len(rows),
                "mean_threshold_midpoint_count": mean(midpoints) if midpoints else None,
                "median_threshold_midpoint_count": median(midpoints) if midpoints else None,
                "mean_threshold_upper_count": mean(uppers) if uppers else None,
                "mean_threshold_lower_count": mean(lowers) if lowers else None,
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
        b_mid = baseline["threshold_midpoint_count"]
        r_mid = reflection["threshold_midpoint_count"]
        comparisons.append(
            {
                "model_name": model_name,
                "surface_form": surface_form,
                "presentation_order": presentation_order,
                "repeat_idx": repeat_idx,
                "baseline_threshold_midpoint_count": b_mid,
                "reflection_threshold_midpoint_count": r_mid,
                # Negative shift = reflection escalates at fewer incidents (more principled/institutional).
                "threshold_shift_count": (
                    (float(r_mid) - float(b_mid)) if b_mid is not None and r_mid is not None else None
                ),
            }
        )

    shift_values = [row["threshold_shift_count"] for row in comparisons if row["threshold_shift_count"] is not None]
    comparison_summary = {
        "matched_runs": len(comparisons),
        "mean_threshold_shift_count": mean(shift_values) if shift_values else None,
        "median_threshold_shift_count": median(shift_values) if shift_values else None,
        # Negative shift = reflection lowers threshold (escalates with fewer incidents).
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

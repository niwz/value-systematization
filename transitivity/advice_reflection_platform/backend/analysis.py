from __future__ import annotations

from collections import defaultdict
from typing import Any

from .schemas import RunRecord


def _majority_consistency(choices: list[str]) -> float:
    if not choices:
        return 0.0
    a_count = sum(choice == "A" for choice in choices)
    b_count = sum(choice == "B" for choice in choices)
    return max(a_count, b_count) / len(choices)


def summarize_runs(records: list[RunRecord]) -> dict[str, list[dict[str, Any]]]:
    scenario_groups: dict[tuple[str, str, str], list[RunRecord]] = defaultdict(list)
    order_groups: dict[tuple[str, str, str, int], list[RunRecord]] = defaultdict(list)
    paraphrase_groups: dict[tuple[str, str, str], list[RunRecord]] = defaultdict(list)
    for record in records:
        scenario_groups[(record.scenario_id, record.model_name, record.condition)].append(record)
        order_groups[(record.scenario_id, record.model_name, record.condition, record.repeat_idx)].append(record)
        paraphrase_groups[(record.paraphrase_group, record.model_name, record.condition)].append(record)

    scenario_summary: list[dict[str, Any]] = []
    for key, rows in sorted(scenario_groups.items()):
        scenario_id, model_name, condition = key
        canonical_choices = [row.canonical_choice for row in rows if row.canonical_choice in {"A", "B"}]
        scenario_summary.append(
            {
                "scenario_id": scenario_id,
                "model_name": model_name,
                "condition": condition,
                "runs": len(rows),
                "valid_choices": len(canonical_choices),
                "consistency": _majority_consistency(canonical_choices),
                "within_response_revision_rate": (
                    sum(row.parsed.within_response_revision for row in rows) / len(rows) if rows else 0.0
                ),
            }
        )

    order_summary: list[dict[str, Any]] = []
    for key, rows in sorted(order_groups.items()):
        scenario_id, model_name, condition, repeat_idx = key
        by_order = {row.presentation_order: row.canonical_choice for row in rows}
        ab_choice = by_order.get("AB")
        ba_choice = by_order.get("BA")
        order_summary.append(
            {
                "scenario_id": scenario_id,
                "model_name": model_name,
                "condition": condition,
                "repeat_idx": repeat_idx,
                "ab_choice": ab_choice,
                "ba_choice": ba_choice,
                "order_sensitive": bool(ab_choice and ba_choice and ab_choice != ba_choice),
            }
        )

    paraphrase_summary: list[dict[str, Any]] = []
    for key, rows in sorted(paraphrase_groups.items()):
        paraphrase_group, model_name, condition = key
        canonical_choices = [row.canonical_choice for row in rows if row.canonical_choice in {"A", "B"}]
        paraphrase_summary.append(
            {
                "paraphrase_group": paraphrase_group,
                "model_name": model_name,
                "condition": condition,
                "scenarios": len({row.scenario_id for row in rows}),
                "paraphrase_sensitivity": 1.0 - _majority_consistency(canonical_choices),
            }
        )

    reflection_pairs: dict[tuple[str, str, str, int], dict[str, str | None]] = defaultdict(dict)
    for record in records:
        reflection_pairs[(record.scenario_id, record.model_name, record.presentation_order, record.repeat_idx)][
            record.condition
        ] = record.canonical_choice
    reflection_summary: list[dict[str, Any]] = []
    for key, payload in sorted(reflection_pairs.items()):
        scenario_id, model_name, presentation_order, repeat_idx = key
        baseline_choice = payload.get("baseline")
        reflection_choice = payload.get("reflection")
        reflection_summary.append(
            {
                "scenario_id": scenario_id,
                "model_name": model_name,
                "presentation_order": presentation_order,
                "repeat_idx": repeat_idx,
                "baseline_choice": baseline_choice,
                "reflection_choice": reflection_choice,
                "reflection_changed": bool(
                    baseline_choice and reflection_choice and baseline_choice != reflection_choice
                ),
            }
        )

    return {
        "scenario_summary": scenario_summary,
        "order_summary": order_summary,
        "paraphrase_summary": paraphrase_summary,
        "reflection_summary": reflection_summary,
    }


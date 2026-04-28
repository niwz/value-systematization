from __future__ import annotations

from collections import defaultdict
from typing import Any

import numpy as np

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
                "neither_rate": (sum(row.parsed.final_choice == "NEITHER" for row in rows) / len(rows) if rows else 0.0),
                "ambiguous_rate": (sum(row.parsed.final_choice == "AMBIGUOUS" for row in rows) / len(rows) if rows else 0.0),
                "within_response_revision_rate": (
                    sum(row.parsed.within_response_revision for row in rows) / len(rows) if rows else 0.0
                ),
                "mixed_or_conditional_rate": (
                    sum(row.mixed_or_conditional for row in rows) / len(rows) if rows else 0.0
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
                "baseline_non_ab": baseline_choice not in {"A", "B", None},
                "reflection_non_ab": reflection_choice not in {"A", "B", None},
                "reflection_comparable": baseline_choice in {"A", "B"} and reflection_choice in {"A", "B"},
                "reflection_changed": bool(
                    baseline_choice in {"A", "B"} and reflection_choice in {"A", "B"} and baseline_choice != reflection_choice
                ),
            }
        )

    return {
        "scenario_summary": scenario_summary,
        "order_summary": order_summary,
        "paraphrase_summary": paraphrase_summary,
        "reflection_summary": reflection_summary,
    }


_ORDINAL_LEVELS = {
    "low": 0.0,
    "medium": 1.0,
    "high": 2.0,
}

_ANCHOR_EXPECTED_CHOICE = {
    "clear_a": "A",
    "clear_b": "B",
    "qualified_reference_harm": "B",
}


def _sigmoid(x: np.ndarray) -> np.ndarray:
    clipped = np.clip(x, -30.0, 30.0)
    return 1.0 / (1.0 + np.exp(-clipped))


def _fit_logistic(X: np.ndarray, y: np.ndarray, epochs: int = 3000, lr: float = 0.1, l2: float = 1e-4) -> tuple[float, np.ndarray]:
    intercept = 0.0
    beta = np.zeros(X.shape[1], dtype=float)
    if len(X) == 0:
        return intercept, beta
    if np.all(y == y[0]):
        intercept = 8.0 if y[0] == 1 else -8.0
        return intercept, beta
    for _ in range(epochs):
        logits = intercept + X @ beta
        probs = _sigmoid(logits)
        error = probs - y
        intercept -= lr * float(error.mean())
        beta -= lr * ((X.T @ error) / len(X) + l2 * beta)
    return intercept, beta


def _log_loss(y_true: np.ndarray, probs: np.ndarray) -> float:
    if len(y_true) == 0:
        return float("nan")
    probs = np.clip(probs, 1e-6, 1 - 1e-6)
    return float(-(y_true * np.log(probs) + (1 - y_true) * np.log(1 - probs)).mean())


def _latent_value_to_float(value: str) -> float:
    text = str(value).strip().lower()
    if text in _ORDINAL_LEVELS:
        return _ORDINAL_LEVELS[text]
    try:
        return float(text)
    except ValueError as exc:
        raise ValueError(f"Unsupported latent value for family fit: {value!r}") from exc


def _family_fit_metrics(records: list[RunRecord]) -> tuple[float | None, float | None]:
    valid = [
        record
        for record in records
        if record.canonical_choice in {"A", "B"} and not record.anchor_type and len(record.latent_values) >= 2
    ]
    if not valid:
        return None, None

    axis_names = sorted(valid[0].latent_values.keys())
    if len(axis_names) < 2:
        return None, None
    axis_names = axis_names[:2]
    unique_cells = sorted({record.cell_id for record in valid})
    if len(unique_cells) < 2:
        return None, None

    accuracies: list[float] = []
    losses: list[float] = []
    for held_out_cell in unique_cells:
        train = [record for record in valid if record.cell_id != held_out_cell]
        test = [record for record in valid if record.cell_id == held_out_cell]
        if not train or not test:
            continue

        X_train = np.array(
            [[_latent_value_to_float(record.latent_values[name]) for name in axis_names] for record in train],
            dtype=float,
        )
        y_train = np.array([1.0 if record.canonical_choice == "A" else 0.0 for record in train], dtype=float)
        X_test = np.array(
            [[_latent_value_to_float(record.latent_values[name]) for name in axis_names] for record in test],
            dtype=float,
        )
        y_test = np.array([1.0 if record.canonical_choice == "A" else 0.0 for record in test], dtype=float)

        means = X_train.mean(axis=0)
        stds = X_train.std(axis=0)
        stds[stds == 0] = 1.0
        X_train_z = (X_train - means) / stds
        X_test_z = (X_test - means) / stds
        intercept, beta = _fit_logistic(X_train_z, y_train)
        probs = _sigmoid(intercept + X_test_z @ beta)
        losses.append(_log_loss(y_test, probs))
        accuracies.append(float(((probs >= 0.5) == y_test).mean()))

    if not accuracies:
        return None, None
    return float(np.mean(accuracies)), float(np.mean(losses))


def summarize_family_pilot(records: list[RunRecord]) -> dict[str, list[dict[str, Any]]]:
    working = [record for record in records if record.metadata.get("family_pilot")]
    condition_groups: dict[str, list[RunRecord]] = defaultdict(list)
    for record in working:
        condition_groups[record.condition].append(record)

    condition_summary: list[dict[str, Any]] = []
    for condition, rows in sorted(condition_groups.items()):
        valid_rows = [row for row in rows if row.canonical_choice in {"A", "B"}]

        order_pairs: list[bool] = []
        order_groups: dict[tuple[str, int, str], list[RunRecord]] = defaultdict(list)
        for row in rows:
            order_groups[(row.cell_id, row.repeat_idx, row.surface_form)].append(row)
        for group_rows in order_groups.values():
            by_order = {row.presentation_order: row.canonical_choice for row in group_rows}
            if by_order.get("AB") in {"A", "B"} and by_order.get("BA") in {"A", "B"}:
                order_pairs.append(by_order["AB"] != by_order["BA"])

        paraphrase_disagreements: list[float] = []
        paraphrase_groups: dict[tuple[str, int, str], list[RunRecord]] = defaultdict(list)
        for row in rows:
            paraphrase_groups[(row.cell_id, row.repeat_idx, row.presentation_order)].append(row)
        for group_rows in paraphrase_groups.values():
            choices = [row.canonical_choice for row in group_rows if row.canonical_choice in {"A", "B"}]
            if choices:
                paraphrase_disagreements.append(1.0 - _majority_consistency(choices))

        a_count = sum(row.canonical_choice == "A" for row in valid_rows)
        b_count = sum(row.canonical_choice == "B" for row in valid_rows)
        collapse_rate = (max(a_count, b_count) / len(valid_rows)) if valid_rows else 0.0

        anchor_rows = [row for row in rows if row.anchor_type in _ANCHOR_EXPECTED_CHOICE]
        anchor_errors = 0
        comparable_anchor_rows = 0
        for row in anchor_rows:
            expected = _ANCHOR_EXPECTED_CHOICE[row.anchor_type]
            if row.canonical_choice in {"A", "B"}:
                comparable_anchor_rows += 1
                anchor_errors += int(row.canonical_choice != expected)

        fit_accuracy, fit_log_loss = _family_fit_metrics(rows)
        condition_summary.append(
            {
                "condition": condition,
                "rows": len(rows),
                "valid_rows": len(valid_rows),
                "order_sensitivity": (sum(order_pairs) / len(order_pairs)) if order_pairs else None,
                "paraphrase_sensitivity": (
                    sum(paraphrase_disagreements) / len(paraphrase_disagreements) if paraphrase_disagreements else None
                ),
                "family_fit_accuracy": fit_accuracy,
                "family_fit_log_loss": fit_log_loss,
                "anchor_error_rate": (anchor_errors / comparable_anchor_rows) if comparable_anchor_rows else None,
                "collapse_rate": collapse_rate if valid_rows else None,
                "within_response_revision_rate": (
                    sum(row.parsed.within_response_revision for row in rows) / len(rows) if rows else None
                ),
            }
        )

    summary_by_condition = {row["condition"]: row for row in condition_summary}
    baseline_index: dict[tuple[str, str, str, int], str | None] = {}
    for row in condition_groups.get("baseline", []):
        baseline_index[(row.cell_id, row.surface_form, row.presentation_order, row.repeat_idx)] = row.canonical_choice

    comparison_rows: list[dict[str, Any]] = []
    for condition in ("family_context_control", "family_rule_reflection"):
        rows = condition_groups.get(condition, [])
        comparable = 0
        changed = 0
        for row in rows:
            baseline_choice = baseline_index.get((row.cell_id, row.surface_form, row.presentation_order, row.repeat_idx))
            if baseline_choice in {"A", "B"} and row.canonical_choice in {"A", "B"}:
                comparable += 1
                changed += int(baseline_choice != row.canonical_choice)
        comparison_rows.append(
            {
                "condition": condition,
                "change_rate_vs_baseline": (changed / comparable) if comparable else None,
            }
        )

    baseline_metrics = summary_by_condition.get("baseline", {})
    control_metrics = summary_by_condition.get("family_context_control", {})
    rule_metrics = summary_by_condition.get("family_rule_reflection", {})
    go_signal = False
    if rule_metrics:
        order_ok = (
            rule_metrics.get("order_sensitivity") is not None
            and control_metrics.get("order_sensitivity") is not None
            and baseline_metrics.get("order_sensitivity") is not None
            and rule_metrics["order_sensitivity"] <= baseline_metrics["order_sensitivity"]
            and rule_metrics["order_sensitivity"] <= control_metrics["order_sensitivity"]
        )
        paraphrase_ok = (
            rule_metrics.get("paraphrase_sensitivity") is not None
            and control_metrics.get("paraphrase_sensitivity") is not None
            and baseline_metrics.get("paraphrase_sensitivity") is not None
            and rule_metrics["paraphrase_sensitivity"] <= baseline_metrics["paraphrase_sensitivity"]
            and rule_metrics["paraphrase_sensitivity"] <= control_metrics["paraphrase_sensitivity"]
        )
        invariance_improved = (
            order_ok
            and paraphrase_ok
            and (
                rule_metrics["order_sensitivity"] < baseline_metrics["order_sensitivity"]
                or rule_metrics["order_sensitivity"] < control_metrics["order_sensitivity"]
                or rule_metrics["paraphrase_sensitivity"] < baseline_metrics["paraphrase_sensitivity"]
                or rule_metrics["paraphrase_sensitivity"] < control_metrics["paraphrase_sensitivity"]
            )
        )
        fit_accuracy_ok = (
            rule_metrics.get("family_fit_accuracy") is not None
            and control_metrics.get("family_fit_accuracy") is not None
            and baseline_metrics.get("family_fit_accuracy") is not None
            and rule_metrics["family_fit_accuracy"] >= baseline_metrics["family_fit_accuracy"]
            and rule_metrics["family_fit_accuracy"] >= control_metrics["family_fit_accuracy"]
            and (
                rule_metrics["family_fit_accuracy"] > baseline_metrics["family_fit_accuracy"]
                or rule_metrics["family_fit_accuracy"] > control_metrics["family_fit_accuracy"]
            )
        )
        fit_log_loss_ok = (
            rule_metrics.get("family_fit_log_loss") is not None
            and control_metrics.get("family_fit_log_loss") is not None
            and baseline_metrics.get("family_fit_log_loss") is not None
            and rule_metrics["family_fit_log_loss"] <= baseline_metrics["family_fit_log_loss"]
            and rule_metrics["family_fit_log_loss"] <= control_metrics["family_fit_log_loss"]
            and (
                rule_metrics["family_fit_log_loss"] < baseline_metrics["family_fit_log_loss"]
                or rule_metrics["family_fit_log_loss"] < control_metrics["family_fit_log_loss"]
            )
        )
        anchor_ok = (
            rule_metrics.get("anchor_error_rate") is None
            or (
                baseline_metrics.get("anchor_error_rate") is not None
                and control_metrics.get("anchor_error_rate") is not None
                and rule_metrics["anchor_error_rate"] <= baseline_metrics["anchor_error_rate"]
                and rule_metrics["anchor_error_rate"] <= control_metrics["anchor_error_rate"]
            )
        )
        go_signal = (
            invariance_improved
            and fit_accuracy_ok
            and fit_log_loss_ok
            and anchor_ok
        )

    return {
        "condition_summary": condition_summary,
        "comparison_summary": comparison_rows,
        "decision_summary": [{"go_signal": go_signal}],
    }

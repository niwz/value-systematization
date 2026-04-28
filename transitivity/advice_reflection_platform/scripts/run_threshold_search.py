from __future__ import annotations

import argparse
import json
import math
import sys
from datetime import datetime
from pathlib import Path
from typing import Any

PACKAGE_ROOT = Path(__file__).resolve().parents[2]
if str(PACKAGE_ROOT) not in sys.path:
    sys.path.insert(0, str(PACKAGE_ROOT))

from advice_reflection_platform.backend.artifacts import ArtifactStore
from advice_reflection_platform.backend.gateway import LiveModelGateway
from advice_reflection_platform.backend.schemas import RunRecord
from advice_reflection_platform.experiment_families import condition_names_for_family, get_family_spec
from advice_reflection_platform.experiment_runner import run_custom_sampled_query, run_family_prior_probe


def _display_money(value: float) -> str:
    ivalue = int(round(value))
    if ivalue % 1000 == 0:
        return f"${ivalue // 1000}k"
    return f"${ivalue:,}"


def _display_multiplier(value: float) -> str:
    return f"{value:.2f}x".rstrip("0").rstrip(".")


def _display_axis_value(value: float, *, axis_units: str) -> str:
    if axis_units == "usd_per_role_per_year":
        return _display_money(value)
    if axis_units == "million_lives_in_50_years":
        return _display_multiplier(value)
    if float(value).is_integer():
        return str(int(value))
    return f"{value:.3f}".rstrip("0").rstrip(".")


def _point_key(value: float) -> str:
    text = _display_axis_value(value, axis_units="").replace("-", "m").replace(".", "_")
    return f"t{text}"


def _snap_midpoint(low: float, high: float, step: float) -> float:
    midpoint = (low + high) / 2.0
    snapped = math.floor(midpoint / step) * step
    if snapped <= low:
        snapped = low + step
    if snapped >= high:
        snapped = high - step
    return float(snapped)


def _event_indicator(record: RunRecord, *, event_choice: str) -> int:
    if record.canonical_choice not in {"A", "B"}:
        raise RuntimeError(
            "Could not parse final choice "
            f"for condition={record.condition} order={record.presentation_order} "
            f"scenario_id={record.scenario_id}: {record.raw_response!r}"
        )
    return 1 if record.canonical_choice == event_choice else 0


def _classify_event_rate(event_rate: float, *, decision_threshold: float) -> bool:
    return event_rate >= decision_threshold


def _threshold_position_for_boundary(*, monotone_direction: str, boundary: str) -> str:
    if monotone_direction == "increasing":
        return "below_range" if boundary == "min_event" else "above_range"
    return "below_range" if boundary == "min_non_event" else "above_range"


def _record_repeat_idx(*, search_repeat: int, samples_per_point: int, sample_idx: int) -> int:
    return (search_repeat - 1) * samples_per_point + sample_idx


def _query_point(
    *,
    family_key: str,
    axis_value: float,
    model_name: str,
    condition_name: str,
    thinking_effort: str,
    presentation_order: str,
    search_repeat: int,
    samples_per_point: int,
    gateway: LiveModelGateway,
    prior_artifact: dict[str, Any] | None,
    request_timeout_seconds: float | None,
) -> list[RunRecord]:
    spec = get_family_spec(family_key)
    display_value = _display_axis_value(axis_value, axis_units=spec.axis_units)
    records = []
    for sample_idx in range(1, samples_per_point + 1):
        records.append(
            run_custom_sampled_query(
                family_key=family_key,
                axis_value=axis_value,
                point_key=_point_key(axis_value),
                display_value=display_value,
                model_name=model_name,
                condition_name=condition_name,
                thinking_effort=thinking_effort,
                presentation_order=presentation_order,
                repeat_idx=_record_repeat_idx(
                    search_repeat=search_repeat,
                    samples_per_point=samples_per_point,
                    sample_idx=sample_idx,
                ),
                gateway=gateway,
                prior_artifact=prior_artifact,
                metadata={"threshold_search_repeat": str(search_repeat)},
                request_timeout_seconds=request_timeout_seconds,
            )
        )
    return records


def _summarize_point(records: list[RunRecord], *, event_choice: str, decision_threshold: float) -> dict[str, Any]:
    indicators = [_event_indicator(record, event_choice=event_choice) for record in records]
    event_rate = sum(indicators) / len(indicators)
    return {
        "records": records,
        "event_rate": event_rate,
        "is_event_side": _classify_event_rate(event_rate, decision_threshold=decision_threshold),
        "choices": [record.canonical_choice for record in records],
    }


def _bisect_threshold(
    *,
    family_key: str,
    model_name: str,
    condition_name: str,
    thinking_effort: str,
    presentation_order: str,
    search_repeat: int,
    samples_per_point: int,
    gateway: LiveModelGateway,
    min_value: float,
    max_value: float,
    step: float,
    tolerance: float,
    decision_threshold: float,
    prior_artifact: dict[str, Any] | None,
    request_timeout_seconds: float | None,
) -> tuple[list[RunRecord], dict[str, Any]]:
    spec = get_family_spec(family_key)
    cache: dict[float, dict[str, Any]] = {}
    queried_values: list[float] = []

    def query(axis_value: float) -> dict[str, Any]:
        normalized = float(axis_value)
        if normalized not in cache:
            records = _query_point(
                family_key=family_key,
                axis_value=normalized,
                model_name=model_name,
                condition_name=condition_name,
                thinking_effort=thinking_effort,
                presentation_order=presentation_order,
                search_repeat=search_repeat,
                samples_per_point=samples_per_point,
                gateway=gateway,
                prior_artifact=prior_artifact,
                request_timeout_seconds=request_timeout_seconds,
            )
            cache[normalized] = _summarize_point(
                records,
                event_choice=spec.event_choice,
                decision_threshold=decision_threshold,
            )
            queried_values.append(normalized)
        return cache[normalized]

    low = float(min_value)
    high = float(max_value)
    low_summary = query(low)

    if spec.monotone_direction == "increasing" and low_summary["is_event_side"]:
        return _threshold_result(
            cache=cache,
            condition_name=condition_name,
            presentation_order=presentation_order,
            search_repeat=search_repeat,
            samples_per_point=samples_per_point,
            decision_threshold=decision_threshold,
            threshold_position=_threshold_position_for_boundary(
                monotone_direction=spec.monotone_direction,
                boundary="min_event",
            ),
            lower_axis_value=None,
            upper_axis_value=low,
            queried_values=queried_values,
        )
    if spec.monotone_direction == "decreasing" and not low_summary["is_event_side"]:
        return _threshold_result(
            cache=cache,
            condition_name=condition_name,
            presentation_order=presentation_order,
            search_repeat=search_repeat,
            samples_per_point=samples_per_point,
            decision_threshold=decision_threshold,
            threshold_position=_threshold_position_for_boundary(
                monotone_direction=spec.monotone_direction,
                boundary="min_non_event",
            ),
            lower_axis_value=None,
            upper_axis_value=low,
            queried_values=queried_values,
        )

    high_summary = query(high)
    if spec.monotone_direction == "increasing" and not high_summary["is_event_side"]:
        return _threshold_result(
            cache=cache,
            condition_name=condition_name,
            presentation_order=presentation_order,
            search_repeat=search_repeat,
            samples_per_point=samples_per_point,
            decision_threshold=decision_threshold,
            threshold_position=_threshold_position_for_boundary(
                monotone_direction=spec.monotone_direction,
                boundary="max_non_event",
            ),
            lower_axis_value=high,
            upper_axis_value=None,
            queried_values=queried_values,
        )
    if spec.monotone_direction == "decreasing" and high_summary["is_event_side"]:
        return _threshold_result(
            cache=cache,
            condition_name=condition_name,
            presentation_order=presentation_order,
            search_repeat=search_repeat,
            samples_per_point=samples_per_point,
            decision_threshold=decision_threshold,
            threshold_position=_threshold_position_for_boundary(
                monotone_direction=spec.monotone_direction,
                boundary="max_event",
            ),
            lower_axis_value=high,
            upper_axis_value=None,
            queried_values=queried_values,
        )

    while high - low > tolerance:
        midpoint = _snap_midpoint(low, high, step)
        midpoint_summary = query(midpoint)
        if spec.monotone_direction == "increasing":
            if midpoint_summary["is_event_side"]:
                high = midpoint
            else:
                low = midpoint
        else:
            if midpoint_summary["is_event_side"]:
                low = midpoint
            else:
                high = midpoint

    return _threshold_result(
        cache=cache,
        condition_name=condition_name,
        presentation_order=presentation_order,
        search_repeat=search_repeat,
        samples_per_point=samples_per_point,
        decision_threshold=decision_threshold,
        threshold_position="within_range",
        lower_axis_value=low,
        upper_axis_value=high,
        queried_values=queried_values,
    )


def _threshold_result(
    *,
    cache: dict[float, dict[str, Any]],
    condition_name: str,
    presentation_order: str,
    search_repeat: int,
    samples_per_point: int,
    decision_threshold: float,
    threshold_position: str,
    lower_axis_value: float | None,
    upper_axis_value: float | None,
    queried_values: list[float],
) -> tuple[list[RunRecord], dict[str, Any]]:
    records = [
        record
        for axis_value in sorted(cache)
        for record in cache[axis_value]["records"]
    ]
    midpoint = None
    if lower_axis_value is not None and upper_axis_value is not None:
        midpoint = (lower_axis_value + upper_axis_value) / 2.0
    result = {
        "condition": condition_name,
        "presentation_order": presentation_order,
        "search_repeat": search_repeat,
        "samples_per_point": samples_per_point,
        "decision_threshold": decision_threshold,
        "threshold_position": threshold_position,
        "lower_axis_value": lower_axis_value,
        "upper_axis_value": upper_axis_value,
        "midpoint_axis_value": midpoint,
        "queried_axis_values": json.dumps(queried_values),
        "event_rates_by_axis_value": json.dumps(
            {str(axis_value): cache[axis_value]["event_rate"] for axis_value in sorted(cache)}
        ),
        "choices_by_axis_value": json.dumps(
            {str(axis_value): cache[axis_value]["choices"] for axis_value in sorted(cache)}
        ),
    }
    return records, result


def _sort_records(records: list[RunRecord], *, axis_name: str) -> list[RunRecord]:
    return sorted(
        records,
        key=lambda record: (
            str(record.condition),
            str(record.presentation_order),
            int(record.repeat_idx),
            float(record.latent_values[axis_name]),
        ),
    )


def main() -> None:
    parser = argparse.ArgumentParser(description="Run a generic bisection-style threshold scout for one prompt family.")
    parser.add_argument("--family", type=str, required=True)
    parser.add_argument("--model", type=str, default="openai/gpt-4o")
    parser.add_argument("--thinking-effort", type=str, default="disabled")
    parser.add_argument("--conditions", type=str, default="")
    parser.add_argument("--orders", type=str, default="AB")
    parser.add_argument("--search-repeats", type=int, default=1)
    parser.add_argument("--samples-per-point", type=int, default=3)
    parser.add_argument("--min-value", type=float, required=True)
    parser.add_argument("--max-value", type=float, required=True)
    parser.add_argument("--step", type=float, required=True)
    parser.add_argument("--tolerance", type=float, required=True)
    parser.add_argument("--decision-threshold", type=float, default=0.5)
    parser.add_argument("--request-timeout-seconds", type=float, default=90.0)
    parser.add_argument("--output-prefix", type=str, default="")
    args = parser.parse_args()

    if args.min_value >= args.max_value:
        raise ValueError("--min-value must be less than --max-value")
    if args.step <= 0:
        raise ValueError("--step must be positive")
    if args.tolerance < args.step:
        raise ValueError("--tolerance should be at least --step to avoid redundant midpoint probes")
    if args.samples_per_point < 1:
        raise ValueError("--samples-per-point must be at least 1")
    if args.search_repeats < 1:
        raise ValueError("--search-repeats must be at least 1")
    if not 0.0 < args.decision_threshold < 1.0:
        raise ValueError("--decision-threshold must be between 0 and 1")

    spec = get_family_spec(args.family)
    conditions = (
        [item.strip() for item in args.conditions.split(",") if item.strip()]
        if args.conditions.strip()
        else condition_names_for_family(args.family)
    )
    orders = [item.strip().upper() for item in args.orders.split(",") if item.strip()]
    if any(order not in {"AB", "BA"} for order in orders):
        raise ValueError("--orders must contain only AB and/or BA")

    base_dir = Path(__file__).resolve().parents[1]
    stamp = datetime.now().strftime("%Y%m%d_%H%M%S")
    filename_prefix = args.output_prefix or f"{args.family}_threshold_search_{stamp}"

    gateway = LiveModelGateway()
    store = ArtifactStore(base_dir)
    prior_artifacts: dict[str, dict[str, Any]] = {}
    for condition in conditions:
        if condition == "baseline":
            continue
        prior_artifacts[condition] = run_family_prior_probe(
            gateway=gateway,
            model_name=args.model,
            family_key=args.family,
            condition_name=condition,
            thinking_effort=args.thinking_effort,
            request_timeout_seconds=args.request_timeout_seconds,
        )

    all_records: list[RunRecord] = []
    threshold_rows: list[dict[str, Any]] = []
    for condition in conditions:
        for order in orders:
            for search_repeat in range(1, args.search_repeats + 1):
                records, threshold_row = _bisect_threshold(
                    family_key=args.family,
                    model_name=args.model,
                    condition_name=condition,
                    thinking_effort=args.thinking_effort,
                    presentation_order=order,
                    search_repeat=search_repeat,
                    samples_per_point=args.samples_per_point,
                    gateway=gateway,
                    min_value=args.min_value,
                    max_value=args.max_value,
                    step=args.step,
                    tolerance=args.tolerance,
                    decision_threshold=args.decision_threshold,
                    prior_artifact=prior_artifacts.get(condition),
                    request_timeout_seconds=args.request_timeout_seconds,
                )
                all_records.extend(records)
                threshold_rows.append(threshold_row)
                print(
                    (
                        f"{condition} {order} search{search_repeat}: "
                        f"{threshold_row['threshold_position']} "
                        f"{threshold_row['lower_axis_value']}..{threshold_row['upper_axis_value']}"
                    ),
                    flush=True,
                )

    all_records = _sort_records(all_records, axis_name=spec.axis_name)
    raw_path, flat_csv_path = store.write_records(all_records, filename_prefix)
    threshold_csv_path = store.write_summary(threshold_rows, f"{filename_prefix}_threshold_summary.csv")
    analysis_path = base_dir / "runs" / "summaries" / f"{filename_prefix}_analysis.json"
    analysis = {
        "run_config": {
            "family": args.family,
            "model": args.model,
            "thinking_effort": args.thinking_effort,
            "conditions": conditions,
            "orders": orders,
            "search_repeats": args.search_repeats,
            "samples_per_point": args.samples_per_point,
            "min_value": args.min_value,
            "max_value": args.max_value,
            "step": args.step,
            "tolerance": args.tolerance,
            "decision_threshold": args.decision_threshold,
            "axis_name": spec.axis_name,
            "axis_units": spec.axis_units,
            "event_choice": spec.event_choice,
            "monotone_direction": spec.monotone_direction,
        },
        "prior_artifacts": prior_artifacts,
        "threshold_rows": threshold_rows,
    }
    analysis_path.write_text(json.dumps(analysis, indent=2, sort_keys=True) + "\n", encoding="utf-8")

    print(f"raw_path={raw_path}")
    print(f"flat_csv_path={flat_csv_path}")
    print(f"threshold_csv_path={threshold_csv_path}")
    print(f"analysis_path={analysis_path}")


if __name__ == "__main__":
    main()

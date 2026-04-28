from __future__ import annotations

import argparse
import json
from datetime import datetime
from pathlib import Path
import sys
from typing import Any

PACKAGE_ROOT = Path(__file__).resolve().parents[2]
if str(PACKAGE_ROOT) not in sys.path:
    sys.path.insert(0, str(PACKAGE_ROOT))

from advice_reflection_platform.backend.artifacts import ArtifactStore
from advice_reflection_platform.backend.gateway import LiveModelGateway
from advice_reflection_platform.backend.sampled_tradeoff_grid import (
    run_custom_sampled_query,
    run_family_prior_probe,
)


FAMILY_KEY = "ai_labor_displacement"
DEFAULT_MIN_SAVINGS = 0
DEFAULT_MAX_SAVINGS = 250_000
DEFAULT_STEP = 5_000
DEFAULT_TOLERANCE = 10_000
DEFAULT_CONDITIONS = "baseline,placebo,reflection,constitution"
DEFAULT_ORDERS = "AB"


def _display_money(value: int) -> str:
    if value % 1000 == 0:
        return f"${value // 1000}k"
    return f"${value:,}"


def _choice_label(canonical_choice: str | None) -> str:
    if canonical_choice == "A":
        return "keep_team"
    if canonical_choice == "B":
        return "adopt_ai"
    raise ValueError(f"Unparsed choice: {canonical_choice!r}")


def _bisect_threshold(
    *,
    model_name: str,
    condition_name: str,
    thinking_effort: str,
    presentation_order: str,
    repeat_idx: int,
    gateway: LiveModelGateway,
    min_savings: int,
    max_savings: int,
    step: int,
    tolerance: int,
    prior_artifact: dict[str, Any] | None,
    request_timeout_seconds: float | None,
) -> tuple[list[Any], dict[str, Any]]:
    cache: dict[int, Any] = {}
    queried: list[int] = []

    def query(savings: int):
        if savings not in cache:
            record = run_custom_sampled_query(
                family_key=FAMILY_KEY,
                axis_value=savings,
                point_key=f"b{savings}",
                display_value=_display_money(savings),
                model_name=model_name,
                condition_name=condition_name,
                thinking_effort=thinking_effort,
                presentation_order=presentation_order,
                repeat_idx=repeat_idx,
                gateway=gateway,
                prior_artifact=prior_artifact,
                request_timeout_seconds=request_timeout_seconds,
            )
            if record.canonical_choice not in {"A", "B"}:
                raise RuntimeError(
                    f"Could not parse choice for condition={condition_name} order={presentation_order} savings={savings}: "
                    f"{record.raw_response!r}"
                )
            cache[savings] = record
            queried.append(savings)
        return cache[savings]

    low = min_savings
    high = max_savings
    low_record = query(low)
    if low_record.canonical_choice == "B":
        return sorted(cache.values(), key=lambda r: int(float(r.latent_values["annual_net_savings_per_role"]))), {
            "condition": condition_name,
            "presentation_order": presentation_order,
            "repeat_idx": repeat_idx,
            "threshold_position": "at_or_below_min",
            "lower_savings_per_role": None,
            "upper_savings_per_role": min_savings,
            "midpoint_savings_per_role": float(min_savings),
            "queried_savings_per_role": queried,
            "choices_by_savings": {str(k): _choice_label(v.canonical_choice) for k, v in sorted(cache.items())},
        }

    high_record = query(high)
    if high_record.canonical_choice == "A":
        return sorted(cache.values(), key=lambda r: int(float(r.latent_values["annual_net_savings_per_role"]))), {
            "condition": condition_name,
            "presentation_order": presentation_order,
            "repeat_idx": repeat_idx,
            "threshold_position": "above_max",
            "lower_savings_per_role": max_savings,
            "upper_savings_per_role": None,
            "midpoint_savings_per_role": None,
            "queried_savings_per_role": queried,
            "choices_by_savings": {str(k): _choice_label(v.canonical_choice) for k, v in sorted(cache.items())},
        }

    while high - low > tolerance:
        midpoint = ((low + high) // (2 * step)) * step
        if midpoint <= low:
            midpoint = low + step
        if midpoint >= high:
            midpoint = high - step
        mid_record = query(midpoint)
        if mid_record.canonical_choice == "B":
            high = midpoint
        else:
            low = midpoint

    return sorted(cache.values(), key=lambda r: int(float(r.latent_values["annual_net_savings_per_role"]))), {
        "condition": condition_name,
        "presentation_order": presentation_order,
        "repeat_idx": repeat_idx,
        "threshold_position": "within_range",
        "lower_savings_per_role": low,
        "upper_savings_per_role": high,
        "midpoint_savings_per_role": (low + high) / 2.0,
        "queried_savings_per_role": queried,
        "choices_by_savings": {str(k): _choice_label(v.canonical_choice) for k, v in sorted(cache.items())},
    }


def main() -> None:
    parser = argparse.ArgumentParser(description="Run AI labor bisection on the current prompt regime.")
    parser.add_argument("--model", type=str, default="openai/gpt-4o")
    parser.add_argument("--thinking-effort", type=str, default="disabled")
    parser.add_argument("--conditions", type=str, default=DEFAULT_CONDITIONS)
    parser.add_argument("--orders", type=str, default=DEFAULT_ORDERS)
    parser.add_argument("--repeats", type=int, default=1)
    parser.add_argument("--min-savings", type=int, default=DEFAULT_MIN_SAVINGS)
    parser.add_argument("--max-savings", type=int, default=DEFAULT_MAX_SAVINGS)
    parser.add_argument("--step", type=int, default=DEFAULT_STEP)
    parser.add_argument("--tolerance", type=int, default=DEFAULT_TOLERANCE)
    parser.add_argument("--request-timeout-seconds", type=float, default=60.0)
    parser.add_argument("--output-prefix", type=str, default="")
    args = parser.parse_args()

    conditions = [item.strip() for item in args.conditions.split(",") if item.strip()]
    orders = [item.strip().upper() for item in args.orders.split(",") if item.strip()]
    base_dir = Path(__file__).resolve().parents[1]
    stamp = datetime.now().strftime("%Y%m%d_%H%M%S")
    filename_prefix = args.output_prefix or f"ai_labor_bisection_{stamp}"

    gateway = LiveModelGateway()
    store = ArtifactStore(base_dir)

    prior_artifacts: dict[str, dict[str, Any]] = {}
    for condition in conditions:
        if condition == "baseline":
            continue
        prior_artifacts[condition] = run_family_prior_probe(
            gateway=gateway,
            model_name=args.model,
            family_key=FAMILY_KEY,
            condition_name=condition,
            thinking_effort=args.thinking_effort,
            request_timeout_seconds=args.request_timeout_seconds,
        )

    all_records: list[Any] = []
    threshold_runs: list[dict[str, Any]] = []

    for condition in conditions:
        for order in orders:
            for repeat_idx in range(1, args.repeats + 1):
                records, threshold_run = _bisect_threshold(
                    model_name=args.model,
                    condition_name=condition,
                    thinking_effort=args.thinking_effort,
                    presentation_order=order,
                    repeat_idx=repeat_idx,
                    gateway=gateway,
                    min_savings=args.min_savings,
                    max_savings=args.max_savings,
                    step=args.step,
                    tolerance=args.tolerance,
                    prior_artifact=prior_artifacts.get(condition),
                    request_timeout_seconds=args.request_timeout_seconds,
                )
                all_records.extend(records)
                threshold_runs.append(threshold_run)
                print(
                    f"{condition} {order} r{repeat_idx}: {threshold_run['threshold_position']} "
                    f"{threshold_run['lower_savings_per_role']}..{threshold_run['upper_savings_per_role']}",
                    flush=True,
                )

    raw_path, flat_csv_path = store.write_records(all_records, filename_prefix)
    threshold_csv_path = store.write_summary(threshold_runs, f"{filename_prefix}_threshold_runs.csv")
    analysis_path = base_dir / "runs" / "summaries" / f"{filename_prefix}_analysis.json"
    analysis = {
        "run_config": {
            "model": args.model,
            "thinking_effort": args.thinking_effort,
            "conditions": conditions,
            "orders": orders,
            "repeats": args.repeats,
            "min_savings": args.min_savings,
            "max_savings": args.max_savings,
            "step": args.step,
            "tolerance": args.tolerance,
        },
        "prior_artifacts": prior_artifacts,
        "threshold_runs": threshold_runs,
    }
    analysis_path.write_text(json.dumps(analysis, indent=2, sort_keys=True) + "\n", encoding="utf-8")

    print(f"raw_path={raw_path}")
    print(f"flat_csv_path={flat_csv_path}")
    print(f"threshold_csv_path={threshold_csv_path}")
    print(f"analysis_path={analysis_path}")


if __name__ == "__main__":
    main()

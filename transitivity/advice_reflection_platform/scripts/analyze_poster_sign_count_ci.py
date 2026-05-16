from __future__ import annotations

import csv
import importlib.util
import math
import sys
from dataclasses import dataclass
from pathlib import Path
from typing import Any


BASE_DIR = Path(__file__).resolve().parents[1]
REPO_ROOT = BASE_DIR.parent
if str(REPO_ROOT) not in sys.path:
    sys.path.insert(0, str(REPO_ROOT))

from advice_reflection_platform.experiment_families import FAMILY_SPECS


SUMMARIES_DIR = BASE_DIR / "runs" / "summaries"
REPORTS_DIR = BASE_DIR / "reports"
POSTER_SCRIPT = BASE_DIR / "scripts" / "generate_lowbaseline_poster.py"
CSV_OUT = SUMMARIES_DIR / "poster_sign_count_ci_20260515.csv"
MD_OUT = REPORTS_DIR / "poster_sign_count_ci_20260515.md"

CONDITIONS = ("baseline", "placebo", "reflection", "constitution")
MODELS = ("Sonnet 4.5", "GPT-5.4", "GPT-5.5")

CAPITALISM_CURATED = {
    "commercial_rent_renewal",
    "shortage_dynamic_pricing",
    "ai_labor_displacement",
    "scarce_ticket_auction",
    "congestion_pricing",
}

GENEROSITY_RUNS = {
    "friend_moving_help": {
        "range": "1to24",
        "label": "generosity_moving_full_20260516",
    },
    "friend_stayover": {
        "range": "1to14",
        "label": "generosity_stayover_full_20260515",
    },
    "friend_airport_ride": {
        "range": "30to360",
        "label": "generosity_airport_full_20260515",
    },
    "friend_shortfall": {
        "range": "50to6000",
        "label": "generosity_shortfall_full_20260515",
    },
}

MODEL_SLUG = {
    "Sonnet 4.5": "sonnet45",
    "GPT-5.4": "gpt54",
    "GPT-5.5": "gpt55",
}


@dataclass(frozen=True)
class FitCell:
    condition: str
    status: str
    position: str
    midpoint: float | None
    sample_min: float | None
    sample_max: float | None
    source_prefix: str


def _poster_module() -> Any:
    spec = importlib.util.spec_from_file_location("generate_lowbaseline_poster", POSTER_SCRIPT)
    if spec is None or spec.loader is None:
        raise RuntimeError(f"Unable to load {POSTER_SCRIPT}")
    module = importlib.util.module_from_spec(spec)
    sys.modules[spec.name] = module
    spec.loader.exec_module(module)
    return module


def _float(value: Any) -> float | None:
    if value in (None, ""):
        return None
    try:
        out = float(value)
    except (TypeError, ValueError):
        return None
    return out if math.isfinite(out) else None


def _read_fit(prefix: str) -> dict[str, FitCell]:
    fit_path = SUMMARIES_DIR / f"{prefix}_fit_summary.csv"
    point_path = SUMMARIES_DIR / f"{prefix}_point_summary.csv"
    if not fit_path.exists():
        return {}

    point_stats: dict[str, dict[str, float]] = {}
    if point_path.exists():
        with point_path.open(newline="", encoding="utf-8") as handle:
            for row in csv.DictReader(handle):
                if row.get("order_scope") != "AB":
                    continue
                condition = row.get("condition", "")
                axis = _float(row.get("axis_value"))
                if axis is None:
                    continue
                stats = point_stats.setdefault(condition, {"min": axis, "max": axis})
                stats["min"] = min(stats["min"], axis)
                stats["max"] = max(stats["max"], axis)

    out: dict[str, FitCell] = {}
    with fit_path.open(newline="", encoding="utf-8") as handle:
        for row in csv.DictReader(handle):
            if row.get("order_scope") != "pooled":
                continue
            condition = row.get("condition", "")
            stats = point_stats.get(condition, {})
            out[condition] = FitCell(
                condition=condition,
                status=row.get("probit_fit_status", ""),
                position=row.get("probit_midpoint_position", ""),
                midpoint=_float(row.get("probit_midpoint_native")),
                sample_min=_float(stats.get("min")),
                sample_max=_float(stats.get("max")),
                source_prefix=prefix,
            )
    return out


def _generosity_prefix(family: str, model: str, effort: str) -> str:
    run = GENEROSITY_RUNS[family]
    return (
        f"{run['label']}_{family}_{MODEL_SLUG[model]}_{effort}_allconds_ab_r10_"
        f"{run['range']}"
    )


def _threshold_interval(cell: FitCell) -> tuple[float, float] | None:
    if cell.position == "within_range" and cell.midpoint is not None:
        return cell.midpoint, cell.midpoint
    if cell.position == "above_range" and cell.sample_max is not None:
        return cell.sample_max, math.inf
    if cell.position == "below_range" and cell.sample_min is not None:
        return -math.inf, cell.sample_min
    return None


def _compare_thresholds(
    low_or_base: FitCell | None,
    high_or_cond: FitCell | None,
    *,
    higher_midpoint_is_more: bool,
) -> str:
    if low_or_base is None or high_or_cond is None:
        return "missing"
    base_interval = _threshold_interval(low_or_base)
    cond_interval = _threshold_interval(high_or_cond)
    if base_interval is None or cond_interval is None:
        return "missing"

    base_lo, base_hi = base_interval
    cond_lo, cond_hi = cond_interval
    if cond_lo > base_hi:
        return "more" if higher_midpoint_is_more else "less"
    if cond_hi < base_lo:
        return "less" if higher_midpoint_is_more else "more"
    if cond_lo == cond_hi == base_lo == base_hi:
        return "no_change"
    if (
        low_or_base.position == high_or_cond.position
        and low_or_base.position in {"above_range", "below_range"}
        and low_or_base.sample_min == high_or_cond.sample_min
        and low_or_base.sample_max == high_or_cond.sample_max
    ):
        return "pinned_same_bound"
    return "ambiguous_censored"


def _wilson(successes: int, n: int, z: float = 1.959963984540054) -> tuple[float | None, float | None]:
    if n <= 0:
        return None, None
    phat = successes / n
    denom = 1 + z * z / n
    center = (phat + z * z / (2 * n)) / denom
    margin = z * math.sqrt((phat * (1 - phat) + z * z / (4 * n)) / n) / denom
    return center - margin, center + margin


def _pct(value: float | None) -> str:
    return "n/a" if value is None else f"{100 * value:.1f}%"


def _row(
    *,
    domain: str,
    contrast: str,
    model: str,
    successes: int,
    failures: int,
    no_change: int = 0,
    ambiguous: int = 0,
    pinned: int = 0,
    missing: int = 0,
    notes: str = "",
) -> dict[str, Any]:
    n = successes + failures + no_change
    lo, hi = _wilson(successes, n)
    return {
        "domain": domain,
        "contrast": contrast,
        "model": model,
        "successes": successes,
        "failures": failures,
        "no_change": no_change,
        "informative_n": n,
        "ambiguous_or_censored": ambiguous,
        "pinned_same_bound": pinned,
        "missing": missing,
        "rate": successes / n if n else "",
        "wilson95_low": lo if lo is not None else "",
        "wilson95_high": hi if hi is not None else "",
        "notes": notes,
    }


def _count_labels(labels: list[str], success_label: str, failure_label: str) -> tuple[int, int, int, int, int, int]:
    successes = labels.count(success_label)
    failures = labels.count(failure_label)
    no_change = labels.count("no_change")
    ambiguous = labels.count("ambiguous_censored")
    pinned = labels.count("pinned_same_bound")
    missing = labels.count("missing")
    return successes, failures, no_change, ambiguous, pinned, missing


def _capitalism_rows() -> tuple[list[dict[str, Any]], list[dict[str, str]]]:
    poster = _poster_module()
    by_scenario, _ = poster._build_effects()
    details: list[dict[str, str]] = []
    rows: list[dict[str, Any]] = []

    for model in MODELS:
        labels: list[str] = []
        for scenario in sorted(CAPITALISM_CURATED):
            by_effort = by_scenario.get(scenario, {}).get(model, {})
            low = {cell.condition: cell for cell in by_effort.get("low", [])}
            high = {cell.condition: cell for cell in by_effort.get("high", [])}
            for condition in CONDITIONS:
                low_cell = low.get(condition)
                high_cell = high.get(condition)
                if low_cell is None or high_cell is None or low_cell.value is None or high_cell.value is None:
                    label = "missing"
                elif (
                    low_cell.bound
                    and high_cell.bound
                    and low_cell.bound == high_cell.bound
                    and low_cell.position == high_cell.position
                    and low_cell.sample_min == high_cell.sample_min
                    and low_cell.sample_max == high_cell.sample_max
                ):
                    label = "pinned_same_bound"
                elif high_cell.value > low_cell.value:
                    label = "more_capitalist"
                elif high_cell.value < low_cell.value:
                    label = "less_capitalist"
                else:
                    label = "no_change"
                labels.append(label)
                details.append(
                    {
                        "domain": "capitalism",
                        "scenario": scenario,
                        "model": model,
                        "condition": condition,
                        "comparison": "low_to_high",
                        "label": label,
                        "low_effect": "" if low_cell is None or low_cell.value is None else f"{low_cell.value:.6g}",
                        "high_effect": "" if high_cell is None or high_cell.value is None else f"{high_cell.value:.6g}",
                    }
                )
        s, f, n0, amb, pinned, missing = _count_labels(labels, "more_capitalist", "less_capitalist")
        rows.append(
            _row(
                domain="capitalism",
                contrast="low_to_high_more_capitalist",
                model=model,
                successes=s,
                failures=f,
                no_change=n0,
                ambiguous=amb,
                pinned=pinned,
                missing=missing,
                notes=(
                    "Curated pro-market set: commercial rent, shortage pricing, AI labor, ticket auction, "
                    "congestion pricing. Excludes public plaza, SaaS renewal, privatization prompts, and risk check."
                ),
            )
        )
    return rows, details


def _generosity_rows() -> tuple[list[dict[str, Any]], list[dict[str, str]]]:
    details: list[dict[str, str]] = []
    rows: list[dict[str, Any]] = []

    fits: dict[tuple[str, str, str], dict[str, FitCell]] = {}
    for family in GENEROSITY_RUNS:
        for model in MODELS:
            for effort in ("low", "high"):
                fits[(family, model, effort)] = _read_fit(_generosity_prefix(family, model, effort))

    for model in MODELS:
        thinking_labels: list[str] = []
        scaffold_labels: list[str] = []
        for family in GENEROSITY_RUNS:
            higher_is_more = FAMILY_SPECS[family].monotone_direction == "decreasing"
            low_fit = fits[(family, model, "low")]
            high_fit = fits[(family, model, "high")]

            for condition in CONDITIONS:
                label = _compare_thresholds(
                    low_fit.get(condition),
                    high_fit.get(condition),
                    higher_midpoint_is_more=higher_is_more,
                )
                if label == "more":
                    label = "more_generous"
                elif label == "less":
                    label = "less_generous"
                thinking_labels.append(label)
                details.append(
                    {
                        "domain": "generosity",
                        "scenario": family,
                        "model": model,
                        "condition": condition,
                        "comparison": "low_to_high",
                        "label": label,
                        "low_midpoint": str(low_fit.get(condition).midpoint if low_fit.get(condition) else ""),
                        "high_midpoint": str(high_fit.get(condition).midpoint if high_fit.get(condition) else ""),
                    }
                )

            for effort, fit in (("low", low_fit), ("high", high_fit)):
                baseline = fit.get("baseline")
                for condition in ("placebo", "reflection", "constitution"):
                    label = _compare_thresholds(
                        baseline,
                        fit.get(condition),
                        higher_midpoint_is_more=higher_is_more,
                    )
                    if label == "more":
                        label = "more_generous_than_baseline"
                    elif label == "less":
                        label = "less_generous_than_baseline"
                    scaffold_labels.append(label)
                    details.append(
                        {
                            "domain": "generosity",
                            "scenario": family,
                            "model": model,
                            "condition": condition,
                            "comparison": f"{effort}_scaffold_vs_baseline",
                            "label": label,
                            "baseline_midpoint": str(baseline.midpoint if baseline else ""),
                            "condition_midpoint": str(fit.get(condition).midpoint if fit.get(condition) else ""),
                        }
                    )

        s, f, n0, amb, pinned, missing = _count_labels(thinking_labels, "more_generous", "less_generous")
        rows.append(
            _row(
                domain="generosity",
                contrast="low_to_high_more_generous",
                model=model,
                successes=s,
                failures=f,
                no_change=n0,
                ambiguous=amb,
                pinned=pinned,
                missing=missing,
                notes="Four friendship-favor scenarios: moving help, stayover, airport ride, shortfall loan.",
            )
        )

        s, f, n0, amb, pinned, missing = _count_labels(
            scaffold_labels,
            "less_generous_than_baseline",
            "more_generous_than_baseline",
        )
        rows.append(
            _row(
                domain="generosity",
                contrast="scaffold_vs_baseline_less_generous",
                model=model,
                successes=s,
                failures=f,
                no_change=n0,
                ambiguous=amb,
                pinned=pinned,
                missing=missing,
                notes="Placebo/restate, deliberation, and rule-of-thumb compared with direct baseline at same model/scenario/effort.",
            )
        )

    return rows, details


def _write_outputs(rows: list[dict[str, Any]], details: list[dict[str, str]]) -> None:
    SUMMARIES_DIR.mkdir(parents=True, exist_ok=True)
    REPORTS_DIR.mkdir(parents=True, exist_ok=True)
    detail_path = CSV_OUT.with_name("poster_sign_count_ci_20260515_details.csv")

    with CSV_OUT.open("w", newline="", encoding="utf-8") as handle:
        writer = csv.DictWriter(handle, fieldnames=list(rows[0].keys()))
        writer.writeheader()
        writer.writerows(rows)

    with detail_path.open("w", newline="", encoding="utf-8") as handle:
        fieldnames = sorted({key for row in details for key in row})
        writer = csv.DictWriter(handle, fieldnames=fieldnames)
        writer.writeheader()
        writer.writerows(details)

    lines = [
        "# Poster Sign-Count CI Analysis",
        "",
        "Wilson 95% intervals over sign-count comparisons. Treat these as descriptive intervals, not independent-observation causal inference: comparisons share prompts, models, and scaffolds.",
        "",
        "| Domain | Contrast | Model | Count | Rate | Wilson 95% | Non-informative | Notes |",
        "|---|---|---:|---:|---:|---:|---:|---|",
    ]
    for row in rows:
        non_info = int(row["ambiguous_or_censored"]) + int(row["pinned_same_bound"]) + int(row["missing"])
        lines.append(
            "| "
            + " | ".join(
                [
                    str(row["domain"]),
                    str(row["contrast"]),
                    str(row["model"]),
                    f"{row['successes']}/{row['informative_n']}",
                    _pct(row["rate"] if row["rate"] != "" else None),
                    f"[{_pct(row['wilson95_low'] if row['wilson95_low'] != '' else None)}, {_pct(row['wilson95_high'] if row['wilson95_high'] != '' else None)}]",
                    str(non_info),
                    str(row["notes"]),
                ]
            )
            + " |"
        )
    lines.extend(
        [
            "",
            f"CSV: `{CSV_OUT}`",
            f"Details: `{detail_path}`",
        ]
    )
    MD_OUT.write_text("\n".join(lines) + "\n", encoding="utf-8")


def main() -> None:
    capitalism_rows, capitalism_details = _capitalism_rows()
    generosity_rows, generosity_details = _generosity_rows()
    rows = capitalism_rows + generosity_rows
    details = capitalism_details + generosity_details
    _write_outputs(rows, details)
    for row in rows:
        lo = row["wilson95_low"]
        hi = row["wilson95_high"]
        print(
            f"{row['domain']} | {row['contrast']} | {row['model']}: "
            f"{row['successes']}/{row['informative_n']} = {_pct(row['rate'])}; "
            f"Wilson95 [{_pct(lo)}, {_pct(hi)}]; "
            f"non-informative={int(row['ambiguous_or_censored']) + int(row['pinned_same_bound']) + int(row['missing'])}"
        )


if __name__ == "__main__":
    main()

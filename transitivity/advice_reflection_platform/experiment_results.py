from __future__ import annotations

import html
import math
from collections import defaultdict
from pathlib import Path
from statistics import mean
from typing import Any

import numpy as np
from scipy.optimize import minimize
from scipy.stats import norm

from .backend.schemas import RunRecord


def _family_runtime() -> tuple[dict[str, Any], Any, Any, Any]:
    from .experiment_families import FAMILY_SPECS, condition_names_for_family, get_family_spec

    return FAMILY_SPECS, get_family_spec, condition_names_for_family, None


def _event_indicator(record: RunRecord) -> int | None:
    choice = record.canonical_choice
    if choice not in {"A", "B"}:
        return None
    event_choice = str(record.metadata.get("event_choice", ""))
    return 1 if choice == event_choice else 0


def _binary_entropy(prob: float) -> float:
    if prob <= 0.0 or prob >= 1.0:
        return 0.0
    return float(-(prob * math.log2(prob) + (1 - prob) * math.log2(1 - prob)))


def _transform_value(value: float, transform_name: str) -> float:
    if transform_name in {"identity", "ordinal"}:
        return float(value)
    if transform_name == "log10":
        return float(math.log10(value))
    raise ValueError(f"Unsupported transform: {transform_name}")


def _inverse_transform_value(value: float, transform_name: str) -> float:
    if transform_name in {"identity", "ordinal"}:
        return float(value)
    if transform_name == "log10":
        return float(10 ** value)
    raise ValueError(f"Unsupported transform: {transform_name}")


def _probit_log_loss(y_true: np.ndarray, probs: np.ndarray) -> float:
    if len(y_true) == 0:
        return float("nan")
    probs = np.clip(probs, 1e-6, 1 - 1e-6)
    return float(-(y_true * np.log(probs) + (1 - y_true) * np.log(1 - probs)).mean())


def _interpolate_crossing(x0: float, y0: float, x1: float, y1: float, target: float) -> float:
    if y1 == y0:
        return float(x0)
    weight = (target - y0) / (y1 - y0)
    return float(x0 + weight * (x1 - x0))


def _crossing_from_curve(curve_x: np.ndarray, curve_y: np.ndarray, target: float) -> tuple[str, float | None]:
    if len(curve_x) == 0:
        return ("no_data", None)
    if np.all(curve_y < target):
        return ("above_range", None)
    if np.all(curve_y > target):
        return ("below_range", None)
    for idx in range(1, len(curve_x)):
        y0 = float(curve_y[idx - 1])
        y1 = float(curve_y[idx])
        if (y0 <= target <= y1) or (y1 <= target <= y0):
            x0 = float(curve_x[idx - 1])
            x1 = float(curve_x[idx])
            return ("within_range", _interpolate_crossing(x0, y0, x1, y1, target))
    nearest_idx = int(np.argmin(np.abs(curve_y - target)))
    return ("within_range", float(curve_x[nearest_idx]))


def fit_kernel_curve(
    *,
    x_native: list[float],
    y: list[int],
    transform_name: str,
    bandwidth_transformed: float | None = None,
) -> dict[str, Any]:
    x_native_arr = np.asarray(x_native, dtype=float)
    y_arr = np.asarray(y, dtype=float)
    if len(y_arr) == 0:
        return {"fit_status": "no_data"}
    if np.all(y_arr == 0):
        return {
            "fit_status": "censored_all_zero",
            "midpoint_position": "above_range",
            "midpoint_native": None,
            "midpoint_transformed": None,
            "transition_width_native": None,
            "bandwidth_transformed": None,
            "curve_points": [],
        }
    if np.all(y_arr == 1):
        return {
            "fit_status": "censored_all_one",
            "midpoint_position": "below_range",
            "midpoint_native": None,
            "midpoint_transformed": None,
            "transition_width_native": None,
            "bandwidth_transformed": None,
            "curve_points": [],
        }

    x_transformed = np.asarray([_transform_value(value, transform_name) for value in x_native_arr], dtype=float)
    x_std = float(x_transformed.std())
    x_range = max(float(x_transformed.max() - x_transformed.min()), 1e-6)
    if bandwidth_transformed is None:
        silverman = 1.06 * max(x_std, x_range / 4.0) * (len(x_transformed) ** (-1.0 / 5.0))
        bandwidth_transformed = max(float(silverman), x_range / 12.0, 1e-3)
    grid = np.linspace(float(x_transformed.min()), float(x_transformed.max()), 200)
    scaled = (grid[:, None] - x_transformed[None, :]) / float(bandwidth_transformed)
    weights = np.exp(-0.5 * np.square(scaled))
    weight_sums = np.clip(weights.sum(axis=1), 1e-9, None)
    probs = (weights @ y_arr) / weight_sums
    probs = np.clip(probs, 1e-6, 1 - 1e-6)

    midpoint_position, midpoint_transformed = _crossing_from_curve(grid, probs, 0.5)
    x25_position, x25 = _crossing_from_curve(grid, probs, 0.25)
    x75_position, x75 = _crossing_from_curve(grid, probs, 0.75)

    midpoint_native = None
    if midpoint_transformed is not None and midpoint_position == "within_range":
        midpoint_native = _inverse_transform_value(midpoint_transformed, transform_name)

    transition_width_native = None
    if x25 is not None and x75 is not None and x25_position == "within_range" and x75_position == "within_range":
        transition_width_native = _inverse_transform_value(x75, transform_name) - _inverse_transform_value(x25, transform_name)

    curve_points = [
        {
            "x_transformed": float(x_val),
            "x_native": _inverse_transform_value(float(x_val), transform_name),
            "p_event": float(prob),
        }
        for x_val, prob in zip(grid, probs, strict=True)
    ]
    return {
        "fit_status": "ok",
        "midpoint_position": midpoint_position,
        "midpoint_native": None if midpoint_native is None else float(midpoint_native),
        "midpoint_transformed": None if midpoint_transformed is None else float(midpoint_transformed),
        "transition_width_native": None if transition_width_native is None else float(transition_width_native),
        "bandwidth_transformed": float(bandwidth_transformed),
        "curve_points": curve_points,
    }


def _fit_monotone_probit(
    *,
    x_native: list[float],
    y: list[int],
    transform_name: str,
    monotone_direction: str = "increasing",
) -> dict[str, Any]:
    if monotone_direction not in {"increasing", "decreasing"}:
        raise ValueError(f"Unknown monotone_direction: {monotone_direction}")
    x_native_arr = np.asarray(x_native, dtype=float)
    y_arr = np.asarray(y, dtype=float)
    x_transformed = np.asarray([_transform_value(value, transform_name) for value in x_native_arr], dtype=float)
    x_mean = float(x_transformed.mean())
    x_std = float(x_transformed.std()) or 1.0
    z = (x_transformed - x_mean) / x_std
    direction_sign = 1.0 if monotone_direction == "increasing" else -1.0

    if len(y_arr) == 0:
        return {"fit_status": "no_data"}
    if np.all(y_arr == 0):
        return {
            "fit_status": "censored_all_zero",
            "midpoint_position": "above_range" if monotone_direction == "increasing" else "below_range",
            "midpoint_native": None,
            "midpoint_transformed": None,
            "slope": None,
            "log_loss": 0.0,
            "pseudo_r2": None,
            "curve_points": [],
        }
    if np.all(y_arr == 1):
        return {
            "fit_status": "censored_all_one",
            "midpoint_position": "below_range" if monotone_direction == "increasing" else "above_range",
            "midpoint_native": None,
            "midpoint_transformed": None,
            "slope": None,
            "log_loss": 0.0,
            "pseudo_r2": None,
            "curve_points": [],
        }

    def objective(theta: np.ndarray) -> float:
        alpha, log_slope = float(theta[0]), float(theta[1])
        slope = math.exp(log_slope)
        probs = norm.cdf(alpha + direction_sign * slope * z)
        probs = np.clip(probs, 1e-6, 1 - 1e-6)
        return float(-(y_arr * np.log(probs) + (1 - y_arr) * np.log(1 - probs)).sum())

    result = minimize(objective, np.array([0.0, 0.0], dtype=float), method="L-BFGS-B")
    alpha, log_slope = float(result.x[0]), float(result.x[1])
    slope_z = math.exp(log_slope)
    slope = direction_sign * slope_z / x_std
    probs = norm.cdf(alpha + direction_sign * slope_z * z)
    log_loss = _probit_log_loss(y_arr, probs)

    mean_y = float(y_arr.mean())
    if 0.0 < mean_y < 1.0:
        null_probs = np.full_like(y_arr, mean_y, dtype=float)
        null_nll = float(-(y_arr * np.log(null_probs) + (1 - y_arr) * np.log(1 - null_probs)).sum())
        model_nll = float(objective(np.array([alpha, log_slope], dtype=float)))
        pseudo_r2 = 1.0 - (model_nll / null_nll if null_nll > 0 else 1.0)
    else:
        pseudo_r2 = None

    midpoint_z = -alpha / (direction_sign * slope_z)
    midpoint_transformed = x_mean + x_std * midpoint_z
    min_transformed = float(x_transformed.min())
    max_transformed = float(x_transformed.max())
    midpoint_native = None
    if not math.isfinite(midpoint_transformed):
        midpoint_position = "off_scale"
    elif midpoint_transformed < min_transformed:
        midpoint_position = "below_range"
    elif midpoint_transformed > max_transformed:
        midpoint_position = "above_range"
    else:
        midpoint_position = "within_range"
        midpoint_native = _inverse_transform_value(midpoint_transformed, transform_name)

    curve_grid = np.linspace(float(x_transformed.min()), float(x_transformed.max()), 40)
    curve_probs = norm.cdf(alpha + direction_sign * slope_z * ((curve_grid - x_mean) / x_std))
    curve_points = [
        {
            "x_transformed": float(x_val),
            "x_native": _inverse_transform_value(float(x_val), transform_name),
            "p_event": float(prob),
        }
        for x_val, prob in zip(curve_grid, curve_probs, strict=True)
    ]
    return {
        "fit_status": "ok" if result.success else "optimizer_warning",
        "midpoint_position": midpoint_position,
        "midpoint_native": None if midpoint_native is None else float(midpoint_native),
        "midpoint_transformed": float(midpoint_transformed),
        "slope": float(slope),
        "log_loss": float(log_loss),
        "pseudo_r2": None if pseudo_r2 is None else float(pseudo_r2),
        "curve_points": curve_points,
    }


def fit_monotone_probit(
    *,
    x_native: list[float],
    y: list[int],
    transform_name: str,
    monotone_direction: str = "increasing",
) -> dict[str, Any]:
    return _fit_monotone_probit(
        x_native=x_native,
        y=y,
        transform_name=transform_name,
        monotone_direction=monotone_direction,
    )


def summarize_sampled_tradeoff_grid(records: list[RunRecord]) -> dict[str, Any]:
    family_specs, get_family_spec, _, _ = _family_runtime()
    point_rows: list[dict[str, Any]] = []
    grouped: dict[tuple[str, str, str, str, str], list[RunRecord]] = defaultdict(list)
    for record in records:
        family_key = str(record.metadata.get("family_key", ""))
        grouped[
            (
                family_key,
                record.model_name,
                record.condition,
                str(record.thinking_effort or "disabled"),
                record.cell_id,
            )
        ].append(record)

    for (family_key, model_name, condition, thinking_effort, cell_id), rows in sorted(grouped.items()):
        valid = [_event_indicator(row) for row in rows]
        valid = [value for value in valid if value is not None]
        spec = get_family_spec(family_key)
        point = next(item for item in spec.ladder if item.key == cell_id)
        event_rate = (sum(valid) / len(valid)) if valid else None
        point_rows.append(
            {
                "family_key": family_key,
                "family_title": spec.title,
                "model_name": model_name,
                "condition": condition,
                "thinking_effort": thinking_effort,
                "point_key": cell_id,
                "axis_value": point.axis_value,
                "display_value": point.display_value,
                "runs": len(rows),
                "valid_runs": len(valid),
                "event_rate": event_rate,
                "entropy": _binary_entropy(event_rate) if event_rate is not None else None,
            }
        )

    fit_rows: list[dict[str, Any]] = []
    fit_groups: dict[tuple[str, str, str, str], list[RunRecord]] = defaultdict(list)
    fit_groups_by_order: dict[tuple[str, str, str, str, str], list[RunRecord]] = defaultdict(list)
    for record in records:
        family_key = str(record.metadata.get("family_key", ""))
        effort = str(record.thinking_effort or "disabled")
        fit_groups[(family_key, record.model_name, record.condition, effort)].append(record)
        fit_groups_by_order[(family_key, record.model_name, record.condition, effort, record.presentation_order)].append(record)

    for (family_key, model_name, condition, thinking_effort), rows in sorted(fit_groups.items()):
        spec = get_family_spec(family_key)
        x_native = []
        y = []
        for row in rows:
            indicator = _event_indicator(row)
            if indicator is None:
                continue
            x_native.append(float(row.latent_values[spec.axis_name]))
            y.append(indicator)
        pooled_fit = _fit_monotone_probit(
            x_native=x_native,
            y=y,
            transform_name=spec.transform_name,
            monotone_direction=spec.monotone_direction,
        )
        point_entropy = [
            row["entropy"]
            for row in point_rows
            if row["family_key"] == family_key
            and row["model_name"] == model_name
            and row["condition"] == condition
            and row["thinking_effort"] == thinking_effort
            and row["entropy"] is not None
        ]
        ab_fit = None
        ba_fit = None
        for order in ("AB", "BA"):
            order_rows = fit_groups_by_order.get((family_key, model_name, condition, thinking_effort, order), [])
            x_order = []
            y_order = []
            for row in order_rows:
                indicator = _event_indicator(row)
                if indicator is None:
                    continue
                x_order.append(float(row.latent_values[spec.axis_name]))
                y_order.append(indicator)
            order_fit = _fit_monotone_probit(
                x_native=x_order,
                y=y_order,
                transform_name=spec.transform_name,
                monotone_direction=spec.monotone_direction,
            )
            if order == "AB":
                ab_fit = order_fit
            else:
                ba_fit = order_fit
        order_gap = None
        if (
            ab_fit
            and ba_fit
            and ab_fit.get("midpoint_native") is not None
            and ba_fit.get("midpoint_native") is not None
        ):
            order_gap = abs(float(ab_fit["midpoint_native"]) - float(ba_fit["midpoint_native"]))
        primary_fit_status = pooled_fit.get("fit_status")
        primary_midpoint_position = pooled_fit.get("midpoint_position")
        primary_midpoint_native = pooled_fit.get("midpoint_native")
        primary_midpoint_transformed = pooled_fit.get("midpoint_transformed")
        primary_slope = pooled_fit.get("slope")
        primary_log_loss = pooled_fit.get("log_loss")
        primary_pseudo_r2 = pooled_fit.get("pseudo_r2")
        primary_curve_points = pooled_fit.get("curve_points", [])
        primary_scope = "pooled"
        if not spec.pooled_fit_primary:
            primary_fit_status = "split_by_order"
            primary_midpoint_position = None
            primary_midpoint_native = None
            primary_midpoint_transformed = None
            primary_slope = None
            primary_log_loss = None
            primary_pseudo_r2 = None
            primary_curve_points = []
            primary_scope = "split_by_order"
        fit_rows.append(
            {
                "family_key": family_key,
                "family_title": spec.title,
                "model_name": model_name,
                "condition": condition,
                "thinking_effort": thinking_effort,
                "event_label": spec.event_label,
                "fit_status": primary_fit_status,
                "midpoint_position": primary_midpoint_position,
                "midpoint_native": primary_midpoint_native,
                "midpoint_transformed": primary_midpoint_transformed,
                "slope": primary_slope,
                "log_loss": primary_log_loss,
                "pseudo_r2": primary_pseudo_r2,
                "mean_entropy": mean(point_entropy) if point_entropy else None,
                "primary_scope": primary_scope,
                "pooled_fit_primary": spec.pooled_fit_primary,
                "pooled_fit_status": pooled_fit.get("fit_status"),
                "pooled_midpoint_position": pooled_fit.get("midpoint_position"),
                "pooled_midpoint_native": pooled_fit.get("midpoint_native"),
                "pooled_midpoint_transformed": pooled_fit.get("midpoint_transformed"),
                "pooled_slope": pooled_fit.get("slope"),
                "pooled_log_loss": pooled_fit.get("log_loss"),
                "pooled_pseudo_r2": pooled_fit.get("pseudo_r2"),
                "ab_midpoint": None if ab_fit is None else ab_fit.get("midpoint_native"),
                "ab_fit_status": None if ab_fit is None else ab_fit.get("fit_status"),
                "ab_midpoint_position": None if ab_fit is None else ab_fit.get("midpoint_position"),
                "ab_slope": None if ab_fit is None else ab_fit.get("slope"),
                "ab_pseudo_r2": None if ab_fit is None else ab_fit.get("pseudo_r2"),
                "ba_midpoint": None if ba_fit is None else ba_fit.get("midpoint_native"),
                "ba_fit_status": None if ba_fit is None else ba_fit.get("fit_status"),
                "ba_midpoint_position": None if ba_fit is None else ba_fit.get("midpoint_position"),
                "ba_slope": None if ba_fit is None else ba_fit.get("slope"),
                "ba_pseudo_r2": None if ba_fit is None else ba_fit.get("pseudo_r2"),
                "order_gap": order_gap,
                "curve_points": primary_curve_points,
                "pooled_curve_points": pooled_fit.get("curve_points", []),
            }
        )

    return {
        "family_specs": {
            family_key: {
                "title": spec.title,
                "axis_name": spec.axis_name,
                "axis_units": spec.axis_units,
                "event_choice": spec.event_choice,
                "event_label": spec.event_label,
                "transform_name": spec.transform_name,
                "monotone_direction": spec.monotone_direction,
                "pooled_fit_primary": spec.pooled_fit_primary,
                "ladder": [
                    {"key": point.key, "axis_value": point.axis_value, "display_value": point.display_value}
                    for point in spec.ladder
                ],
            }
            for family_key, spec in family_specs.items()
        },
        "point_summary": point_rows,
        "cross_family_summary": fit_rows,
    }


def _fmt_number(value: Any, *, money: bool = False) -> str:
    if value is None:
        return "n/a"
    number = float(value)
    if money:
        return f"${number:,.0f}"
    if abs(number) >= 100:
        return f"{number:,.1f}"
    if abs(number) >= 10:
        return f"{number:.2f}"
    return f"{number:.3f}"


def _fmt_midpoint(row: dict[str, Any], spec: Any) -> str:
    midpoint = row.get("midpoint_native")
    if midpoint is None:
        return str(row.get("midpoint_position") or "n/a")
    return _fmt_number(midpoint, money=spec.axis_units == "usd_per_role_per_year")


def _condition_color(condition: str) -> str:
    return {
        "baseline": "#264653",
        "placebo": "#6c8ead",
        "reflection": "#d97706",
        "constitution": "#9b2226",
    }.get(condition, "#555555")


def _family_plot_svg(summary: dict[str, Any], family_key: str, thinking_effort: str) -> str:
    _, get_family_spec, condition_names_for_family, _ = _family_runtime()
    spec = get_family_spec(family_key)
    rows = [
        row
        for row in summary["point_summary"]
        if row["family_key"] == family_key and row["thinking_effort"] == thinking_effort
    ]
    if not rows:
        return "<p>No data.</p>"
    width = 430
    height = 220
    margin_left = 50
    margin_right = 20
    margin_top = 20
    margin_bottom = 40
    x_min = min(point.axis_value for point in spec.ladder)
    x_max = max(point.axis_value for point in spec.ladder)
    x_span = max(x_max - x_min, 1.0)

    def px(value: float) -> float:
        return margin_left + ((value - x_min) / x_span) * (width - margin_left - margin_right)

    def py(prob: float) -> float:
        return margin_top + (1.0 - prob) * (height - margin_top - margin_bottom)

    lines = [
        f'<svg viewBox="0 0 {width} {height}" class="family-svg" aria-label="{html.escape(spec.title)} {html.escape(thinking_effort)}">'
    ]
    lines.append(
        f'<line x1="{margin_left}" y1="{height - margin_bottom}" x2="{width - margin_right}" y2="{height - margin_bottom}" stroke="#9fb3c8" stroke-width="1"/>'
    )
    lines.append(
        f'<line x1="{margin_left}" y1="{margin_top}" x2="{margin_left}" y2="{height - margin_bottom}" stroke="#9fb3c8" stroke-width="1"/>'
    )
    for tick in [0.0, 0.5, 1.0]:
        y = py(tick)
        lines.append(f'<line x1="{margin_left}" y1="{y}" x2="{width - margin_right}" y2="{y}" stroke="#e3ecf3" stroke-width="1"/>')
        lines.append(f'<text x="12" y="{y + 4}" font-size="11" fill="#425466">{tick:.1f}</text>')
    for point in spec.ladder:
        x = px(float(point.axis_value))
        lines.append(f'<line x1="{x}" y1="{margin_top}" x2="{x}" y2="{height - margin_bottom}" stroke="#f1f5f9" stroke-width="1"/>')
        lines.append(
            f'<text x="{x}" y="{height - 14}" font-size="11" fill="#425466" text-anchor="middle">{html.escape(point.display_value)}</text>'
        )
    for condition in condition_names_for_family(family_key):
        condition_rows = sorted(
            [row for row in rows if row["condition"] == condition and row["event_rate"] is not None],
            key=lambda item: float(item["axis_value"]),
        )
        if not condition_rows:
            continue
        points = " ".join(
            f"{px(float(row['axis_value']))},{py(float(row['event_rate']))}"
            for row in condition_rows
        )
        color = _condition_color(condition)
        lines.append(f'<polyline points="{points}" fill="none" stroke="{color}" stroke-width="2.5"/>')
        for row in condition_rows:
            lines.append(
                f'<circle cx="{px(float(row["axis_value"]))}" cy="{py(float(row["event_rate"]))}" r="4" fill="{color}"/>'
            )
    lines.append("</svg>")
    return "".join(lines)


def render_sampled_tradeoff_report(summary: dict[str, Any], *, report_title: str) -> str:
    family_specs, get_family_spec, _, _ = _family_runtime()
    fit_rows = summary["cross_family_summary"]
    family_keys = [key for key in family_specs if any(row["family_key"] == key for row in fit_rows)]
    thinking_efforts = ["disabled", "low", "medium", "high"]
    lines = [
        "<!doctype html>",
        '<html lang="en"><head><meta charset="utf-8"><meta name="viewport" content="width=device-width, initial-scale=1">',
        f"<title>{html.escape(report_title)}</title>",
        "<style>",
        "body{font-family:Georgia,serif;background:linear-gradient(180deg,#eef4f8,#dfe9f1);color:#13202b;margin:0;padding:32px;}",
        ".wrap{max-width:1280px;margin:0 auto;}",
        "h1,h2,h3{margin:0 0 12px 0;} p{line-height:1.45;} .card{background:#ffffff;border:1px solid #d6e2ec;border-radius:18px;padding:20px;box-shadow:0 10px 30px rgba(40,64,85,.08);margin:16px 0;}",
        "table{width:100%;border-collapse:collapse;font-size:14px;} th,td{padding:8px 10px;border-bottom:1px solid #e5edf3;text-align:left;vertical-align:top;} th{font-size:12px;letter-spacing:.04em;text-transform:uppercase;color:#526272;}",
        ".grid{display:grid;grid-template-columns:repeat(auto-fit,minmax(280px,1fr));gap:14px;} .plot-card{background:#f8fbfd;border:1px solid #e1ebf2;border-radius:14px;padding:12px;}",
        ".legend{display:flex;gap:12px;flex-wrap:wrap;font-size:12px;color:#526272;margin-top:8px;} .sw{display:inline-block;width:10px;height:10px;border-radius:50%;margin-right:6px;}",
        ".family-svg{width:100%;height:auto;display:block;} .note{font-size:13px;color:#526272;} code{background:#edf3f8;padding:1px 5px;border-radius:5px;}",
        "</style></head><body><div class='wrap'>",
        f"<h1>{html.escape(report_title)}</h1>",
        "<p>This memo reports local revealed-tradeoff fits rather than a global utility surface. Each family defines a 1D axis, sampled choices are fit with a monotone probit curve, and the key outputs are midpoint, slope, fit quality, and AB/BA order gap.</p>",
        "<div class='card'><h2>Cross-Family Fit Summary</h2><table><thead><tr><th>Family</th><th>Condition</th><th>Effort</th><th>Midpoint</th><th>Slope</th><th>Pseudo-R²</th><th>Order Gap</th><th>Entropy</th></tr></thead><tbody>",
    ]
    for row in fit_rows:
        spec = get_family_spec(row["family_key"])
        is_money = spec.axis_units == "usd_per_role_per_year"
        lines.append(
            "<tr>"
            f"<td>{html.escape(spec.title)}</td>"
            f"<td>{html.escape(row['condition'])}</td>"
            f"<td>{html.escape(row['thinking_effort'])}</td>"
            f"<td>{html.escape(_fmt_midpoint(row, spec))}</td>"
            f"<td>{_fmt_number(row['slope'])}</td>"
            f"<td>{_fmt_number(row['pseudo_r2'])}</td>"
            f"<td>{_fmt_number(row['order_gap'], money=is_money)}</td>"
            f"<td>{_fmt_number(row['mean_entropy'])}</td>"
            "</tr>"
        )
    lines.append("</tbody></table></div>")

    lines.append("<div class='card'><h2>Legend</h2><div class='legend'>")
    for condition in ["baseline", "placebo", "reflection", "constitution"]:
        lines.append(
            f"<span><span class='sw' style='background:{_condition_color(condition)}'></span>{html.escape(condition)}</span>"
        )
    lines.append("</div><p class='note'>Plots show sampled event rates rather than deterministic thresholds. Lower or higher midpoints are family-specific directional changes; steeper slopes and smaller AB/BA gaps indicate more coherent local tradeoffs.</p></div>")

    lines.append("<div class='card'><h2>Family Curves</h2>")
    for family_key in family_keys:
        spec = get_family_spec(family_key)
        lines.append(f"<h3>{html.escape(spec.title)}</h3><div class='grid'>")
        for effort in thinking_efforts:
            if not any(row["family_key"] == family_key and row["thinking_effort"] == effort for row in fit_rows):
                continue
            lines.append(
                f"<div class='plot-card'><strong>{html.escape(effort)}</strong>{_family_plot_svg(summary, family_key, effort)}</div>"
            )
        lines.append("</div>")
    lines.append("</div></div></body></html>")
    return "".join(lines)


def write_sampled_tradeoff_report(summary: dict[str, Any], *, base_dir: Path, filename: str, report_title: str) -> Path:
    reports_dir = base_dir / "reports"
    reports_dir.mkdir(parents=True, exist_ok=True)
    report_path = reports_dir / filename
    report_path.write_text(render_sampled_tradeoff_report(summary, report_title=report_title), encoding="utf-8")
    return report_path

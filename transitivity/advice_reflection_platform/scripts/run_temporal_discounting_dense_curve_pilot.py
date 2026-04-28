from __future__ import annotations

import argparse
import concurrent.futures
import html
import json
import math
import threading
from collections import defaultdict
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
    fit_kernel_curve,
    fit_monotone_probit,
    get_family_spec,
    run_custom_sampled_query,
    run_family_prior_probe,
)


FAMILY_KEY = "social_discount_rate"
DEFAULT_POINTS = [1.0, 1.05, 1.1, 1.15, 1.2, 1.25, 1.3, 1.35, 1.4, 1.45, 1.5, 1.6, 1.8]
DEFAULT_CONDITIONS = "baseline,reflection"
DEFAULT_ORDERS = "AB"
_THREAD_LOCAL = threading.local()


def _binary_entropy(prob: float | None) -> float | None:
    if prob is None or prob <= 0.0 or prob >= 1.0:
        return 0.0 if prob is not None else None
    return float(-(prob * math.log2(prob) + (1 - prob) * math.log2(1 - prob)))


def _event_indicator(record: dict[str, Any]) -> int | None:
    canonical = record.get("canonical_choice")
    if canonical not in {"A", "B"}:
        return None
    event_choice = str(record.get("metadata", {}).get("event_choice", ""))
    return 1 if canonical == event_choice else 0


def _display_multiplier(value: float) -> str:
    return f"{value:.2f}x".rstrip("0").rstrip(".")


def _fmt_multiplier(value: float | None) -> str:
    if value is None:
        return "n/a"
    return _display_multiplier(value)


def _fmt_number(value: float | None) -> str:
    if value is None:
        return "n/a"
    if abs(value) >= 100:
        return f"{value:,.1f}"
    if abs(value) >= 10:
        return f"{value:.2f}"
    return f"{value:.3f}"


def _worker_gateway() -> LiveModelGateway:
    gateway = getattr(_THREAD_LOCAL, "gateway", None)
    if gateway is None:
        gateway = LiveModelGateway()
        _THREAD_LOCAL.gateway = gateway
    return gateway


def _group_point_rows(
    records: list[dict[str, Any]],
    *,
    axis_name: str,
    include_pooled: bool,
) -> list[dict[str, Any]]:
    grouped: dict[tuple[str, str, float], list[int]] = defaultdict(list)
    pooled: dict[tuple[str, float], list[int]] = defaultdict(list)
    for record in records:
        indicator = _event_indicator(record)
        if indicator is None:
            continue
        axis_value = float(record["latent_values"][axis_name])
        condition = str(record["condition"])
        order = str(record["presentation_order"])
        grouped[(condition, order, axis_value)].append(indicator)
        if include_pooled:
            pooled[(condition, axis_value)].append(indicator)

    rows: list[dict[str, Any]] = []
    for (condition, order, axis_value), values in sorted(grouped.items()):
        rate = sum(values) / len(values)
        rows.append(
            {
                "condition": condition,
                "order_scope": order,
                "axis_value": axis_value,
                "display_value": _display_multiplier(axis_value),
                "runs": len(values),
                "event_rate": rate,
                "entropy": _binary_entropy(rate),
            }
        )
    if include_pooled:
        for (condition, axis_value), values in sorted(pooled.items()):
            rate = sum(values) / len(values)
            rows.append(
                {
                    "condition": condition,
                    "order_scope": "pooled",
                    "axis_value": axis_value,
                    "display_value": _display_multiplier(axis_value),
                    "runs": len(values),
                    "event_rate": rate,
                    "entropy": _binary_entropy(rate),
                }
            )
    return rows


def _fit_scope(
    records: list[dict[str, Any]],
    *,
    condition: str,
    order_scope: str,
    axis_name: str,
    transform_name: str,
    monotone_direction: str,
) -> dict[str, Any]:
    if order_scope == "pooled":
        relevant = [row for row in records if row["condition"] == condition]
    else:
        relevant = [
            row for row in records if row["condition"] == condition and row["presentation_order"] == order_scope
        ]
    x_native = []
    y = []
    for row in relevant:
        indicator = _event_indicator(row)
        if indicator is None:
            continue
        x_native.append(float(row["latent_values"][axis_name]))
        y.append(indicator)
    probit = fit_monotone_probit(
        x_native=x_native,
        y=y,
        transform_name=transform_name,
        monotone_direction=monotone_direction,
    )
    kernel = fit_kernel_curve(x_native=x_native, y=y, transform_name=transform_name)
    return {
        "condition": condition,
        "order_scope": order_scope,
        "probit": probit,
        "kernel": kernel,
    }


def _plot_svg(
    *,
    axis_values: list[float],
    point_rows: list[dict[str, Any]],
    probit_curve: list[dict[str, Any]],
    kernel_curve: list[dict[str, Any]],
    title: str,
) -> str:
    width = 440
    height = 240
    margin_left = 48
    margin_right = 16
    margin_top = 18
    margin_bottom = 44
    x_min = float(min(axis_values))
    x_max = float(max(axis_values))
    x_span = max(x_max - x_min, 1e-6)

    def px(value: float) -> float:
        return margin_left + ((value - x_min) / x_span) * (width - margin_left - margin_right)

    def py(prob: float) -> float:
        return margin_top + (1.0 - prob) * (height - margin_top - margin_bottom)

    lines = [f'<svg viewBox="0 0 {width} {height}" class="plot" aria-label="{html.escape(title)}">']
    lines.append(
        f'<line x1="{margin_left}" y1="{height - margin_bottom}" x2="{width - margin_right}" y2="{height - margin_bottom}" stroke="#9fb3c8" stroke-width="1"/>'
    )
    lines.append(
        f'<line x1="{margin_left}" y1="{margin_top}" x2="{margin_left}" y2="{height - margin_bottom}" stroke="#9fb3c8" stroke-width="1"/>'
    )
    for tick in (0.0, 0.5, 1.0):
        y = py(tick)
        lines.append(
            f'<line x1="{margin_left}" y1="{y}" x2="{width - margin_right}" y2="{y}" stroke="#e4edf4" stroke-width="1"/>'
        )
        lines.append(f'<text x="12" y="{y + 4}" font-size="11" fill="#425466">{tick:.1f}</text>')
    for axis_value in axis_values:
        x = px(float(axis_value))
        lines.append(
            f'<line x1="{x}" y1="{margin_top}" x2="{x}" y2="{height - margin_bottom}" stroke="#f2f6fa" stroke-width="1"/>'
        )
        lines.append(
            f'<text x="{x}" y="{height - 14}" font-size="11" fill="#425466" text-anchor="middle">{html.escape(_display_multiplier(axis_value))}</text>'
        )

    if probit_curve:
        probit_points = " ".join(
            f"{px(float(point['x_native']))},{py(float(point['p_event']))}" for point in probit_curve
        )
        lines.append(f'<polyline points="{probit_points}" fill="none" stroke="#264653" stroke-width="2.5"/>')
    if kernel_curve:
        kernel_points = " ".join(
            f"{px(float(point['x_native']))},{py(float(point['p_event']))}" for point in kernel_curve
        )
        lines.append(
            f'<polyline points="{kernel_points}" fill="none" stroke="#d97706" stroke-width="2.5" stroke-dasharray="7 5"/>'
        )
    empirical_points = " ".join(
        f"{px(float(row['axis_value']))},{py(float(row['event_rate']))}" for row in point_rows
    )
    lines.append(f'<polyline points="{empirical_points}" fill="none" stroke="#6c8ead" stroke-width="2"/>')
    for row in point_rows:
        lines.append(
            f'<circle cx="{px(float(row["axis_value"]))}" cy="{py(float(row["event_rate"]))}" r="4" fill="#6c8ead"/>'
        )
    lines.append("</svg>")
    return "".join(lines)


def render_report(
    *,
    fit_rows: list[dict[str, Any]],
    point_rows: list[dict[str, Any]],
    axis_values: list[float],
    model_name: str,
    thinking_effort: str,
    repeats_per_order: int,
    conditions: list[str],
    orders: list[str],
) -> str:
    plot_rows = [row for row in fit_rows if row["order_scope"] != "pooled"]
    lines = [
        "<!doctype html>",
        '<html lang="en"><head><meta charset="utf-8"><meta name="viewport" content="width=device-width, initial-scale=1">',
        f"<title>Temporal Discounting Dense Curve Pilot: {html.escape(model_name)} · {html.escape(thinking_effort)}</title>",
        "<style>",
        "body{font-family:Georgia,serif;background:linear-gradient(180deg,#eef4f8,#dfe9f1);color:#13202b;margin:0;padding:32px;}",
        ".wrap{max-width:1240px;margin:0 auto;}",
        "h1,h2,h3{margin:0 0 12px 0;} p{line-height:1.45;} .card{background:#ffffff;border:1px solid #d6e2ec;border-radius:18px;padding:20px;box-shadow:0 10px 30px rgba(40,64,85,.08);margin:16px 0;}",
        ".grid{display:grid;grid-template-columns:repeat(auto-fit,minmax(320px,1fr));gap:14px;}",
        ".plot-card{background:#f8fbfd;border:1px solid #e1ebf2;border-radius:14px;padding:12px;}",
        ".plot{width:100%;height:auto;display:block;} .legend{display:flex;gap:16px;flex-wrap:wrap;font-size:12px;color:#526272;} .sw{display:inline-block;width:12px;height:3px;margin-right:6px;vertical-align:middle;}",
        ".sw.probit{background:#264653;} .sw.kernel{background:#d97706;} .sw.empirical{background:#6c8ead;}",
        "table{width:100%;border-collapse:collapse;font-size:14px;} th,td{padding:8px 10px;border-bottom:1px solid #e5edf3;text-align:left;vertical-align:top;} th{font-size:12px;letter-spacing:.04em;text-transform:uppercase;color:#526272;}",
        "code{background:#edf3f8;padding:1px 5px;border-radius:5px;} .note{font-size:13px;color:#526272;}",
        "</style></head><body><div class='wrap'>",
        f"<h1>Temporal Discounting Dense Curve Pilot: {html.escape(model_name)}</h1>",
        f"<p>Thinking effort: <code>{html.escape(thinking_effort)}</code>. Conditions: {', '.join(f'<code>{html.escape(condition)}</code>' for condition in conditions)}. Orders: {', '.join(f'<code>{html.escape(order)}</code>' for order in orders)}. Repeats per order and rung: {repeats_per_order}.</p>",
        "<div class='card'><h2>Estimator Summary</h2><table><thead><tr><th>Condition</th><th>Scope</th><th>Probit Midpoint</th><th>Probit Slope</th><th>Probit Pseudo-R²</th><th>Kernel Midpoint</th><th>Mean Entropy</th></tr></thead><tbody>",
    ]
    for row in fit_rows:
        probit = row["probit"]
        kernel = row["kernel"]
        lines.append(
            "<tr>"
            f"<td>{html.escape(row['condition'])}</td>"
            f"<td>{html.escape(row['order_scope'])}</td>"
            f"<td>{html.escape(_fmt_multiplier(probit.get('midpoint_native')) if probit.get('midpoint_native') is not None else str(probit.get('midpoint_position')))}</td>"
            f"<td>{html.escape(_fmt_number(probit.get('slope')))}</td>"
            f"<td>{html.escape(_fmt_number(probit.get('pseudo_r2')))}</td>"
            f"<td>{html.escape(_fmt_multiplier(kernel.get('midpoint_native')) if kernel.get('midpoint_native') is not None else str(kernel.get('midpoint_position')))}</td>"
            f"<td>{html.escape(_fmt_number(row.get('mean_entropy')))}</td>"
            "</tr>"
        )
    lines.append("</tbody></table></div>")
    lines.append("<div class='card'><h2>Curves</h2><div class='legend'>")
    lines.append("<span><span class='sw empirical'></span>Empirical sampled rate</span>")
    lines.append("<span><span class='sw probit'></span>Probit fit</span>")
    lines.append("<span><span class='sw kernel'></span>Kernel smoother</span>")
    lines.append("</div><div class='grid'>")
    for row in plot_rows:
        curve_points = [
            point for point in point_rows if point["condition"] == row["condition"] and point["order_scope"] == row["order_scope"]
        ]
        title = f"{row['condition']} · {row['order_scope']}"
        lines.append("<div class='plot-card'>")
        lines.append(f"<h3>{html.escape(title)}</h3>")
        lines.append(
            _plot_svg(
                axis_values=axis_values,
                point_rows=curve_points,
                probit_curve=row["probit"].get("curve_points", []),
                kernel_curve=row["kernel"].get("curve_points", []),
                title=title,
            )
        )
        lines.append("</div>")
    lines.append("</div></div>")
    lines.append("<div class='card'><h2>Point Summary</h2><table><thead><tr><th>Condition</th><th>Scope</th><th>Multiplier</th><th>Event Rate</th><th>Entropy</th><th>Runs</th></tr></thead><tbody>")
    for row in point_rows:
        lines.append(
            "<tr>"
            f"<td>{html.escape(row['condition'])}</td>"
            f"<td>{html.escape(row['order_scope'])}</td>"
            f"<td>{html.escape(row['display_value'])}</td>"
            f"<td>{html.escape(_fmt_number(row['event_rate']))}</td>"
            f"<td>{html.escape(_fmt_number(row['entropy']))}</td>"
            f"<td>{row['runs']}</td>"
            "</tr>"
        )
    lines.append("</tbody></table></div></div></body></html>")
    return "".join(lines)


def main() -> None:
    parser = argparse.ArgumentParser(description="Run a dense temporal-discounting curve pilot with probit and kernel fits.")
    parser.add_argument("--model", type=str, default="claude-sonnet-4-6")
    parser.add_argument("--thinking-effort", type=str, default="disabled")
    parser.add_argument("--repeats-per-order", type=int, default=10)
    parser.add_argument("--points", type=str, default=",".join(str(value) for value in DEFAULT_POINTS))
    parser.add_argument("--conditions", type=str, default=DEFAULT_CONDITIONS)
    parser.add_argument("--orders", type=str, default=DEFAULT_ORDERS)
    parser.add_argument("--max-workers", type=int, default=1)
    parser.add_argument("--request-timeout-seconds", type=float, default=90.0)
    parser.add_argument("--output-prefix", type=str, default="")
    args = parser.parse_args()

    point_values = [float(item.strip()) for item in args.points.split(",") if item.strip()]
    conditions = [item.strip() for item in args.conditions.split(",") if item.strip()]
    orders = [item.strip() for item in args.orders.split(",") if item.strip()]
    if not conditions:
        raise ValueError("At least one condition must be specified.")
    if not orders:
        raise ValueError("At least one order must be specified.")
    if args.max_workers < 1:
        raise ValueError("--max-workers must be at least 1")

    spec = get_family_spec(FAMILY_KEY)
    point_keys = [f"d{idx+1}" for idx, _ in enumerate(point_values)]
    display_values = [_display_multiplier(value) for value in point_values]
    base_dir = Path(__file__).resolve().parents[1]
    stamp = datetime.now().strftime("%Y%m%d_%H%M%S")
    filename_prefix = args.output_prefix or f"temporal_discounting_{stamp}"
    store = ArtifactStore(base_dir)

    gateway = LiveModelGateway()
    prior_artifacts: dict[str, Any] = {}
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

    jobs: list[dict[str, Any]] = []
    for condition in conditions:
        for point_key, axis_value, display_value in zip(point_keys, point_values, display_values, strict=True):
            for order in orders:
                for repeat_idx in range(1, args.repeats_per_order + 1):
                    jobs.append(
                        {
                            "condition": condition,
                            "axis_value": axis_value,
                            "point_key": point_key,
                            "display_value": display_value,
                            "order": order,
                            "repeat_idx": repeat_idx,
                        }
                    )

    def _run_job(job: dict[str, Any]) -> Any:
        return run_custom_sampled_query(
            family_key=FAMILY_KEY,
            axis_value=job["axis_value"],
            point_key=job["point_key"],
            display_value=job["display_value"],
            model_name=args.model,
            condition_name=job["condition"],
            thinking_effort=args.thinking_effort,
            presentation_order=job["order"],
            repeat_idx=job["repeat_idx"],
            gateway=_worker_gateway(),
            prior_artifact=prior_artifacts.get(job["condition"]),
            request_timeout_seconds=args.request_timeout_seconds,
        )

    records = []
    total = len(jobs)
    completed = 0
    if args.max_workers == 1:
        for job in jobs:
            records.append(_run_job(job))
            completed += 1
            if completed % 10 == 0 or completed == total:
                print(f"progress {completed}/{total}", flush=True)
    else:
        with concurrent.futures.ThreadPoolExecutor(max_workers=args.max_workers) as executor:
            futures = [executor.submit(_run_job, job) for job in jobs]
            for future in concurrent.futures.as_completed(futures):
                records.append(future.result())
                completed += 1
                if completed % 10 == 0 or completed == total:
                    print(f"progress {completed}/{total}", flush=True)

    records.sort(
        key=lambda record: (
            str(record.condition),
            float(record.latent_values[spec.axis_name]),
            str(record.presentation_order),
            int(record.repeat_idx),
        )
    )

    raw_path, flat_csv_path = store.write_records(records, filename_prefix)
    serialized_records = [record.to_flat_dict() for record in records]
    include_pooled = len(orders) > 1
    point_rows = _group_point_rows(
        serialized_records,
        axis_name=spec.axis_name,
        include_pooled=include_pooled,
    )
    fit_rows: list[dict[str, Any]] = []
    scopes = list(orders) + (["pooled"] if include_pooled else [])
    for condition in conditions:
        for order_scope in scopes:
            fit = _fit_scope(
                serialized_records,
                condition=condition,
                order_scope=order_scope,
                axis_name=spec.axis_name,
                transform_name=spec.transform_name,
                monotone_direction=spec.monotone_direction,
            )
            entropy_rows = [
                row["entropy"]
                for row in point_rows
                if row["condition"] == condition and row["order_scope"] == order_scope and row["entropy"] is not None
            ]
            fit_rows.append(
                {
                    **fit,
                    "mean_entropy": (sum(entropy_rows) / len(entropy_rows)) if entropy_rows else None,
                }
            )

    analysis = {
        "family_key": FAMILY_KEY,
        "model": args.model,
        "thinking_effort": args.thinking_effort,
        "points": point_values,
        "repeats_per_order": args.repeats_per_order,
        "orders": orders,
        "conditions": conditions,
        "prior_artifacts": prior_artifacts,
        "fit_rows": fit_rows,
        "point_rows": point_rows,
        "monotone_direction": spec.monotone_direction,
        "pooled_fit_included": include_pooled,
    }

    fit_csv_rows = []
    for row in fit_rows:
        fit_csv_rows.append(
            {
                "condition": row["condition"],
                "order_scope": row["order_scope"],
                "probit_fit_status": row["probit"].get("fit_status"),
                "probit_midpoint_native": row["probit"].get("midpoint_native"),
                "probit_midpoint_position": row["probit"].get("midpoint_position"),
                "probit_slope": row["probit"].get("slope"),
                "probit_pseudo_r2": row["probit"].get("pseudo_r2"),
                "kernel_fit_status": row["kernel"].get("fit_status"),
                "kernel_midpoint_native": row["kernel"].get("midpoint_native"),
                "kernel_midpoint_position": row["kernel"].get("midpoint_position"),
                "kernel_transition_width_native": row["kernel"].get("transition_width_native"),
                "kernel_bandwidth_transformed": row["kernel"].get("bandwidth_transformed"),
                "mean_entropy": row.get("mean_entropy"),
            }
        )

    fit_csv_path = store.write_summary(fit_csv_rows, f"{filename_prefix}_fit_summary.csv")
    point_csv_path = store.write_summary(point_rows, f"{filename_prefix}_point_summary.csv")
    analysis_path = base_dir / "runs" / "summaries" / f"{filename_prefix}_analysis.json"
    analysis_path.write_text(json.dumps(analysis, indent=2, sort_keys=True) + "\n", encoding="utf-8")
    report_path = base_dir / "reports" / f"{filename_prefix}.html"
    report_path.write_text(
        render_report(
            fit_rows=fit_rows,
            point_rows=point_rows,
            axis_values=point_values,
            model_name=args.model,
            thinking_effort=args.thinking_effort,
            repeats_per_order=args.repeats_per_order,
            conditions=conditions,
            orders=orders,
        ),
        encoding="utf-8",
    )

    print(f"raw_path={raw_path}")
    print(f"flat_csv_path={flat_csv_path}")
    print(f"fit_csv_path={fit_csv_path}")
    print(f"point_csv_path={point_csv_path}")
    print(f"analysis_path={analysis_path}")
    print(f"report_path={report_path}")
    print(
        json.dumps(
            {
                "model": args.model,
                "thinking_effort": args.thinking_effort,
                "conditions": conditions,
                "orders": orders,
                "max_workers": args.max_workers,
                "points": point_values,
                "repeats_per_order": args.repeats_per_order,
            },
            indent=2,
            sort_keys=True,
        )
    )


if __name__ == "__main__":
    main()

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
    render_family_reflection_prompt,
    run_custom_sampled_query,
    run_family_prior_probe,
)


FAMILY_KEY = "ai_labor_displacement"
DEFAULT_POINTS = [25_000, 40_000, 55_000, 70_000, 85_000, 100_000, 115_000, 130_000, 145_000]
DEFAULT_CONDITIONS = "baseline,placebo,reflection,constitution"
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


def _display_money(value: int) -> str:
    if value % 1000 == 0:
        if value % 10_000 == 0:
            return f"${value // 1000}k"
        return f"${value / 1000:.1f}k"
    return f"${value:,}"


def _worker_gateway() -> LiveModelGateway:
    gateway = getattr(_THREAD_LOCAL, "gateway", None)
    if gateway is None:
        gateway = LiveModelGateway()
        _THREAD_LOCAL.gateway = gateway
    return gateway


def _group_point_rows(records: list[dict[str, Any]]) -> list[dict[str, Any]]:
    grouped: dict[tuple[str, str, int], list[int]] = defaultdict(list)
    pooled: dict[tuple[str, int], list[int]] = defaultdict(list)
    for record in records:
        indicator = _event_indicator(record)
        if indicator is None:
            continue
        axis_value = int(float(record["latent_values"]["annual_net_savings_per_role"]))
        condition = str(record["condition"])
        order = str(record["presentation_order"])
        grouped[(condition, order, axis_value)].append(indicator)
        pooled[(condition, axis_value)].append(indicator)

    rows: list[dict[str, Any]] = []
    for (condition, order, axis_value), values in sorted(grouped.items()):
        rate = sum(values) / len(values)
        rows.append(
            {
                "condition": condition,
                "order_scope": order,
                "axis_value": axis_value,
                "display_value": _display_money(axis_value),
                "runs": len(values),
                "event_rate": rate,
                "entropy": _binary_entropy(rate),
            }
        )
    for (condition, axis_value), values in sorted(pooled.items()):
        rate = sum(values) / len(values)
        rows.append(
            {
                "condition": condition,
                "order_scope": "pooled",
                "axis_value": axis_value,
                "display_value": _display_money(axis_value),
                "runs": len(values),
                "event_rate": rate,
                "entropy": _binary_entropy(rate),
            }
        )
    return rows


def _fit_scope(records: list[dict[str, Any]], *, condition: str, order_scope: str) -> dict[str, Any]:
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
        x_native.append(float(row["latent_values"]["annual_net_savings_per_role"]))
        y.append(indicator)
    transform_name = get_family_spec(FAMILY_KEY).transform_name
    probit = fit_monotone_probit(x_native=x_native, y=y, transform_name=transform_name)
    kernel = fit_kernel_curve(x_native=x_native, y=y, transform_name=transform_name)
    return {
        "condition": condition,
        "order_scope": order_scope,
        "probit": probit,
        "kernel": kernel,
    }


def _fmt_money(value: float | None) -> str:
    if value is None:
        return "n/a"
    return f"${value:,.0f}"


def _fmt_number(value: float | None) -> str:
    if value is None:
        return "n/a"
    if abs(value) >= 100:
        return f"{value:,.1f}"
    if abs(value) >= 10:
        return f"{value:.2f}"
    return f"{value:.3f}"


def _plot_svg(
    *,
    axis_values: list[int],
    point_rows: list[dict[str, Any]],
    probit_curve: list[dict[str, Any]],
    kernel_curve: list[dict[str, Any]],
    title: str,
) -> str:
    width = 420
    height = 240
    margin_left = 48
    margin_right = 16
    margin_top = 18
    margin_bottom = 44
    x_min = float(min(axis_values))
    x_max = float(max(axis_values))
    x_span = max(x_max - x_min, 1.0)

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
        lines.append(f'<line x1="{x}" y1="{margin_top}" x2="{x}" y2="{height - margin_bottom}" stroke="#f2f6fa" stroke-width="1"/>')
        lines.append(
            f'<text x="{x}" y="{height - 14}" font-size="11" fill="#425466" text-anchor="middle">{html.escape(_display_money(axis_value))}</text>'
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
    axis_values: list[int],
    model_name: str,
    repeats_per_order: int,
    conditions: list[str],
    orders: list[str],
) -> str:
    order_plot_rows = [
        row
        for row in fit_rows
        if row["order_scope"] in orders
    ]
    lines = [
        "<!doctype html>",
        '<html lang="en"><head><meta charset="utf-8"><meta name="viewport" content="width=device-width, initial-scale=1">',
        f"<title>AI Labor Dense Curve Pilot: {html.escape(model_name)}</title>",
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
        f"<h1>AI Labor Dense Curve Pilot: {html.escape(model_name)}</h1>",
        f"<p>Dense sampled follow-up on the AI labor-displacement family. Conditions: {', '.join(f'<code>{html.escape(condition)}</code>' for condition in conditions)}. Orders used: {', '.join(f'<code>{html.escape(order)}</code>' for order in orders)}. There are {repeats_per_order} repeats per order and {repeats_per_order * len(orders)} samples per savings point.</p>",
        "<div class='card'><h2>Estimator Summary</h2><table><thead><tr><th>Condition</th><th>Scope</th><th>Probit Midpoint</th><th>Probit Slope</th><th>Probit Pseudo-R²</th><th>Kernel Midpoint</th><th>Kernel 25-75 Width</th><th>Mean Entropy</th></tr></thead><tbody>",
    ]
    for row in fit_rows:
        probit = row["probit"]
        kernel = row["kernel"]
        lines.append(
            "<tr>"
            f"<td>{html.escape(row['condition'])}</td>"
            f"<td>{html.escape(row['order_scope'])}</td>"
            f"<td>{html.escape(_fmt_money(probit.get('midpoint_native')) if probit.get('midpoint_native') is not None else str(probit.get('midpoint_position')))}</td>"
            f"<td>{html.escape(_fmt_number(probit.get('slope')))}</td>"
            f"<td>{html.escape(_fmt_number(probit.get('pseudo_r2')))}</td>"
            f"<td>{html.escape(_fmt_money(kernel.get('midpoint_native')) if kernel.get('midpoint_native') is not None else str(kernel.get('midpoint_position')))}</td>"
            f"<td>{html.escape(_fmt_money(kernel.get('transition_width_native')))}</td>"
            f"<td>{html.escape(_fmt_number(row.get('mean_entropy')))}</td>"
            "</tr>"
        )
    lines.append("</tbody></table></div>")
    lines.append("<div class='card'><h2>Order-Specific Curves</h2><div class='legend'>")
    lines.append("<span><span class='sw empirical'></span>Empirical sampled rate</span>")
    lines.append("<span><span class='sw probit'></span>Probit fit</span>")
    lines.append("<span><span class='sw kernel'></span>Kernel smoother</span>")
    lines.append("</div><div class='grid'>")
    for row in order_plot_rows:
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
        lines.append(
            f"<p class='note'>Probit midpoint: <strong>{html.escape(_fmt_money(row['probit'].get('midpoint_native')) if row['probit'].get('midpoint_native') is not None else str(row['probit'].get('midpoint_position')))}</strong>. "
            f"Kernel midpoint: <strong>{html.escape(_fmt_money(row['kernel'].get('midpoint_native')) if row['kernel'].get('midpoint_native') is not None else str(row['kernel'].get('midpoint_position')))}</strong>.</p>"
        )
        lines.append("</div>")
    lines.append("</div></div>")
    lines.append("<div class='card'><h2>Point Summary</h2><table><thead><tr><th>Condition</th><th>Scope</th><th>Savings</th><th>Event Rate</th><th>Entropy</th><th>Runs</th></tr></thead><tbody>")
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
    parser = argparse.ArgumentParser(description="Run a dense AI labor curve pilot with probit and kernel fits.")
    parser.add_argument("--model", type=str, default="openai/gpt-4o")
    parser.add_argument("--thinking-effort", type=str, default="disabled")
    parser.add_argument("--repeats-per-order", type=int, default=5)
    parser.add_argument("--points", type=str, default=",".join(str(value) for value in DEFAULT_POINTS))
    parser.add_argument("--conditions", type=str, default=DEFAULT_CONDITIONS)
    parser.add_argument("--orders", type=str, default="AB,BA")
    parser.add_argument("--max-workers", type=int, default=1)
    parser.add_argument("--request-timeout-seconds", type=float, default=90.0)
    parser.add_argument("--output-prefix", type=str, default="")
    args = parser.parse_args()

    point_values = [int(item.strip()) for item in args.points.split(",") if item.strip()]
    conditions = [item.strip() for item in args.conditions.split(",") if item.strip()]
    orders = [item.strip().upper() for item in args.orders.split(",") if item.strip()]
    if not conditions:
        raise ValueError("At least one condition must be specified.")
    if not orders:
        raise ValueError("At least one order must be specified.")
    if any(order not in {"AB", "BA"} for order in orders):
        raise ValueError("--orders must contain only AB and/or BA")
    if args.max_workers < 1:
        raise ValueError("--max-workers must be at least 1")
    point_keys = [f"d{idx+1}" for idx, _ in enumerate(point_values)]
    display_values = [_display_money(value) for value in point_values]
    base_dir = Path(__file__).resolve().parents[1]
    stamp = datetime.now().strftime("%Y%m%d_%H%M%S")
    filename_prefix = args.output_prefix or f"ai_labor_dense_curve_pilot_{stamp}"

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

    jobs: list[dict[str, Any]] = []
    for condition in conditions:
        for point_key, axis_value, display_value in zip(point_keys, point_values, display_values, strict=True):
            for order in orders:
                for repeat_idx in range(1, args.repeats_per_order + 1):
                    jobs.append(
                        {
                            "condition": condition,
                            "point_key": point_key,
                            "axis_value": axis_value,
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
            float(record.latent_values["annual_net_savings_per_role"]),
            str(record.presentation_order),
            int(record.repeat_idx),
        )
    )

    raw_path, flat_csv_path = store.write_records(records, filename_prefix)
    serialized_records = [record.to_flat_dict() for record in records]
    point_rows = _group_point_rows(serialized_records)
    fit_rows = []
    for condition in conditions:
        pooled_fit = _fit_scope(serialized_records, condition=condition, order_scope="pooled")
        mean_entropy = [
            row["entropy"] for row in point_rows if row["condition"] == condition and row["order_scope"] == "pooled"
        ]
        order_fits = {order: _fit_scope(serialized_records, condition=condition, order_scope=order) for order in orders}
        ab_fit = order_fits.get("AB")
        ba_fit = order_fits.get("BA")
        probit_order_gap = None
        if (
            ab_fit is not None
            and ba_fit is not None
            and ab_fit["probit"].get("midpoint_native") is not None
            and ba_fit["probit"].get("midpoint_native") is not None
        ):
            probit_order_gap = abs(float(ab_fit["probit"]["midpoint_native"]) - float(ba_fit["probit"]["midpoint_native"]))
        kernel_order_gap = None
        if (
            ab_fit is not None
            and ba_fit is not None
            and ab_fit["kernel"].get("midpoint_native") is not None
            and ba_fit["kernel"].get("midpoint_native") is not None
        ):
            kernel_order_gap = abs(float(ab_fit["kernel"]["midpoint_native"]) - float(ba_fit["kernel"]["midpoint_native"]))
        pooled_fit["mean_entropy"] = sum(mean_entropy) / len(mean_entropy) if mean_entropy else None
        pooled_fit["probit_order_gap"] = probit_order_gap
        pooled_fit["kernel_order_gap"] = kernel_order_gap
        fit_rows.append({**pooled_fit, "mean_entropy": pooled_fit["mean_entropy"]})
        for order in orders:
            order_point_rows = [row for row in point_rows if row["condition"] == condition and row["order_scope"] == order]
            fit_rows.append(
                {
                    **order_fits[order],
                    "mean_entropy": (
                        sum(row["entropy"] for row in order_point_rows) / len(order_point_rows)
                        if order_point_rows
                        else None
                    ),
                }
            )

    analysis = {
        "family_key": FAMILY_KEY,
        "model": args.model,
        "thinking_effort": args.thinking_effort,
        "points": point_values,
        "repeats_per_order": args.repeats_per_order,
        "orders": orders,
        "samples_per_point_total": args.repeats_per_order * len(orders),
        "conditions": conditions,
        "prior_artifacts": prior_artifacts,
        "fit_rows": fit_rows,
        "point_rows": point_rows,
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
                "probit_order_gap": row.get("probit_order_gap"),
                "kernel_order_gap": row.get("kernel_order_gap"),
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
                "max_workers": args.max_workers,
                "orders": orders,
                "points": point_values,
                "repeats_per_order": args.repeats_per_order,
            },
            indent=2,
            sort_keys=True,
        )
    )


if __name__ == "__main__":
    main()

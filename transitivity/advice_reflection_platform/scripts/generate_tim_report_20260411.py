from __future__ import annotations

import html
import json
from pathlib import Path
import sys
from typing import Any


PACKAGE_ROOT = Path(__file__).resolve().parents[2]
if str(PACKAGE_ROOT) not in sys.path:
    sys.path.insert(0, str(PACKAGE_ROOT))


BASE_DIR = PACKAGE_ROOT / "advice_reflection_platform"


AI_LABOR_DISABLED_PATHS = {
    "baseline": BASE_DIR / "runs" / "summaries" / "ai_labor_gpt54_densecurve_baselineonly_r1_analysis.json",
    "placebo": BASE_DIR / "runs" / "summaries" / "ai_labor_gpt54_densecurve_placebo_r1_analysis.json",
    "reflection": BASE_DIR / "runs" / "summaries" / "ai_labor_gpt54_densecurve_reflectiononly_r1_analysis.json",
    "productivity_reflection": BASE_DIR / "runs" / "summaries" / "ai_labor_gpt54_densecurve_productivity_reflection_r1_analysis.json",
}
AI_LABOR_MEDIUM_PATH = BASE_DIR / "runs" / "summaries" / "ai_labor_gpt54_densecurve_medium_all4_r1_analysis.json"

SAT_PATHS = {
    ("gemini-3.1-pro-preview", "disabled"): BASE_DIR
    / "runs"
    / "summaries"
    / "admissions_gemini31propreview_densecurve_all4_disabled_candidate_r1_analysis.json",
    ("gemini-3.1-pro-preview", "medium"): BASE_DIR
    / "runs"
    / "summaries"
    / "admissions_gemini31propreview_densecurve_all4_medium_candidate_r1_analysis.json",
    ("gemini-3.1-flash-lite-preview", "disabled"): BASE_DIR
    / "runs"
    / "summaries"
    / "admissions_gemini31flashlitepreview_densecurve_all4_disabled_candidate_r1_analysis.json",
    ("gemini-3.1-flash-lite-preview", "medium"): BASE_DIR
    / "runs"
    / "summaries"
    / "admissions_gemini31flashlitepreview_densecurve_all4_medium_candidate_r1_analysis.json",
}

AI_CONDITION_LABELS = {
    "baseline": "Baseline",
    "placebo": "Placebo",
    "reflection": "Reflection",
    "productivity_reflection": "Productivity Reflection",
}
SAT_CONDITION_LABELS = {
    "baseline": "Baseline",
    "placebo": "Placebo",
    "reflection": "Reflection",
    "preparedness_reflection": "Preparedness Reflection",
}
COLORS = {
    "baseline": "#264653",
    "placebo": "#6c8ead",
    "reflection": "#d97706",
    "productivity_reflection": "#c2410c",
    "preparedness_reflection": "#8b5cf6",
}


def _load_json(path: Path) -> dict[str, Any]:
    return json.loads(path.read_text(encoding="utf-8"))


def _fmt_money(value: float | None) -> str:
    if value is None:
        return "n/a"
    return f"${value:,.0f}"


def _fmt_sat(value: float | None) -> str:
    if value is None:
        return "n/a"
    return f"{value:,.0f}"


def _fmt_number(value: float | None) -> str:
    if value is None:
        return "n/a"
    if abs(value) >= 100:
        return f"{value:,.1f}"
    if abs(value) >= 10:
        return f"{value:.2f}"
    return f"{value:.3f}"


def _pooled_fit_map(analysis: dict[str, Any]) -> dict[str, dict[str, Any]]:
    return {row["condition"]: row for row in analysis["fit_rows"] if row["order_scope"] == "pooled"}


def _pooled_point_map(analysis: dict[str, Any]) -> dict[str, list[dict[str, Any]]]:
    grouped: dict[str, list[dict[str, Any]]] = {}
    for row in analysis["point_rows"]:
        if row["order_scope"] != "pooled":
            continue
        grouped.setdefault(row["condition"], []).append(row)
    for rows in grouped.values():
        rows.sort(key=lambda row: float(row["axis_value"]))
    return grouped


def _build_ai_labor_disabled() -> dict[str, Any]:
    analyses = {condition: _load_json(path) for condition, path in AI_LABOR_DISABLED_PATHS.items()}
    sample = next(iter(analyses.values()))
    fit_rows = []
    point_rows: list[dict[str, Any]] = []
    for condition, analysis in analyses.items():
        fit_rows.extend(analysis["fit_rows"])
        point_rows.extend(analysis["point_rows"])
    return {
        "family": "ai_labor_displacement",
        "label": "AI Labor",
        "model": "openai/gpt-5.4",
        "thinking_effort": "disabled",
        "conditions": list(AI_CONDITION_LABELS),
        "points": sample["points"],
        "fit_rows": fit_rows,
        "point_rows": point_rows,
    }


def _plot_svg(
    *,
    axis_values: list[float],
    grouped_points: dict[str, list[dict[str, Any]]],
    grouped_fits: dict[str, dict[str, Any]],
    labels: dict[str, str],
    axis_formatter,
    width: int = 520,
    height: int = 280,
) -> str:
    margin_left = 56
    margin_right = 18
    margin_top = 18
    margin_bottom = 52
    x_min = float(min(axis_values))
    x_max = float(max(axis_values))
    x_span = max(x_max - x_min, 1.0)

    def px(value: float) -> float:
        return margin_left + ((value - x_min) / x_span) * (width - margin_left - margin_right)

    def py(prob: float) -> float:
        return margin_top + (1.0 - prob) * (height - margin_top - margin_bottom)

    lines = [f'<svg viewBox="0 0 {width} {height}" class="plot">']
    lines.append(
        f'<line x1="{margin_left}" y1="{height - margin_bottom}" x2="{width - margin_right}" y2="{height - margin_bottom}" stroke="#9fb3c8" stroke-width="1"/>'
    )
    lines.append(
        f'<line x1="{margin_left}" y1="{margin_top}" x2="{margin_left}" y2="{height - margin_bottom}" stroke="#9fb3c8" stroke-width="1"/>'
    )
    for tick in (0.0, 0.5, 1.0):
        y = py(tick)
        lines.append(
            f'<line x1="{margin_left}" y1="{y}" x2="{width - margin_right}" y2="{y}" stroke="#edf3f8" stroke-width="1"/>'
        )
        lines.append(f'<text x="16" y="{y + 4}" font-size="11" fill="#526272">{tick:.1f}</text>')
    for axis_value in axis_values:
        x = px(float(axis_value))
        lines.append(
            f'<line x1="{x}" y1="{margin_top}" x2="{x}" y2="{height - margin_bottom}" stroke="#f4f7fa" stroke-width="1"/>'
        )
        lines.append(
            f'<text x="{x}" y="{height - 16}" font-size="11" fill="#526272" text-anchor="middle">{html.escape(axis_formatter(float(axis_value)))}</text>'
        )

    for condition, label in labels.items():
        fit_row = grouped_fits.get(condition)
        point_rows = grouped_points.get(condition, [])
        color = COLORS[condition]
        if fit_row:
            curve_points = fit_row["probit"].get("curve_points", [])
            if curve_points:
                probit_points = " ".join(
                    f"{px(float(point['x_native']))},{py(float(point['p_event']))}" for point in curve_points
                )
                lines.append(
                    f'<polyline points="{probit_points}" fill="none" stroke="{color}" stroke-width="2.5"/>'
                )
        if point_rows:
            empirical_points = " ".join(
                f"{px(float(row['axis_value']))},{py(float(row['event_rate']))}" for row in point_rows
            )
            lines.append(
                f'<polyline points="{empirical_points}" fill="none" stroke="{color}" stroke-width="1.25" stroke-dasharray="4 4" opacity="0.65"/>'
            )
            for row in point_rows:
                lines.append(
                    f'<circle cx="{px(float(row["axis_value"]))}" cy="{py(float(row["event_rate"]))}" r="3.5" fill="{color}" opacity="0.95"/>'
                )
    lines.append("</svg>")
    return "".join(lines)


def _build_summary_row(
    *,
    fit_row: dict[str, Any],
    midpoint_formatter,
    thinking_label: str,
    condition_label: str,
    gap_formatter,
) -> str:
    probit = fit_row["probit"]
    kernel = fit_row["kernel"]
    probit_midpoint = midpoint_formatter(probit.get("midpoint_native")) if probit.get("midpoint_native") is not None else str(probit.get("midpoint_position"))
    kernel_midpoint = midpoint_formatter(kernel.get("midpoint_native")) if kernel.get("midpoint_native") is not None else str(kernel.get("midpoint_position"))
    return (
        "<tr>"
        f"<td>{html.escape(thinking_label)}</td>"
        f"<td>{html.escape(condition_label)}</td>"
        f"<td>{html.escape(probit_midpoint)}</td>"
        f"<td>{html.escape(_fmt_number(probit.get('slope')))}</td>"
        f"<td>{html.escape(_fmt_number(probit.get('pseudo_r2')))}</td>"
        f"<td>{html.escape(kernel_midpoint)}</td>"
        f"<td>{html.escape(_fmt_number(fit_row.get('mean_entropy')))}</td>"
        f"<td>{html.escape(gap_formatter(fit_row.get('probit_order_gap')))}</td>"
        "</tr>"
    )


def _load_sat_bundle() -> list[dict[str, Any]]:
    rows = []
    for (model_short, effort), path in SAT_PATHS.items():
        analysis = _load_json(path)
        rows.append(
            {
                "model_short": model_short,
                "thinking_effort": effort,
                "analysis": analysis,
            }
        )
    return rows


def render_report() -> str:
    ai_disabled = _build_ai_labor_disabled()
    ai_medium = _load_json(AI_LABOR_MEDIUM_PATH)
    sat_bundles = _load_sat_bundle()

    ai_disabled_fits = _pooled_fit_map(ai_disabled)
    ai_disabled_points = _pooled_point_map(ai_disabled)
    ai_medium_fits = _pooled_fit_map(ai_medium)
    ai_medium_points = _pooled_point_map(ai_medium)

    sat_fit_maps = {
        (bundle["model_short"], bundle["thinking_effort"]): _pooled_fit_map(bundle["analysis"]) for bundle in sat_bundles
    }
    sat_point_maps = {
        (bundle["model_short"], bundle["thinking_effort"]): _pooled_point_map(bundle["analysis"]) for bundle in sat_bundles
    }

    pro_disabled = sat_fit_maps[("gemini-3.1-pro-preview", "disabled")]
    pro_medium = sat_fit_maps[("gemini-3.1-pro-preview", "medium")]
    flash_disabled = sat_fit_maps[("gemini-3.1-flash-lite-preview", "disabled")]
    flash_medium = sat_fit_maps[("gemini-3.1-flash-lite-preview", "medium")]

    lines = [
        "<!doctype html>",
        '<html lang="en"><head><meta charset="utf-8"><meta name="viewport" content="width=device-width, initial-scale=1">',
        "<title>Tim Report · April 11, 2026</title>",
        "<style>",
        "body{font-family:Georgia,serif;background:#f3f7fb;color:#13202b;margin:0;padding:28px;}",
        ".wrap{max-width:1320px;margin:0 auto;}",
        "h1,h2,h3{margin:0 0 10px 0;} p{line-height:1.55;margin:10px 0;} .muted{color:#526272;font-size:14px;}",
        ".card{background:#fff;border:1px solid #d8e3ec;border-radius:18px;padding:20px;box-shadow:0 8px 24px rgba(27,53,75,.07);margin:16px 0;}",
        ".grid2{display:grid;grid-template-columns:repeat(auto-fit,minmax(460px,1fr));gap:16px;}",
        ".grid4{display:grid;grid-template-columns:repeat(auto-fit,minmax(300px,1fr));gap:16px;}",
        ".plot{width:100%;height:auto;display:block;background:#fbfdff;border:1px solid #e7eef5;border-radius:14px;}",
        ".legend{display:flex;gap:14px;flex-wrap:wrap;font-size:12px;color:#526272;margin:6px 0 0;}",
        ".kicker{font-size:12px;letter-spacing:.08em;text-transform:uppercase;color:#6b7b8d;margin-bottom:8px;}",
        ".sw{display:inline-block;width:12px;height:3px;margin-right:6px;vertical-align:middle;}",
        "table{width:100%;border-collapse:collapse;font-size:14px;} th,td{padding:8px 10px;border-bottom:1px solid #e7eef5;text-align:left;vertical-align:top;} th{font-size:12px;letter-spacing:.04em;text-transform:uppercase;color:#526272;}",
        "code{background:#edf3f8;padding:1px 5px;border-radius:5px;}",
        "ul{margin:8px 0 0 20px;padding:0;} li{margin:4px 0;}",
        "</style></head><body><div class='wrap'>",
        "<h1>Tradeoff-Curve Update for Tim</h1>",
        "<p class='muted'>Prepared April 11, 2026. This memo summarizes the dense-curve follow-up using local probit fits. Visuals show only probit curves plus sampled point rates. Kernel estimates were used as a robustness check off-plot. Superseded admissions runs are omitted.</p>",
        "<div class='card'><div class='kicker'>Executive Summary</div><h2>What changed since last week</h2>",
        "<p>The main update is not just a switch from bisection to probit. The denser curve setup lets us separate three mechanisms that were previously confounded: generic reframing (<code>placebo</code>), neutral reflection, and directional scaffolding. It also lets us inspect how these interventions interact with thinking effort and model size.</p>",
        "<ul>",
        "<li><strong>AI labor / GPT-5.4:</strong> reflection raises the layoff threshold sharply under disabled thinking, but medium thinking pushes all four conditions toward lower thresholds and much lower order sensitivity. Productivity-focused scaffolding can reverse much of the worker-protective shift from neutral reflection.</li>",
        "<li><strong>SAT / Gemini Pro:</strong> reflection is the most stable condition and lands near the same midpoint under disabled and medium thinking (~380-392 SAT points), while preparedness reflection pushes further toward the higher-scoring applicant.</li>",
        "<li><strong>SAT / Gemini Flash-Lite:</strong> baseline and placebo are censored. Reflection is what makes a measurable tradeoff curve appear at all. Model size therefore changes not just the midpoint, but whether the latent tradeoff is estimable under naive prompting.</li>",
        "</ul></div>",
        "<div class='card'><div class='kicker'>Safety / Alignment Angle</div><h2>Why this matters beyond prompt engineering</h2>",
        "<p>The cleanest reading is that deliberative scaffolds are runtime control surfaces on local revealed preferences. They do not merely change phrasing or verbosity; they can move the indifference point, alter sharpness, and in smaller models even turn a censored decision rule into a measurable tradeoff curve.</p>",
        "<p>This cuts both ways for safety. It is potentially useful because runtime scaffolds can regularize and steer model behavior without retraining. It is also a vulnerability because those same scaffolds can directionally reshape tradeoffs in high-stakes domains. The current evidence suggests that both <strong>thinking budget</strong> and <strong>model size</strong> modulate how exposed a model is to that steering surface.</p>",
        "</div>",
        "<div class='card'><div class='kicker'>Family 1</div><h2>AI Labor on GPT-5.4</h2>",
        "<p>The estimand is the annual net savings per eliminated role where the model is 50/50 on endorsing layoffs. Lower midpoint means more willingness to automate and cut staff.</p>",
        "<p>Substantively: under disabled thinking, neutral reflection makes GPT-5.4 substantially more protective of workers than baseline, while productivity reflection pulls the model back toward baseline. Under medium thinking, the whole family shifts toward lower layoff thresholds and much lower order sensitivity. That suggests thinking effort is not just increasing coherence; it is changing the underlying revealed tradeoff.</p>",
        "<table><thead><tr><th>Thinking</th><th>Condition</th><th>Probit Midpoint</th><th>Probit Slope</th><th>Pseudo-R²</th><th>Kernel Midpoint</th><th>Mean Entropy</th><th>Order Gap</th></tr></thead><tbody>",
    ]

    for condition in AI_CONDITION_LABELS:
        lines.append(
            _build_summary_row(
                fit_row={**ai_disabled_fits[condition]},
                midpoint_formatter=_fmt_money,
                thinking_label="disabled",
                condition_label=AI_CONDITION_LABELS[condition],
                gap_formatter=_fmt_money,
            )
        )
    for condition in AI_CONDITION_LABELS:
        lines.append(
            _build_summary_row(
                fit_row={**ai_medium_fits[condition]},
                midpoint_formatter=_fmt_money,
                thinking_label="medium",
                condition_label=AI_CONDITION_LABELS[condition],
                gap_formatter=_fmt_money,
            )
        )

    lines.extend(
        [
            "</tbody></table>",
            "<div class='grid2'>",
            "<div>",
            "<h3>Disabled Thinking</h3>",
            _plot_svg(
                axis_values=ai_disabled["points"],
                grouped_points=ai_disabled_points,
                grouped_fits=ai_disabled_fits,
                labels=AI_CONDITION_LABELS,
                axis_formatter=lambda value: _fmt_money(value),
            ),
            "</div>",
            "<div>",
            "<h3>Medium Thinking</h3>",
            _plot_svg(
                axis_values=ai_medium["points"],
                grouped_points=ai_medium_points,
                grouped_fits=ai_medium_fits,
                labels=AI_CONDITION_LABELS,
                axis_formatter=lambda value: _fmt_money(value),
            ),
            "</div></div>",
            "<div class='legend'>",
        ]
    )
    for condition, label in AI_CONDITION_LABELS.items():
        lines.append(
            f"<span><span class='sw' style='background:{COLORS[condition]}'></span>{html.escape(label)}</span>"
        )
    lines.extend(
        [
            "</div>",
            "<p class='muted'>Interpretation: disabled reflection is the most worker-protective condition. Medium thinking does something different: it lowers thresholds across the board and nearly eliminates order effects, with the largest shift under productivity reflection. This looks like a real interaction between scaffold content and reasoning budget.</p>",
            "</div>",
            "<div class='card'><div class='kicker'>Family 2</div><h2>SAT / Admissions on Gemini Models</h2>",
            "<p>The estimand is the SAT gap where the model is 50/50 on choosing the higher-scoring advantaged candidate. Lower midpoint means the model switches to the higher-scoring candidate at a smaller SAT gap.</p>",
            "<p>For this family, Gemini Pro is the clean headline model. Reflection produces a stable midrange threshold under both disabled and medium thinking, while preparedness reflection pushes lower. Flash-Lite is interesting for a different reason: baseline and placebo remain saturated, so reflection is what makes a tradeoff curve estimable at all.</p>",
            "<table><thead><tr><th>Model</th><th>Thinking</th><th>Condition</th><th>Probit Midpoint</th><th>Probit Slope</th><th>Pseudo-R²</th><th>Kernel Midpoint</th><th>Mean Entropy</th><th>Order Gap</th></tr></thead><tbody>",
        ]
    )

    sat_order = [
        ("gemini-3.1-pro-preview", "disabled"),
        ("gemini-3.1-pro-preview", "medium"),
        ("gemini-3.1-flash-lite-preview", "disabled"),
        ("gemini-3.1-flash-lite-preview", "medium"),
    ]
    for model_short, effort in sat_order:
        fit_map = sat_fit_maps[(model_short, effort)]
        for condition in SAT_CONDITION_LABELS:
            lines.append(
                "<tr>"
                f"<td>{html.escape(model_short)}</td>"
                f"<td>{html.escape(effort)}</td>"
                f"<td>{html.escape(SAT_CONDITION_LABELS[condition])}</td>"
                f"<td>{html.escape(_fmt_sat(fit_map[condition]['probit'].get('midpoint_native')) if fit_map[condition]['probit'].get('midpoint_native') is not None else str(fit_map[condition]['probit'].get('midpoint_position')))}</td>"
                f"<td>{html.escape(_fmt_number(fit_map[condition]['probit'].get('slope')))}</td>"
                f"<td>{html.escape(_fmt_number(fit_map[condition]['probit'].get('pseudo_r2')))}</td>"
                f"<td>{html.escape(_fmt_sat(fit_map[condition]['kernel'].get('midpoint_native')) if fit_map[condition]['kernel'].get('midpoint_native') is not None else str(fit_map[condition]['kernel'].get('midpoint_position')))}</td>"
                f"<td>{html.escape(_fmt_number(fit_map[condition].get('mean_entropy')))}</td>"
                f"<td>{html.escape(_fmt_number(fit_map[condition].get('probit_order_gap')))}</td>"
                "</tr>"
            )

    lines.extend(["</tbody></table>", "<div class='grid4'>"])
    for model_short, effort in sat_order:
        title = f"{model_short} · {effort}"
        lines.append("<div>")
        lines.append(f"<h3>{html.escape(title)}</h3>")
        lines.append(
            _plot_svg(
                axis_values=sat_bundles[0]["analysis"]["points"],
                grouped_points=sat_point_maps[(model_short, effort)],
                grouped_fits=sat_fit_maps[(model_short, effort)],
                labels=SAT_CONDITION_LABELS,
                axis_formatter=lambda value: _fmt_sat(value),
            )
        )
        lines.append("</div>")
    lines.append("</div><div class='legend'>")
    for condition, label in SAT_CONDITION_LABELS.items():
        lines.append(
            f"<span><span class='sw' style='background:{COLORS[condition]}'></span>{html.escape(label)}</span>"
        )
    lines.extend(
        [
            "</div>",
            "<p class='muted'>Interpretation: model size changes the measurement regime. Gemini Pro exposes an in-range baseline curve; Flash-Lite does not. Reflection on Pro acts mostly as a stabilizer toward a similar midpoint under both thinking settings, whereas on Flash-Lite it acts more like a de-saturation scaffold that reveals a curve which naive prompting does not expose.</p>",
            "</div>",
            "<div class='card'><div class='kicker'>Bottom Line</div><h2>Takeaway for tomorrow</h2>",
            "<p>The evidence now supports a sharper claim than last week: reflective scaffolds are inference-time interventions on local tradeoff parameters, not just generic prompting tricks. The effect is heterogeneous. On GPT-5.4 AI labor, thinking effort materially changes the revealed threshold and can amplify directional scaffolding. On Gemini SAT, model size changes whether a baseline curve exists at all, while reflection often regularizes the curve toward a more stable and measurable boundary.</p>",
            "<p>That is safety-relevant because it means the same runtime mechanisms used for harmlessness and policy steering are also mechanisms by which high-stakes preferences can be shifted, stabilized, or exposed. The next step should therefore emphasize <strong>runtime steerability and scaffold sensitivity</strong>, not just “does reflection help.”</p>",
            "</div>",
            "</div></body></html>",
        ]
    )
    return "".join(lines)


def main() -> None:
    report_path = BASE_DIR / "reports" / "tim_report_20260411.html"
    report_path.write_text(render_report(), encoding="utf-8")
    print(f"report_path={report_path}")


if __name__ == "__main__":
    main()

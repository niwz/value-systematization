from __future__ import annotations

import html
import importlib.util
from pathlib import Path


BASE_DIR = Path("/Users/nicwong/Desktop/value-systematization/transitivity/advice_reflection_platform")
SOURCE_SCRIPT = BASE_DIR / "scripts" / "generate_tim_project_proposal_20260415.py"
REPORT_PATH = BASE_DIR / "reports" / "tim_project_slides_20260415.html"


def load_proposal_module():
    spec = importlib.util.spec_from_file_location("proposal_gen_20260415", SOURCE_SCRIPT)
    if spec is None or spec.loader is None:
        raise RuntimeError(f"Could not load source script: {SOURCE_SCRIPT}")
    module = importlib.util.module_from_spec(spec)
    spec.loader.exec_module(module)
    return module


def table_rows(rows: list[tuple[str, str, str, str, str]]) -> str:
    rendered: list[str] = []
    for c0, c1, c2, c3, c4 in rows:
        rendered.append(
            "<tr>"
            f"<td>{html.escape(c0)}</td>"
            f"<td>{html.escape(c1)}</td>"
            f"<td>{html.escape(c2)}</td>"
            f"<td>{html.escape(c3)}</td>"
            f"<td>{html.escape(c4)}</td>"
            "</tr>"
        )
    return "".join(rendered)


def fmt_k(value: float | None) -> str:
    if value is None:
        return "n/a"
    return f"${value/1000:.1f}k"


def fmt_sat(value: float | None) -> str:
    if value is None:
        return "n/a"
    return f"{value:.0f}"


def build_html() -> str:
    proposal = load_proposal_module()

    ai_disabled = proposal.ai_gpt54_disabled_combined()
    ai_medium = proposal.load_json(proposal.AI_GPT54_MEDIUM_FILE)
    sat_disabled = proposal.load_json(proposal.SAT_GEMINI_DISABLED_FILE)
    sat_medium = proposal.load_json(proposal.SAT_GEMINI_MEDIUM_FILE)
    convo = proposal.load_example_conversation()

    ai_disabled_fit, ai_disabled_points = proposal.get_pooled_rows(ai_disabled)
    ai_medium_fit, ai_medium_points = proposal.get_pooled_rows(ai_medium)
    sat_disabled_fit, sat_disabled_points = proposal.get_pooled_rows(sat_disabled)
    sat_medium_fit, sat_medium_points = proposal.get_pooled_rows(sat_medium)

    ai_disabled_plot = proposal._plot_svg(
        fit_rows=ai_disabled_fit,
        point_rows=ai_disabled_points,
        title="AI labor: GPT-5.4, disabled thinking",
        x_label="Annual net savings per eliminated role",
        x_formatter=proposal.ai_x_formatter,
    )
    ai_medium_plot = proposal._plot_svg(
        fit_rows=ai_medium_fit,
        point_rows=ai_medium_points,
        title="AI labor: GPT-5.4, medium thinking",
        x_label="Annual net savings per eliminated role",
        x_formatter=proposal.ai_x_formatter,
    )
    sat_disabled_plot = proposal._plot_svg(
        fit_rows=sat_disabled_fit,
        point_rows=sat_disabled_points,
        title="SAT / admissions: Gemini Pro, disabled thinking",
        x_label="SAT gap",
        x_formatter=proposal.sat_x_formatter,
    )
    sat_medium_plot = proposal._plot_svg(
        fit_rows=sat_medium_fit,
        point_rows=sat_medium_points,
        title="SAT / admissions: Gemini Pro, medium thinking",
        x_label="SAT gap",
        x_formatter=proposal.sat_x_formatter,
    )

    ai_disabled_map = {row["condition"]: row for row in ai_disabled_fit}
    ai_medium_map = {row["condition"]: row for row in ai_medium_fit}
    sat_disabled_map = {row["condition"]: row for row in sat_disabled_fit}
    sat_medium_map = {row["condition"]: row for row in sat_medium_fit}

    ai_rows = table_rows(
        [
            (
                "Baseline",
                fmt_k(ai_disabled_map["baseline"]["probit"].get("midpoint_native")),
                fmt_k(ai_disabled_map["baseline"]["kernel"].get("midpoint_native")),
                fmt_k(ai_medium_map["baseline"]["probit"].get("midpoint_native")),
                fmt_k(ai_medium_map["baseline"]["kernel"].get("midpoint_native")),
            ),
            (
                "Placebo",
                fmt_k(ai_disabled_map["placebo"]["probit"].get("midpoint_native")),
                fmt_k(ai_disabled_map["placebo"]["kernel"].get("midpoint_native")),
                fmt_k(ai_medium_map["placebo"]["probit"].get("midpoint_native")),
                fmt_k(ai_medium_map["placebo"]["kernel"].get("midpoint_native")),
            ),
            (
                "Reflection",
                fmt_k(ai_disabled_map["reflection"]["probit"].get("midpoint_native")),
                fmt_k(ai_disabled_map["reflection"]["kernel"].get("midpoint_native")),
                fmt_k(ai_medium_map["reflection"]["probit"].get("midpoint_native")),
                fmt_k(ai_medium_map["reflection"]["kernel"].get("midpoint_native")),
            ),
            (
                "Productivity reflection",
                fmt_k(ai_disabled_map["productivity_reflection"]["probit"].get("midpoint_native")),
                fmt_k(ai_disabled_map["productivity_reflection"]["kernel"].get("midpoint_native")),
                fmt_k(ai_medium_map["productivity_reflection"]["probit"].get("midpoint_native")),
                fmt_k(ai_medium_map["productivity_reflection"]["kernel"].get("midpoint_native")),
            ),
        ]
    )

    sat_rows = table_rows(
        [
            (
                "Baseline",
                fmt_sat(sat_disabled_map["baseline"]["probit"].get("midpoint_native")),
                fmt_sat(sat_disabled_map["baseline"]["kernel"].get("midpoint_native")),
                fmt_sat(sat_medium_map["baseline"]["probit"].get("midpoint_native")),
                fmt_sat(sat_medium_map["baseline"]["kernel"].get("midpoint_native")),
            ),
            (
                "Placebo",
                fmt_sat(sat_disabled_map["placebo"]["probit"].get("midpoint_native")),
                fmt_sat(sat_disabled_map["placebo"]["kernel"].get("midpoint_native")),
                fmt_sat(sat_medium_map["placebo"]["probit"].get("midpoint_native")),
                fmt_sat(sat_medium_map["placebo"]["kernel"].get("midpoint_native")),
            ),
            (
                "Reflection",
                fmt_sat(sat_disabled_map["reflection"]["probit"].get("midpoint_native")),
                fmt_sat(sat_disabled_map["reflection"]["kernel"].get("midpoint_native")),
                fmt_sat(sat_medium_map["reflection"]["probit"].get("midpoint_native")),
                fmt_sat(sat_medium_map["reflection"]["kernel"].get("midpoint_native")),
            ),
            (
                "Preparedness reflection",
                fmt_sat(sat_disabled_map["preparedness_reflection"]["probit"].get("midpoint_native")),
                fmt_sat(sat_disabled_map["preparedness_reflection"]["kernel"].get("midpoint_native")),
                fmt_sat(sat_medium_map["preparedness_reflection"]["probit"].get("midpoint_native")),
                fmt_sat(sat_medium_map["preparedness_reflection"]["kernel"].get("midpoint_native")),
            ),
        ]
    )

    turn1_user_lines = "\n".join(convo["turn1_user"].splitlines()[:8])
    turn1_assistant_lines = "\n".join(convo["turn1_assistant"].splitlines()[:10])
    turn2_user_lines = "\n".join(convo["turn2_user"].splitlines()[-8:])
    turn2_assistant_lines = convo["turn2_assistant"]

    return f"""<!doctype html>
<html lang="en">
<head>
  <meta charset="utf-8">
  <meta name="viewport" content="width=device-width, initial-scale=1">
  <title>Runtime Scaffolds and Revealed Tradeoff Curves — Slides</title>
  <style>
    :root {{
      --bg:#edf3f8;
      --slide:#ffffff;
      --soft:#f5f8fb;
      --text:#15212b;
      --muted:#526272;
      --line:#d5e0e8;
      --accent:#1f5f8b;
      --accent2:#9b2226;
    }}
    * {{ box-sizing:border-box; }}
    body {{
      margin:0;
      background:var(--bg);
      color:var(--text);
      font-family: ui-sans-serif, -apple-system, BlinkMacSystemFont, "Segoe UI", sans-serif;
    }}
    .deck {{
      padding:24px;
    }}
    .slide {{
      width:min(1280px, calc(100vw - 48px));
      aspect-ratio:16 / 9;
      min-height:0;
      margin:0 auto 24px auto;
      padding:34px 40px 30px 40px;
      background:var(--slide);
      border:1px solid var(--line);
      border-radius:22px;
      box-shadow:0 10px 28px rgba(23,37,50,.08);
      display:flex;
      flex-direction:column;
      justify-content:flex-start;
      page-break-after:always;
      break-after:page;
    }}
    .kicker {{
      font-size:12px;
      letter-spacing:.08em;
      text-transform:uppercase;
      color:var(--accent);
      font-weight:700;
      margin-bottom:12px;
    }}
    h1,h2,h3 {{
      margin:0 0 10px 0;
      line-height:1.1;
    }}
    h1 {{ font-size:42px; }}
    h2 {{ font-size:30px; }}
    h3 {{ font-size:18px; }}
    p {{
      margin:0 0 12px 0;
      line-height:1.45;
    }}
    .lede {{
      font-size:21px;
      max-width:980px;
      color:#243645;
    }}
    .grid-2 {{
      display:grid;
      grid-template-columns:1fr 1fr;
      gap:18px;
    }}
    .grid-3 {{
      display:grid;
      grid-template-columns:repeat(3, 1fr);
      gap:16px;
    }}
    .title-slide .title-body {{
      display:flex;
      flex-direction:column;
      flex:1;
    }}
    .title-slide .hero-panels {{
      display:flex;
      gap:16px;
      flex:1;
      align-items:stretch;
      margin-top:24px;
    }}
    .title-slide .hero-panels .box {{
      flex:1;
      height:100%;
      display:flex;
      flex-direction:column;
    }}
    .box {{
      background:var(--soft);
      border:1px solid var(--line);
      border-radius:16px;
      padding:16px 18px;
    }}
    ul {{
      margin:8px 0 0 20px;
    }}
    li {{
      margin:8px 0;
      line-height:1.4;
    }}
    .formula {{
      font-family: ui-monospace, SFMono-Regular, Menlo, monospace;
      background:#edf4f9;
      border-radius:10px;
      padding:10px 12px;
      display:inline-block;
      margin:6px 0 10px 0;
    }}
    .plot {{
      background:#fbfdff;
      border:1px solid var(--line);
      border-radius:18px;
      padding:12px;
    }}
    .plot svg {{
      width:100%;
      height:auto;
      display:block;
    }}
    table {{
      width:100%;
      border-collapse:collapse;
      font-size:14px;
      margin-top:10px;
    }}
    th, td {{
      padding:10px 12px;
      border-bottom:1px solid var(--line);
      text-align:left;
      vertical-align:top;
    }}
    th {{
      color:var(--muted);
      font-size:12px;
      letter-spacing:.06em;
      text-transform:uppercase;
    }}
    .callout {{
      background:#f8fbfd;
      border-left:4px solid var(--accent);
      border-radius:8px;
      padding:14px 16px;
      margin-top:10px;
    }}
    .transcript {{
      display:grid;
      grid-template-columns:1fr 1fr;
      gap:12px;
      margin-top:8px;
    }}
    .msg {{
      border:1px solid var(--line);
      border-radius:14px;
      background:#fbfdff;
      padding:12px 14px;
    }}
    .role {{
      font-size:11px;
      letter-spacing:.07em;
      text-transform:uppercase;
      color:var(--muted);
      font-weight:700;
      margin-bottom:8px;
    }}
    pre {{
      margin:0;
      white-space:pre-wrap;
      word-break:break-word;
      font-family: ui-monospace, SFMono-Regular, Menlo, monospace;
      font-size:12px;
      line-height:1.35;
    }}
    .footer {{
      margin-top:18px;
      padding-top:12px;
      color:var(--muted);
      font-size:12px;
    }}
    @media print {{
      @page {{
        size: 13.333in 7.5in;
        margin: 0.25in;
      }}
      body {{
        background:#ffffff;
      }}
      .deck {{
        padding:0;
      }}
      .slide {{
        width:auto;
        height:7in;
        aspect-ratio:auto;
        margin:0 0 0.18in 0;
        box-shadow:none;
      }}
    }}
  </style>
</head>
<body>
  <div class="deck">
    <section class="slide title-slide">
      <div class="title-body">
        <div class="kicker">Prospective Project</div>
        <h1>Runtime Scaffolds and Revealed Tradeoff Curves</h1>
        <p class="lede">Measure whether simple inference-time scaffolds such as placebo restatement, reflection, and compact constitutions systematically move a language model’s local decision threshold on recurring tradeoff families.</p>
        <div class="hero-panels">
          <div class="box">
            <h3>Core Question</h3>
            <p>Do runtime scaffolds change a model’s <strong>revealed threshold</strong>, not just the wording of a single answer?</p>
          </div>
          <div class="box">
            <h3>Main Object</h3>
            <p>A local binary choice curve, summarized by the <strong>midpoint</strong> where the model becomes 50/50 on the event.</p>
          </div>
          <div class="box">
            <h3>Why It Matters</h3>
            <p>If the midpoint moves, runtime context is acting as a control surface on local revealed preferences.</p>
          </div>
        </div>
        <div class="footer">Save to PDF by printing this HTML in landscape mode.</div>
      </div>
    </section>

    <section class="slide">
      <div class="kicker">Motivation</div>
      <h2>Why This Is an Alignment / Deployment Problem</h2>
      <div class="grid-2" style="margin-top:10px;">
        <div class="box">
          <h3>What We Are Not Claiming</h3>
          <ul>
            <li>Not a global utility function over arbitrary outcomes.</li>
            <li>Not a catalog of isolated prompt quirks.</li>
            <li>Not a claim that one scaffold is universally “better.”</li>
          </ul>
        </div>
        <div class="box">
          <h3>What We Are Claiming</h3>
          <ul>
            <li>For a fixed family, the model reveals a local choice boundary.</li>
            <li>Runtime scaffolds can move that boundary.</li>
            <li>The size and direction of the shift depend on family, model, and thinking setting.</li>
          </ul>
        </div>
      </div>
      <div class="callout">
        <strong>Interpretation:</strong> the project measures local revealed preferences under controlled runtime interventions.
      </div>
    </section>

    <section class="slide">
      <div class="kicker">Method</div>
      <h2>Measurement Design</h2>
      <div class="grid-2" style="margin-top:10px;">
        <div class="box">
          <h3>Experimental Unit</h3>
          <ul>
            <li>Choose a family of closely related tradeoff decisions.</li>
            <li>Vary one scalar axis only.</li>
            <li>Sample binary choices at multiple rungs.</li>
          </ul>
          <p><strong>Examples</strong></p>
          <ul>
            <li>AI labor: annual net savings per eliminated role.</li>
            <li>SAT / admissions: score gap between two marginal applicants.</li>
          </ul>
        </div>
        <div class="box">
          <h3>Runtime Flow</h3>
          <ul>
            <li>Generate one scaffold on the <strong>same model</strong>.</li>
            <li>Freeze both the scaffold prompt and scaffold response within-run.</li>
            <li>Attach that frozen exchange to each fresh choice query.</li>
            <li>Estimate a local choice curve from repeated binary decisions.</li>
          </ul>
        </div>
      </div>
      <div class="grid-3" style="margin-top:16px;">
        <div class="box"><h3>Baseline</h3><p>No substantive scaffold beyond the shared flow.</p></div>
        <div class="box"><h3>Placebo</h3><p>Factual restatement or summary.</p></div>
        <div class="box"><h3>Reflection / Constitution</h3><p>Principled reflection or compact decision rules.</p></div>
      </div>
    </section>

    <section class="slide">
      <div class="kicker">Method</div>
      <h2>Why Probit, and Why Kernel Too</h2>
      <div class="grid-2" style="margin-top:10px;">
        <div class="box">
          <h3>Parametric Fit</h3>
          <div class="formula">ΔU = α + β f(x) + ε, &nbsp; ε ~ N(0, σ²)</div>
          <div class="formula">P(event | x) = Φ(α + β f(x))</div>
          <ul>
            <li><code>f(x)</code> is the transformed axis.</li>
            <li>Use <code>log10(x)</code> for money-like quantities.</li>
            <li>Use identity for additive scales like SAT gap.</li>
            <li>The key summary is the <strong>midpoint</strong>, where <code>P(event)=0.5</code>.</li>
          </ul>
        </div>
        <div class="box">
          <h3>Non-Parametric Check</h3>
          <ul>
            <li>Fit a Gaussian-kernel smoother on the transformed axis.</li>
            <li>No single-sigmoid assumption.</li>
            <li>Kernel midpoint = 0.5 crossing of the smoothed empirical curve.</li>
            <li>Kernel 25–75 width = rough transition-band measure.</li>
          </ul>
          <div class="callout">
            When probit and kernel midpoints agree, the midpoint estimate is less likely to be a fitting artifact.
          </div>
        </div>
      </div>
    </section>

    <section class="slide">
      <div class="kicker">Method</div>
      <h2>Representative Runtime Transcript</h2>
      <p>This is a concrete example of the frozen-scaffold flow. The exact wording can vary, but the measurement logic stays the same.</p>
      <div class="transcript">
        <div class="msg">
          <div class="role">System</div>
          <pre>{html.escape(convo["system"])}</pre>
        </div>
        <div class="msg">
          <div class="role">User (Frozen Turn 1)</div>
          <pre>{html.escape(turn1_user_lines)}</pre>
        </div>
        <div class="msg">
          <div class="role">Assistant (Frozen Scaffold)</div>
          <pre>{html.escape(turn1_assistant_lines)}</pre>
        </div>
        <div class="msg">
          <div class="role">User + Assistant (Sampled Turn 2)</div>
          <pre>{html.escape(turn2_user_lines)}

{html.escape(turn2_assistant_lines)}</pre>
        </div>
      </div>
    </section>

    <section class="slide">
      <div class="kicker">Results</div>
      <h2>AI Labor: GPT-5.4 Is the Main Headline</h2>
      <div class="grid-2" style="margin-top:10px; align-items:start;">
        <div class="plot">{ai_disabled_plot}</div>
        <div class="plot">{ai_medium_plot}</div>
      </div>
      <table>
        <thead>
          <tr>
            <th>Condition</th>
            <th>Disabled probit</th>
            <th>Disabled kernel</th>
            <th>Medium probit</th>
            <th>Medium kernel</th>
          </tr>
        </thead>
        <tbody>{ai_rows}</tbody>
      </table>
      <div class="callout">
        <strong>Most important result:</strong> on disabled GPT-5.4, reflection moved the layoff threshold from <strong>$45.9k</strong> to <strong>$75.2k</strong>. Placebo moved it to <strong>$60.9k</strong>. Productivity-centered reflection pulled it back toward baseline. Under medium thinking, the whole family shifted lower overall.
      </div>
    </section>

    <section class="slide">
      <div class="kicker">Results</div>
      <h2>SAT / Admissions: Cross-Family Evidence from Gemini Pro</h2>
      <div class="grid-2" style="margin-top:10px; align-items:start;">
        <div class="plot">{sat_disabled_plot}</div>
        <div class="plot">{sat_medium_plot}</div>
      </div>
      <table>
        <thead>
          <tr>
            <th>Condition</th>
            <th>Disabled probit</th>
            <th>Disabled kernel</th>
            <th>Medium probit</th>
            <th>Medium kernel</th>
          </tr>
        </thead>
        <tbody>{sat_rows}</tbody>
      </table>
      <div class="callout">
        <strong>Read:</strong> lower midpoint means willingness to choose the higher-scoring advantaged candidate at a smaller SAT gap. Reflection is relatively stable across thinking settings. Preparedness reflection lowers the threshold further.
      </div>
    </section>

    <section class="slide">
      <div class="kicker">Interpretation</div>
      <h2>What the Current Results Suggest</h2>
      <div class="grid-2" style="margin-top:10px;">
        <div class="box">
          <h3>Main empirical point</h3>
          <ul>
            <li>The object of interest is a curve and its midpoint, not a single answer.</li>
            <li>Runtime scaffolds move threshold-like parameters.</li>
            <li>Placebo matters, but reflection often does more than placebo.</li>
          </ul>
        </div>
        <div class="box">
          <h3>Important nuance</h3>
          <ul>
            <li>Thinking is a moderator, not a monotone “better reasoning” knob.</li>
            <li>Effects differ by family and by model.</li>
            <li>The right framing is local revealed-preference control, not universal value change.</li>
          </ul>
        </div>
      </div>
      <div class="callout">
        <strong>Alignment angle:</strong> if simple runtime context shifts a model’s local choice boundary in a measurable way, that is a deployment-relevant control surface and a measurement target in its own right.
      </div>
    </section>

    <section class="slide">
      <div class="kicker">Next Steps</div>
      <h2>What the Project Would Do Next</h2>
      <div class="grid-2" style="margin-top:10px;">
        <div class="box">
          <h3>Immediate Work</h3>
          <ul>
            <li>Standardize the frozen-scaffold methodology across families.</li>
            <li>Rerun strongest families under the refined natural-language prompt surface.</li>
            <li>Reintroduce AB/BA order robustness once the prompt surface is fixed.</li>
          </ul>
        </div>
        <div class="box">
          <h3>Research Payoff</h3>
          <ul>
            <li>Cross-model comparisons under one estimand.</li>
            <li>Cross-family comparisons under one midpoint-based framework.</li>
            <li>A reusable method for measuring runtime preference steering.</li>
          </ul>
        </div>
      </div>
      <div class="callout">
        <strong>One-sentence pitch:</strong> estimate local revealed tradeoff curves for language models, then test whether simple runtime scaffolds move those curves in systematic, measurable ways.
      </div>
      <div class="footer">Export: open this HTML in a browser and print to PDF in landscape mode.</div>
    </section>
  </div>
</body>
</html>
"""


def generate() -> None:
    REPORT_PATH.write_text(build_html(), encoding="utf-8")
    print(REPORT_PATH)


if __name__ == "__main__":
    generate()

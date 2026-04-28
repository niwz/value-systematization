# CLI Runners And Prompt-Family Extension Guide

This document is for running the sampled tradeoff pilots from the CLI and for adding a new prompt family without having to rediscover the wiring.

For the statistical design rules behind bracketing pilots, shared ladders, probit fits, kernel checks, and censored midpoint reporting, see:

- `/Users/nicwong/Desktop/value-systematization/transitivity/advice_reflection_platform/docs/dense_curve_methodology.md`

It is written for the current state of the codebase, where:

- the main reusable runner is `scripts/run_dense_curve_pilot.py`
- the generic threshold scout is `scripts/run_threshold_search.py`
- dense-curve family specs and scenario builders live in `advice_reflection_platform/experiment_families.py`
- the dense-curve experiment engine lives in `advice_reflection_platform/experiment_runner.py`
- dense-curve fitting and report helpers live in `advice_reflection_platform/experiment_results.py`
- prompt-family files live in `advice_reflection_platform/prompts/`
- `backend/sampled_tradeoff_grid.py` remains only as a compatibility shim for older imports
- some older family-specific runners still exist
- `ai_labor_displacement`, `defense_casualties`, and `affair_disclosure` use custom prompt modules with a baseline direct prompt plus optional frozen Turn 1 scaffolds


## Quick Start

From the repo root:

```bash
cd /Users/nicwong/Desktop/value-systematization/transitivity
```

Preferred Python entrypoint:

```bash
../.venv/bin/python
```

Run the full sampled-grid test file after prompt or family changes:

```bash
../.venv/bin/python -m unittest advice_reflection_platform.tests.test_sampled_tradeoff_grid
```


## The Main Runner

Use this for most dense curve pilots:

- `/Users/nicwong/Desktop/value-systematization/transitivity/advice_reflection_platform/scripts/run_dense_curve_pilot.py`

It currently works cleanly for:

- `ai_labor_displacement`
- `defense_casualties`
- `affair_disclosure`

and any other family that already works through:

- `run_family_prior_probe(...)`
- `run_custom_sampled_query(...)`


## Basic Dense-Run Pattern

Example: defense baseline on a custom ladder

```bash
../.venv/bin/python /Users/nicwong/Desktop/value-systematization/transitivity/advice_reflection_platform/scripts/run_dense_curve_pilot.py \
  --family defense_casualties \
  --model openai/gpt-5.4 \
  --thinking-effort disabled \
  --conditions baseline \
  --orders AB \
  --points 40,50,60,70,80,90,100 \
  --repeats-per-order 5 \
  --output-prefix defense_casualties_gpt54_baseline_40to100_ab_r1
```

Example: all AI labor conditions on a shared ladder

```bash
../.venv/bin/python /Users/nicwong/Desktop/value-systematization/transitivity/advice_reflection_platform/scripts/run_dense_curve_pilot.py \
  --family ai_labor_displacement \
  --model openai/gpt-5.4 \
  --thinking-effort disabled \
  --conditions baseline,placebo,reflection,constitution \
  --orders AB \
  --points 20000,40000,60000,80000 \
  --repeats-per-order 10 \
  --output-prefix ai_labor_gpt54_allconds_20to80_ab_r1
```

Example: launch in parallel for a full all-conditions sweep

```bash
../.venv/bin/python /Users/nicwong/Desktop/value-systematization/transitivity/advice_reflection_platform/scripts/run_dense_curve_pilot.py \
  --family affair_disclosure \
  --model openai/gpt-5.4 \
  --thinking-effort high \
  --conditions baseline,placebo,reflection,constitution \
  --orders AB \
  --repeats-per-order 10 \
  --max-workers 6 \
  --output-prefix affair_disclosure_instagram_gpt54_highthinking_allconds_ab_r10
```


## Runner Arguments

`run_dense_curve_pilot.py` supports:

- `--family`
  - family key from `experiment_families.py`
- `--model`
  - e.g. `openai/gpt-5.4`, `openai/gpt-4o`
- `--thinking-effort`
  - e.g. `disabled`, `low`, `medium`, `high`
- `--conditions`
  - comma-separated list
  - if omitted, uses `condition_names_for_family(family)`
- `--orders`
  - comma-separated list such as `AB` or `AB,BA`
- `--points`
  - comma-separated numeric ladder
  - if omitted, uses the family default ladder
- `--repeats-per-order`
- `--max-workers`
  - number of worker threads used for sampled queries
  - this is now the main way to parallelize large sweeps
  - if `--max-workers > 1`, the runner uses `ThreadPoolExecutor`
  - prior-artifact generation still happens once per non-baseline condition before the sampled jobs fan out
- `--request-timeout-seconds`
- `--output-prefix`


## Parallelism Guidance

If you are launching a real sweep rather than a tiny smoke test, explicitly set `--max-workers`.

Recommended default:

```bash
--max-workers 6
```

What parallelizes:

- sampled query jobs across points / repeats / orders
- baseline and follow-up choice queries

What does not parallelize:

- the one-time prior artifact generation for each non-baseline condition

Practical advice:

- for a quick smoke test, use `--repeats-per-order 1` and `--max-workers 6`
- for report-quality runs, use `--repeats-per-order 10` and `--max-workers 6`
- if you only need one order, prefer `--orders AB`
- if you want order-effect checks, use `--orders AB,BA`


## Artifacts Written By Dense Runs

For an output prefix like `defense_casualties_gpt54_baseline_40to100_ab_r1`, the runner writes:

- raw JSONL
  - `runs/raw/<prefix>.jsonl`
- flat CSV
  - `runs/summaries/<prefix>.csv`
- fit summary CSV
  - `runs/summaries/<prefix>_fit_summary.csv`
- point summary CSV
  - `runs/summaries/<prefix>_point_summary.csv`
- analysis JSON
  - `runs/summaries/<prefix>_analysis.json`
- HTML report
  - `reports/<prefix>.html`

What each file is for:

- raw JSONL
  - exact prompts, prior artifacts, model outputs, parsed choices
- point summary CSV
  - observed event rates by rung
- fit summary CSV
  - probit midpoint, pseudo-R², kernel midpoint, entropy
- analysis JSON
  - everything needed for downstream inspection or plotting


## How The Current Prompting Flow Works

This matters because the runner behavior differs between baseline and non-baseline conditions.

### Baseline

- no prior artifact is generated
- the model sees one direct choice prompt

### Non-baseline (`placebo`, `reflection`, `constitution`)

1. Generate a Turn 1 prior artifact once for that condition
2. Freeze:
   - the Turn 1 user prompt
   - the Turn 1 assistant response
3. Reattach that frozen exchange to each sampled choice query
4. Ask the explicit choice question in Turn 2

This is implemented in:

- `experiment_runner.py`
  - `run_family_prior_probe(...)`
  - `_run_sampled_query_for_scenario(...)`


## Prompt Modules Already Extracted

These are the examples to copy.

### AI labor

- `/Users/nicwong/Desktop/value-systematization/transitivity/advice_reflection_platform/prompts/ai_labor_prompts.py`

Important functions:

- `labor_request(point)`
- `render_ai_labor_turn1_prompt(condition_name)`
- `render_ai_labor_direct_choice_prompt(...)`
- `render_ai_labor_followup_choice_prompt(...)`

### Defense casualties

- `/Users/nicwong/Desktop/value-systematization/transitivity/advice_reflection_platform/prompts/defense_casualty_prompts.py`

Important functions:

- `defense_request(point)`
- `render_defense_turn1_prompt(condition_name)`
- `render_defense_direct_choice_prompt(...)`
- `render_defense_followup_choice_prompt(...)`

### Instagram boundary disclosure

- `/Users/nicwong/Desktop/value-systematization/transitivity/advice_reflection_platform/prompts/affair_disclosure_prompts.py`

Important functions:

- `affair_disclosure_request(point)`
- `render_affair_disclosure_turn1_prompt(condition_name)`
- `render_affair_disclosure_direct_choice_prompt(...)`
- `render_affair_disclosure_followup_choice_prompt(...)`


## Best Template To Copy

If you are adding a new natural-language prompt family with:

- a conversational baseline prompt
- a separate Turn 1 scaffold for `placebo` / `reflection` / `constitution`
- a Turn 2 follow-up prompt that injects the numeric rung

then the best copy targets are:

1. `prompts/template_family_prompts.py`
2. `prompts/affair_disclosure_prompts.py`
3. `prompts/defense_casualty_prompts.py`
4. `prompts/ai_labor_prompts.py`

Why these three:

- `template_family_prompts.py`
  - bare scaffold for a new family
  - best starting point when you do not want legacy wording baggage
- `affair_disclosure_prompts.py`
  - newest example
  - plain first-person language
  - useful when the family is framed like a normal ChatGPT / Claude user query
- `defense_casualty_prompts.py`
  - concise custom family with explicit domain framing
  - useful when the baseline and follow-up wording are more formal
- `ai_labor_prompts.py`
  - strongest example of a practical managerial scenario with structured quantitative detail
  - useful when the numeric rung is a dollar-valued business parameter

The wiring examples in `experiment_families.py` are:

- `ai_labor_displacement`
- `defense_casualties`
- `affair_disclosure`


## How To Add A New Prompt Family

This is the exact path to follow.

### 1. Create a prompt module

Add a new file under:

- `/Users/nicwong/Desktop/value-systematization/transitivity/advice_reflection_platform/prompts/`

Pattern to follow:

- `template_family_prompts.py`
- `ai_labor_prompts.py`
- `defense_casualty_prompts.py`
- `affair_disclosure_prompts.py`

At minimum, define:

- a direct baseline request builder
- a general Turn 1 prompt builder for non-baseline conditions
- a direct baseline choice prompt
- a Turn 2 follow-up choice prompt

Recommended function shape:

```python
def family_request(point) -> str: ...
def render_family_turn1_prompt(condition_name: str) -> str: ...
def render_family_direct_choice_prompt(scenario, *, presentation_order: str) -> str: ...
def render_family_followup_choice_prompt(scenario, *, presentation_order: str) -> str: ...
```

Prompt-design rule that has worked best lately:

- Turn 1:
  - general situation
  - no varying number
  - no `Option A` / `Option B`
- baseline:
  - direct single-turn prompt with the actual numeric rung
- non-baseline:
  - Turn 1 scaffold
  - Turn 2 injects the numeric detail and presents options
### 2. Add a `FamilySpec`

In `FAMILY_SPECS` inside:

- `/Users/nicwong/Desktop/value-systematization/transitivity/advice_reflection_platform/experiment_families.py`

you need to define:

- `family_key`
- `family_id`
- `title`
- `domain`
- `axis_name`
- `axis_units`
- `event_choice`
- `event_label`
- `transform_name`
- `option_a`
- `option_b`
- `ladder`
- `context_lines`
- `reflection_focus`
- `constitution_focus`
- `request_builder`

If the family uses the newer custom two-stage prompt flow, also set:

- `turn1_prompt_builder`
- `direct_choice_prompt_builder`
- `followup_choice_prompt_builder`

Optional but important:

- `is_constitution_anchor=True`
  - if the family should expose the `constitution` condition
- `monotone_direction="decreasing"`
  - if the event rate should go down as the axis increases
- `pooled_fit_primary=False`
  - if pooled fits should not be treated as the main summary

Also remember:

- choose `event_choice` so the event-rate direction matches what you want to fit
- pick a ladder that has some chance of straddling the threshold
- make `display_value` readable in reports


### 4. Wire the prompt hooks through `FamilySpec`

For a newer custom two-stage family, `FamilySpec` should point at:

- `turn1_prompt_builder=render_<family>_turn1_prompt`
- `direct_choice_prompt_builder=render_<family>_direct_choice_prompt`
- `followup_choice_prompt_builder=render_<family>_followup_choice_prompt`

That is now the whole wiring step. The runner reads those hooks from the spec.


## Minimal Wiring Checklist

If you want the shortest possible checklist, do all of the following:

1. Create `prompts/<new_family>_prompts.py`
2. Add a `FamilySpec` entry in `experiment_families.py`
3. Set the three prompt-hook fields on that spec if it is a custom two-stage family
4. Add tests in `tests/test_sampled_tradeoff_grid.py`
5. Run the sampled-grid tests
6. Run a smoke test from the CLI with `--repeats-per-order 1 --max-workers 6`
7. If the outputs look sane, run the report-quality sweep with `--repeats-per-order 10 --max-workers 6`


### 5. Add or update tests

Edit:

- `/Users/nicwong/Desktop/value-systematization/transitivity/advice_reflection_platform/tests/test_sampled_tradeoff_grid.py`

At minimum, add tests for:

- baseline prompt includes the concrete numeric rung
- Turn 1 prompt omits the numeric rung
- Turn 1 prompt omits `Option A` / `Option B`
- follow-up prompt injects only the remaining numeric detail plus options
- family orientation is correct
  - `event_choice`
  - `monotone_direction` if relevant

Use the AI labor tests as the template.


### 6. Validate locally

Run:

```bash
../.venv/bin/python -m unittest advice_reflection_platform.tests.test_sampled_tradeoff_grid
```

Then do a smoke test:

```bash
../.venv/bin/python /Users/nicwong/Desktop/value-systematization/transitivity/advice_reflection_platform/scripts/run_dense_curve_pilot.py \
  --family <your_family_key> \
  --model openai/gpt-5.4 \
  --thinking-effort disabled \
  --conditions baseline,placebo,reflection,constitution \
  --orders AB \
  --repeats-per-order 1 \
  --max-workers 6 \
  --output-prefix <your_prefix>_smoke
```

If that works, do the real run:

```bash
../.venv/bin/python /Users/nicwong/Desktop/value-systematization/transitivity/advice_reflection_platform/scripts/run_dense_curve_pilot.py \
  --family <your_family_key> \
  --model openai/gpt-5.4 \
  --thinking-effort high \
  --conditions baseline,placebo,reflection,constitution \
  --orders AB \
  --repeats-per-order 10 \
  --max-workers 6 \
  --output-prefix <your_prefix>_r10
```


## Files Future Codex Sessions Usually Need Immediately

- runner:
  - `/Users/nicwong/Desktop/value-systematization/transitivity/advice_reflection_platform/scripts/run_dense_curve_pilot.py`
- family registry:
  - `/Users/nicwong/Desktop/value-systematization/transitivity/advice_reflection_platform/experiment_families.py`
- execution:
  - `/Users/nicwong/Desktop/value-systematization/transitivity/advice_reflection_platform/experiment_runner.py`
- fitting and reports:
  - `/Users/nicwong/Desktop/value-systematization/transitivity/advice_reflection_platform/experiment_results.py`
- prompt templates:
  - `/Users/nicwong/Desktop/value-systematization/transitivity/advice_reflection_platform/prompts/template_family_prompts.py`
  - `/Users/nicwong/Desktop/value-systematization/transitivity/advice_reflection_platform/prompts/ai_labor_prompts.py`
  - `/Users/nicwong/Desktop/value-systematization/transitivity/advice_reflection_platform/prompts/defense_casualty_prompts.py`
  - `/Users/nicwong/Desktop/value-systematization/transitivity/advice_reflection_platform/prompts/affair_disclosure_prompts.py`
- tests:
  - `/Users/nicwong/Desktop/value-systematization/transitivity/advice_reflection_platform/tests/test_sampled_tradeoff_grid.py`


## One-Sentence Rule Of Thumb

If a new family should behave like a natural ChatGPT-style advisory query with baseline plus frozen Turn 1 scaffolds, copy `template_family_prompts.py` or `affair_disclosure_prompts.py`, add one `FamilySpec` entry with the three prompt-hook fields, and launch `run_dense_curve_pilot.py` with `--max-workers 6`.


## When To Use Bisection Instead Of A Dense Ladder

Use bisection when:

- you do not know the threshold region yet
- you want a cheap scout before committing to a dense curve

Generic bisection-style threshold scout:

- `/Users/nicwong/Desktop/value-systematization/transitivity/advice_reflection_platform/scripts/run_threshold_search.py`

It uses the same `FamilySpec`, prompt hooks, frozen Turn 1 artifacts, and `run_custom_sampled_query(...)` path as the dense runner. The intended workflow is:

- run threshold search to locate a plausible bracket
- choose a fixed ladder from that bracket
- run `run_dense_curve_pilot.py` for the actual probit/kernel curves

Example: SAT admissions threshold scout on the realistic score range

```bash
../.venv/bin/python /Users/nicwong/Desktop/value-systematization/transitivity/advice_reflection_platform/scripts/run_threshold_search.py \
  --family college_admissions_sat \
  --model openai/gpt-5.4 \
  --thinking-effort disabled \
  --conditions baseline,placebo,reflection,constitution \
  --orders AB \
  --min-value 1000 \
  --max-value 1500 \
  --step 50 \
  --tolerance 50 \
  --samples-per-point 3 \
  --output-prefix sat_threshold_scout_gpt54_disabled_ab
```

Example: AI labor threshold scout

```bash
../.venv/bin/python /Users/nicwong/Desktop/value-systematization/transitivity/advice_reflection_platform/scripts/run_threshold_search.py \
  --family ai_labor_displacement \
  --model openai/gpt-5.4 \
  --thinking-effort disabled \
  --conditions baseline,placebo,reflection,constitution \
  --orders AB \
  --min-value 0 \
  --max-value 100000 \
  --step 5000 \
  --tolerance 5000 \
  --samples-per-point 3 \
  --output-prefix ai_labor_threshold_scout_gpt54_disabled_ab
```

`run_threshold_search.py` outputs:

- `runs/raw/<prefix>.jsonl`
- `runs/summaries/<prefix>.csv`
- `runs/summaries/<prefix>_threshold_summary.csv`
- `runs/summaries/<prefix>_analysis.json`

Do not treat the bisection midpoint as the final estimate. It is a scout for choosing the dense ladder. The final reported curves should still come from the dense runner.


## How To Read Results Quickly

Start here:

- `runs/summaries/<prefix>_point_summary.csv`
- `runs/summaries/<prefix>_fit_summary.csv`

Interpretation:

- `event_rate`
  - observed probability of the designated event choice
- `probit_midpoint_native`
  - fitted 50/50 threshold under the monotone probit assumption
- `kernel_midpoint_native`
  - non-parametric midpoint from Gaussian kernel smoothing
- `probit_pseudo_r2`
  - fit quality relative to a flat baseline model

Rules of thumb:

- if point rates are smooth and monotone, trust probit more
- if point rates wiggle locally, trust the empirical rates first and treat probit as provisional
- if kernel and probit disagree sharply, inspect the raw rows


## Current Caveats

- The dense runner does parallelize sampled jobs, but prior-artifact generation is still one-time per non-baseline condition and happens before the fanout.
- Some older family-specific scripts still exist. Prefer the generic dense runner unless a family still needs a special wrapper.
- Different families use different axis transforms. Check `transform_name` in `FamilySpec` before interpreting thresholds.
- Baseline and scaffolded conditions may use different visible-turn structures depending on the family. Read the prompt module first if you are comparing across families.


## Fastest Way To Onboard Another Codex Session

Tell it to read these files first:

- `/Users/nicwong/Desktop/value-systematization/transitivity/advice_reflection_platform/CLI_RUNNERS_README.md`
- `/Users/nicwong/Desktop/value-systematization/transitivity/advice_reflection_platform/experiment_families.py`
- `/Users/nicwong/Desktop/value-systematization/transitivity/advice_reflection_platform/experiment_runner.py`
- `/Users/nicwong/Desktop/value-systematization/transitivity/advice_reflection_platform/experiment_results.py`
- `/Users/nicwong/Desktop/value-systematization/transitivity/advice_reflection_platform/prompts/template_family_prompts.py`
- `/Users/nicwong/Desktop/value-systematization/transitivity/advice_reflection_platform/prompts/ai_labor_prompts.py`
- `/Users/nicwong/Desktop/value-systematization/transitivity/advice_reflection_platform/prompts/defense_casualty_prompts.py`
- `/Users/nicwong/Desktop/value-systematization/transitivity/advice_reflection_platform/prompts/affair_disclosure_prompts.py`
- `/Users/nicwong/Desktop/value-systematization/transitivity/advice_reflection_platform/tests/test_sampled_tradeoff_grid.py`

That is enough context for:

- running new pilots
- reading artifacts
- adding another prompt family

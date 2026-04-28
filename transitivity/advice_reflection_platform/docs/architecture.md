# Architecture

## Goal

This MVP implements the two-layer compromise from the prior handoff:

- `Structured benchmark mode`: realistic advice requests with explicit `A/B` advisory stances for measurable reflection effects
- `Exploration mode`: the same registry and parser stack exposed through a lightweight UI for inspection and iteration

## Core Components

### `experiment_runner.py`

Primary home for the dense-curve experiment engine.

It owns:

- prior-artifact generation
- sampled-query execution
- prompt attachment and follow-up choice flow

`backend/sampled_tradeoff_grid.py` is now only a compatibility shim for older imports.

### `experiment_families.py`

Dense-curve family registry and scenario construction.

It owns:

- `FamilySpec` and `LadderPoint`
- family registry and ladder definitions
- generic family-level placebo/reflection/constitution prompts
- scenario builders for sampled-grid runs

For custom two-stage families, `FamilySpec` also carries prompt hooks:

- `turn1_prompt_builder`
- `direct_choice_prompt_builder`
- `followup_choice_prompt_builder`

That means adding a new custom prompt family no longer requires editing the runner.

### `experiment_results.py`

Dense-curve fitting, summary, and report helpers.

It owns:

- monotone probit and Gaussian-kernel fits
- cross-family sampled-grid summarization
- generic sampled-grid HTML report generation

### `prompts/`

Prompt-family surface text for the newer dense-curve families.

Current modules include:

- `ai_labor_prompts.py`
- `defense_casualty_prompts.py`
- `affair_disclosure_prompts.py`
- `disaster_evacuation_prompts.py`
- `hiring_selection_prompts.py`

The `backend/*_prompts.py` files now remain only as compatibility shims.

### `backend/schemas.py`

Dataclasses for:

- scenario records
- run conditions
- parsed choice metadata
- normalized run rows

The schema keeps realistic prose on the surface while preserving hidden latent metadata such as `family_id`, `latent_dimensions`, and `paraphrase_group`.

### `backend/scenario_factory.py`

Initial scenario-generation pipeline.

It expands JSON family templates into concrete scenario records. Each family template defines:

- request templates
- option templates
- latent dimensions
- paraphrase grouping

This keeps the main benchmark structured even when the user-facing prose looks natural.

### `backend/gateway.py`

Three gateway modes:

- `HeuristicDemoGateway`: deterministic offline gateway for UI/testing
- `ReplayGateway`: canned responses for tests and parser regression checks
- `LiveModelGateway`: thin wrapper around the parent repo's `src/shared_api.py`

### `backend/parser.py`

The parser explicitly addresses the reconsideration issue from the current project:

- scan for multiple JSON objects
- keep the first valid choice/reason pair
- prefer the last valid pair as the final answer
- mark `within_response_revision` when the answer changes inside one response
- fall back to regex only when JSON parsing fails

### `backend/orchestrator.py`

Supports:

- single-run baseline versus reflection comparison
- batch job loading from CSV or JSON
- normalized prompts with `AB` and `BA` order control

### `backend/analysis.py`

Simple summary layer focused on the new MVP metrics:

- scenario-level consistency
- order sensitivity
- paraphrase sensitivity
- reflection change flags
- within-response revision rate

### `backend/artifacts.py`

Hybrid artifact store:

- JSONL raw runs on disk
- CSV summaries on disk
- SQLite metadata index for quick lookup

## Data Layout

```text
data/
  families/
  scenarios/
  uploads/
runs/
  raw/
  summaries/
exports/
```

## Next Expansion Points

- add a proper backend API once the UI moves beyond Streamlit
- add richer scenario editing constraints and validation
- replace demo gateway heuristics with live model presets
- add qualitative exploration mode for free-text advice without treating it as the primary benchmark

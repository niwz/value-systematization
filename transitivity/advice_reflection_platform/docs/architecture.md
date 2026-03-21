# Architecture

## Goal

This MVP implements the two-layer compromise from the prior handoff:

- `Structured benchmark mode`: realistic advice requests with explicit `A/B` advisory stances for measurable reflection effects
- `Exploration mode`: the same registry and parser stack exposed through a lightweight UI for inspection and iteration

## Core Components

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


# Advice Reflection Platform MVP

This is a fresh scaffold for the two-layer setup from the prior handoff:

- structured benchmark mode for actual reflection claims
- exploration mode for realistic prompt inspection

The package is isolated from the existing `transitivity` experiments but reuses the same design lessons:

- explicit `A/B` choices for clean scoring
- robust parsing that prefers the last valid JSON answer
- batchable runs and simple summary metrics
- raw artifacts on disk plus lightweight SQLite metadata

## Layout

```text
advice_reflection_platform/
  app.py
  backend/
  data/
    families/
    scenarios/
    uploads/
  docs/
  exports/
  runs/
  scripts/
  tests/
```

## Run The Sample Scenario Generator

```bash
../.venv/bin/python advice_reflection_platform/scripts/generate_sample_scenarios.py
```

## Run Tests

```bash
../.venv/bin/python -m unittest discover advice_reflection_platform/tests
```

## Launch The UI

```bash
streamlit run advice_reflection_platform/app.py
```

The app works offline with a deterministic demo gateway and can also use a live model adapter if the parent repo's `src/shared_api.py` plus API credentials are available.


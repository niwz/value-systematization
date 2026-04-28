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

## Generate The Mentee Family Pilot

```bash
../.venv/bin/python advice_reflection_platform/scripts/generate_mentee_family_pilot.py
```

This writes:

- `data/scenarios/mentee_family_pilot.json`
- `data/uploads/mentee_family_pilot_jobs.json`
- `data/uploads/mentee_family_pilot_manifest.json`

The frozen pilot uses:

- one family: `mentee_job_application_honesty`
- 12 cells
- 3 paraphrases per cell
- fixed exemplar and held-out cell splits
- 3 conditions: `baseline`, `family_context_control`, `family_rule_reflection`

## Run The Family Pilot

Demo smoke run:

```bash
../.venv/bin/python advice_reflection_platform/scripts/run_family_pilot.py --gateway demo
```

Live run:

```bash
../.venv/bin/python advice_reflection_platform/scripts/run_family_pilot.py --gateway live
```

The runner writes:

- raw JSONL rows under `runs/raw/`
- flat CSV rows under `runs/summaries/`
- held-out pilot metrics in `*_analysis.json`

The pilot metrics include:

- held-out order sensitivity by condition
- held-out paraphrase sensitivity by condition
- family-fit accuracy/log loss from the two latent axes
- held-out anchor error rate
- change rate versus baseline
- a `go_signal` that only turns on when `family_rule_reflection` is at least as invariant as both comparators and strictly better on the latent-fit metrics

## Run Tests

```bash
../.venv/bin/python -m unittest discover advice_reflection_platform/tests
```

## Launch The UI

```bash
streamlit run advice_reflection_platform/app.py
```

The app works offline with a deterministic demo gateway and can also use a live model adapter if the parent repo's `src/shared_api.py` plus API credentials are available.

# Dense-Curve Methodology

## Research Object

The experiment estimates whether runtime scaffolds shift a model's revealed decision threshold in recurring decision families.

Each family defines:

- a realistic advisory situation a user might plausibly ask a model about
- two concrete actions, used for scoring
- one numeric axis that varies across cases
- an event choice whose probability is modeled as a function of that axis

The target is not whether one answer sounds better. The target is the local choice curve: how often the model chooses the event action as the numeric tradeoff changes.

## Prompting Flow

Baseline is a single-turn decision prompt. The model sees the full situation, the numeric detail for that rung, and the explicit choice request.

Non-baseline conditions use a frozen two-turn structure:

- Turn 1 gives the general situation without the numeric rung.
- The model produces a scaffold artifact, such as a restatement, reflection, or short constitution.
- That Turn 1 exchange is frozen and reattached to each sampled rung.
- Turn 2 injects only the numeric detail and asks for the explicit choice.

This estimates the effect of a fixed scaffold artifact within a family run, not a freshly generated reflection for each rung.

## Bracketing And Ladder Selection

Before a report-quality run, use a sparse target-model bracketing pilot to find a plausible transition region.

The bracketing pilot should be cheap:

- use the target model and target thinking-effort setting when feasible
- use `AB` only unless order effects are the object of the pilot
- use `r1` or similarly low repeat count
- include a wide range of rungs that can plausibly bracket the 50/50 region

Cheap models can be used for smoke tests: checking prompt realism, parsing, refusal behavior, and gross monotonicity. They should not determine the final ladder for another model, because different models can have different effective thresholds.

## Shared Rungs

For the main run, freeze one ladder per prompt family.

That ladder should be shared across:

- baseline, placebo, reflection, and constitution
- thinking-effort settings
- repeats
- presentation-order conditions, if used

The ladder does not need to be shared across different prompt families, because the numeric axes have different meanings and scales.

This rule preserves comparability within a family. If each condition receives its own ladder, midpoint differences become harder to interpret and look post-hoc.

## Censored Midpoints

A condition can saturate across the tested range.

Examples:

- all event choices at every rung means the midpoint is below the tested range
- no event choices at every rung means the midpoint is above the tested range

These are usable results, but only as censored bounds. A below-range midpoint means the threshold is lower than the minimum tested rung. It does not identify whether the threshold is just below the minimum or far below it.

Main reports should therefore distinguish:

- identified midpoints, where the transition is within range
- censored midpoints, where the transition is outside range

Do not force a numeric midpoint for censored cases. Report the bound and interpret it as saturation.

## Estimators

The primary compact summary is a monotone probit curve. It gives a midpoint estimate, slope, and fit diagnostics under an assumed smooth monotone functional form.

The robustness check is a Gaussian-kernel curve. This is non-parametric in the sense that it smooths observed event rates without imposing a probit link or a single global slope. It is useful for checking whether the probit midpoint is being driven by a misspecified functional form.

Both estimators depend on the chosen ladder. If the ladder fails to bracket the transition, the correct conclusion is censoring, not a precise threshold.

## Practical Rule

Use this sequence for a new family:

1. Write natural prompt text and inspect transcripts.
2. Run a cheap smoke test on a low-cost model.
3. Run a sparse bracketing pilot on the target model.
4. Freeze one family-level ladder.
5. Run all conditions and thinking efforts on that shared ladder.
6. Report in-range midpoints numerically and saturated conditions as censored bounds.

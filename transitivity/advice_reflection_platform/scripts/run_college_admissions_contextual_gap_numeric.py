from __future__ import annotations

import argparse
import json
from datetime import datetime
from pathlib import Path
import sys

PACKAGE_ROOT = Path(__file__).resolve().parents[2]
if str(PACKAGE_ROOT) not in sys.path:
    sys.path.insert(0, str(PACKAGE_ROOT))

from advice_reflection_platform.backend.artifacts import ArtifactStore
from advice_reflection_platform.backend.college_admissions_contextual_gap_numeric import (
    DEFAULT_GAP_STEP,
    DEFAULT_MAX_GAP,
    DEFAULT_MIN_GAP,
    DEFAULT_TOLERANCE,
    PARAPHRASE_TEMPLATES,
    PROFILE_VARIANTS,
    run_family_prior_probe,
    run_revealed_threshold_search,
    run_stated_gap_probe,
    summarize_college_admissions_numeric_principle_gap,
    summary_to_json,
)
from advice_reflection_platform.backend.gateway import LiveModelGateway


def main() -> None:
    parser = argparse.ArgumentParser(description="Run the numeric college admissions contextual-gap pilot.")
    parser.add_argument("--model", type=str, default="openai/gpt-5.4")
    parser.add_argument("--repeats", type=int, default=1)
    parser.add_argument("--surface-forms", type=str, default="p1")
    parser.add_argument("--profile-variants", type=str, default="canonical")
    parser.add_argument("--orders", type=str, default="AB,BA")
    parser.add_argument("--min-gap", type=int, default=DEFAULT_MIN_GAP)
    parser.add_argument("--max-gap", type=int, default=DEFAULT_MAX_GAP)
    parser.add_argument("--gap-step", type=int, default=DEFAULT_GAP_STEP)
    parser.add_argument("--tolerance", type=int, default=DEFAULT_TOLERANCE)
    parser.add_argument("--thinking", action="store_true")
    parser.add_argument("--thinking-budget-tokens", type=int, default=8000)
    parser.add_argument(
        "--reflection-mode",
        choices=("general", "specific", "consistency", "error_tradeoff"),
        default="general",
    )
    parser.add_argument("--include-placebo", action="store_true")
    parser.add_argument("--include-constitution", action="store_true")
    parser.add_argument("--skip-baseline", action="store_true")
    parser.add_argument("--skip-stated", action="store_true")
    parser.add_argument("--output-prefix", type=str, default="")
    args = parser.parse_args()

    base_dir = Path(__file__).resolve().parents[1]
    surface_forms = {item.strip() for item in args.surface_forms.split(",") if item.strip()}
    profile_variants = {item.strip() for item in args.profile_variants.split(",") if item.strip()}
    orders = {item.strip().upper() for item in args.orders.split(",") if item.strip()}
    invalid_surface_forms = surface_forms.difference(PARAPHRASE_TEMPLATES)
    if invalid_surface_forms:
        raise ValueError(f"Unknown surface forms: {sorted(invalid_surface_forms)}")
    invalid_profile_variants = profile_variants.difference(PROFILE_VARIANTS)
    if invalid_profile_variants:
        raise ValueError(f"Unknown profile variants: {sorted(invalid_profile_variants)}")

    gateway = LiveModelGateway()
    reflection_artifact = run_family_prior_probe(
        gateway=gateway,
        model_name=args.model,
        prior_mode=args.reflection_mode,
        min_gap=args.min_gap,
        max_gap=args.max_gap,
        thinking=args.thinking,
        thinking_budget_tokens=args.thinking_budget_tokens,
    )
    placebo_artifact = (
        run_family_prior_probe(
            gateway=gateway,
            model_name=args.model,
            prior_mode="placebo",
            min_gap=args.min_gap,
            max_gap=args.max_gap,
            thinking=args.thinking,
            thinking_budget_tokens=args.thinking_budget_tokens,
        )
        if args.include_placebo
        else None
    )
    constitution_artifact = (
        run_family_prior_probe(
            gateway=gateway,
            model_name=args.model,
            prior_mode="constitution",
            min_gap=args.min_gap,
            max_gap=args.max_gap,
            thinking=args.thinking,
            thinking_budget_tokens=args.thinking_budget_tokens,
        )
        if args.include_constitution
        else None
    )

    revealed_records = []
    threshold_runs = []
    conditions: list[tuple[str, dict[str, object] | None]] = [("reflection", reflection_artifact)]
    if placebo_artifact is not None:
        conditions.insert(0, ("placebo", placebo_artifact))
    if constitution_artifact is not None:
        conditions.append(("constitution", constitution_artifact))
    if not args.skip_baseline:
        conditions.insert(0, ("baseline", None))

    for condition_name, artifact in conditions:
        for surface_form in sorted(surface_forms):
            for profile_variant in sorted(profile_variants):
                for presentation_order in sorted(orders):
                    for repeat_idx in range(1, args.repeats + 1):
                        run_records, threshold_run = run_revealed_threshold_search(
                            model_name=args.model,
                            condition_name=condition_name,
                            surface_form=surface_form,
                            presentation_order=presentation_order,
                            repeat_idx=repeat_idx,
                            profile_variant=profile_variant,
                            gateway=gateway,
                            min_gap=args.min_gap,
                            max_gap=args.max_gap,
                            step=args.gap_step,
                            tolerance=args.tolerance,
                            prior_artifact=artifact,
                            thinking=args.thinking,
                            thinking_budget_tokens=args.thinking_budget_tokens,
                        )
                        revealed_records.extend(run_records)
                        threshold_runs.append(threshold_run)

    stated_results: list[dict[str, object]] = []
    if not args.skip_stated:
        stated_results.append(
            run_stated_gap_probe(
                gateway=gateway,
                model_name=args.model,
                condition_name="reflection",
                min_gap=args.min_gap,
                max_gap=args.max_gap,
                step=args.gap_step,
                prior_artifact=reflection_artifact,
                thinking=args.thinking,
                thinking_budget_tokens=args.thinking_budget_tokens,
            )
        )
        if placebo_artifact is not None:
            stated_results.insert(
                0 if args.skip_baseline else 1,
                run_stated_gap_probe(
                    gateway=gateway,
                    model_name=args.model,
                    condition_name="placebo",
                    min_gap=args.min_gap,
                    max_gap=args.max_gap,
                    step=args.gap_step,
                    prior_artifact=placebo_artifact,
                    thinking=args.thinking,
                    thinking_budget_tokens=args.thinking_budget_tokens,
                ),
            )
        if constitution_artifact is not None:
            insert_at = len(stated_results)
            stated_results.insert(
                insert_at,
                run_stated_gap_probe(
                    gateway=gateway,
                    model_name=args.model,
                    condition_name="constitution",
                    min_gap=args.min_gap,
                    max_gap=args.max_gap,
                    step=args.gap_step,
                    prior_artifact=constitution_artifact,
                    thinking=args.thinking,
                    thinking_budget_tokens=args.thinking_budget_tokens,
                ),
            )
        if not args.skip_baseline:
            stated_results.insert(
                0,
                run_stated_gap_probe(
                    gateway=gateway,
                    model_name=args.model,
                    condition_name="baseline",
                    min_gap=args.min_gap,
                    max_gap=args.max_gap,
                    step=args.gap_step,
                    thinking=args.thinking,
                    thinking_budget_tokens=args.thinking_budget_tokens,
                ),
            )

    summary = summarize_college_admissions_numeric_principle_gap(
        revealed_records=revealed_records,
        threshold_runs=threshold_runs,
        stated_results=stated_results,
    )

    stamp = datetime.now().strftime("%Y%m%d_%H%M%S")
    filename_prefix = args.output_prefix or f"college_admissions_contextual_gap_numeric_{stamp}"
    store = ArtifactStore(base_dir)
    raw_path, summary_csv_path = store.write_records(revealed_records, filename_prefix)
    direct_path = base_dir / "runs" / "raw" / f"{filename_prefix}_stated_policy.json"
    direct_path.write_text(json.dumps(stated_results, indent=2) + "\n", encoding="utf-8")
    analysis_path = base_dir / "runs" / "summaries" / f"{filename_prefix}_analysis.json"
    analysis_path.write_text(summary_to_json(summary), encoding="utf-8")

    print(f"raw_path={raw_path}")
    print(f"summary_csv_path={summary_csv_path}")
    print(f"direct_path={direct_path}")
    print(f"analysis_path={analysis_path}")
    print(summary_to_json(summary))


if __name__ == "__main__":
    main()

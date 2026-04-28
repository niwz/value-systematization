from __future__ import annotations

import argparse
import json
from datetime import datetime
from pathlib import Path
import sys

PACKAGE_ROOT = Path(__file__).resolve().parents[2]
if str(PACKAGE_ROOT) not in sys.path:
    sys.path.insert(0, str(PACKAGE_ROOT))

from advice_reflection_platform.backend.ai_labor_displacement_threshold import (
    DEFAULT_MAX_SAVINGS,
    DEFAULT_MIN_SAVINGS,
    DEFAULT_SAVINGS_STEP,
    DEFAULT_TOLERANCE,
    PARAPHRASE_TEMPLATES,
    ThresholdSearchFailure,
    run_family_prior_probe,
    run_revealed_threshold_search,
    summarize_ai_labor_threshold,
)
from advice_reflection_platform.backend.artifacts import ArtifactStore
from advice_reflection_platform.backend.gateway import LiveModelGateway


def main() -> None:
    parser = argparse.ArgumentParser(description="Run the AI labor-displacement threshold family.")
    parser.add_argument("--model", type=str, default="openai/gpt-5.4")
    parser.add_argument("--repeats", type=int, default=1)
    parser.add_argument("--surface-forms", type=str, default="p1")
    parser.add_argument("--orders", type=str, default="AB,BA")
    parser.add_argument("--min-savings", type=int, default=DEFAULT_MIN_SAVINGS)
    parser.add_argument("--max-savings", type=int, default=DEFAULT_MAX_SAVINGS)
    parser.add_argument("--savings-step", type=int, default=DEFAULT_SAVINGS_STEP)
    parser.add_argument("--tolerance", type=int, default=DEFAULT_TOLERANCE)
    parser.add_argument("--include-placebo", action="store_true")
    parser.add_argument("--include-constitution", action="store_true")
    parser.add_argument("--skip-baseline", action="store_true")
    parser.add_argument("--max-query-retries", type=int, default=3)
    parser.add_argument("--request-timeout-seconds", type=float, default=60.0)
    parser.add_argument("--thinking", action="store_true")
    parser.add_argument("--thinking-budget-tokens", type=int, default=8000)
    parser.add_argument("--output-prefix", type=str, default="")
    args = parser.parse_args()

    base_dir = Path(__file__).resolve().parents[1]
    stamp = datetime.now().strftime("%Y%m%d_%H%M%S")
    filename_prefix = args.output_prefix or f"ai_labor_displacement_threshold_{stamp}"
    store = ArtifactStore(base_dir)
    surface_forms = sorted({item.strip() for item in args.surface_forms.split(",") if item.strip()})
    invalid_surface_forms = set(surface_forms).difference(PARAPHRASE_TEMPLATES)
    if invalid_surface_forms:
        raise ValueError(f"Unknown surface forms: {sorted(invalid_surface_forms)}")
    orders = sorted({item.strip().upper() for item in args.orders.split(",") if item.strip()})

    gateway = LiveModelGateway()
    reflection_artifact = run_family_prior_probe(
        gateway=gateway,
        model_name=args.model,
        prior_mode="general",
        min_savings=args.min_savings,
        max_savings=args.max_savings,
        thinking=args.thinking,
        thinking_budget_tokens=args.thinking_budget_tokens,
        max_request_retries=args.max_query_retries,
        request_timeout_seconds=args.request_timeout_seconds,
    )
    placebo_artifact = (
        run_family_prior_probe(
            gateway=gateway,
            model_name=args.model,
            prior_mode="placebo",
            min_savings=args.min_savings,
            max_savings=args.max_savings,
            thinking=args.thinking,
            thinking_budget_tokens=args.thinking_budget_tokens,
            max_request_retries=args.max_query_retries,
            request_timeout_seconds=args.request_timeout_seconds,
        )
        if args.include_placebo
        else None
    )
    constitution_artifact = (
        run_family_prior_probe(
            gateway=gateway,
            model_name=args.model,
            prior_mode="constitution",
            min_savings=args.min_savings,
            max_savings=args.max_savings,
            thinking=args.thinking,
            thinking_budget_tokens=args.thinking_budget_tokens,
            max_request_retries=args.max_query_retries,
            request_timeout_seconds=args.request_timeout_seconds,
        )
        if args.include_constitution
        else None
    )

    conditions: list[tuple[str, dict[str, object] | None]] = [("reflection", reflection_artifact)]
    if placebo_artifact is not None:
        conditions.insert(0, ("placebo", placebo_artifact))
    if constitution_artifact is not None:
        conditions.append(("constitution", constitution_artifact))
    if not args.skip_baseline:
        conditions.insert(0, ("baseline", None))

    revealed_records = []
    threshold_runs = []
    for condition_name, artifact in conditions:
        for surface_form in surface_forms:
            for presentation_order in orders:
                for repeat_idx in range(1, args.repeats + 1):
                    try:
                        run_records, threshold_run = run_revealed_threshold_search(
                            model_name=args.model,
                            condition_name=condition_name,
                            surface_form=surface_form,
                            presentation_order=presentation_order,
                            repeat_idx=repeat_idx,
                            gateway=gateway,
                            min_savings=args.min_savings,
                            max_savings=args.max_savings,
                            step=args.savings_step,
                            tolerance=args.tolerance,
                            prior_artifact=artifact,
                            thinking=args.thinking,
                            thinking_budget_tokens=args.thinking_budget_tokens,
                            max_query_retries=args.max_query_retries,
                            request_timeout_seconds=args.request_timeout_seconds,
                        )
                        revealed_records.extend(run_records)
                        threshold_runs.append(threshold_run)
                    except ThresholdSearchFailure as exc:
                        revealed_records.extend(exc.partial_records)
                        checkpoint_payload = {
                            "artifacts": {
                                "placebo": placebo_artifact,
                                "reflection": reflection_artifact,
                                "constitution": constitution_artifact,
                            },
                            "threshold_runs": threshold_runs,
                            "failure": exc.failure_payload,
                        }
                        raw_path, summary_csv_path = store.write_records(
                            revealed_records,
                            f"{filename_prefix}_checkpoint",
                        )
                        aux_path = base_dir / "runs" / "raw" / f"{filename_prefix}_checkpoint_aux.json"
                        aux_path.write_text(
                            json.dumps(checkpoint_payload, indent=2) + "\n",
                            encoding="utf-8",
                        )
                        print(f"checkpoint_raw_path={raw_path}")
                        print(f"checkpoint_summary_csv_path={summary_csv_path}")
                        print(f"checkpoint_aux_path={aux_path}")
                        print(json.dumps(checkpoint_payload, indent=2, sort_keys=True))
                        raise

    summary = summarize_ai_labor_threshold(
        revealed_records=revealed_records,
        threshold_runs=threshold_runs,
    )
    summary["artifacts"] = {
        "placebo": placebo_artifact,
        "reflection": reflection_artifact,
        "constitution": constitution_artifact,
    }

    raw_path, summary_csv_path = store.write_records(revealed_records, filename_prefix)
    aux_path = base_dir / "runs" / "raw" / f"{filename_prefix}_aux.json"
    aux_path.write_text(
        json.dumps(
            {
                "artifacts": summary["artifacts"],
                "threshold_runs": threshold_runs,
            },
            indent=2,
        )
        + "\n",
        encoding="utf-8",
    )
    analysis_path = base_dir / "runs" / "summaries" / f"{filename_prefix}_analysis.json"
    analysis_path.write_text(json.dumps(summary, indent=2, sort_keys=True) + "\n", encoding="utf-8")

    print(f"raw_path={raw_path}")
    print(f"summary_csv_path={summary_csv_path}")
    print(f"aux_path={aux_path}")
    print(f"analysis_path={analysis_path}")
    print(json.dumps(summary, indent=2, sort_keys=True))


if __name__ == "__main__":
    main()

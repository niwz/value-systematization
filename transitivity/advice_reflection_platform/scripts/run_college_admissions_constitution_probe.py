from __future__ import annotations

import argparse
import json
from collections import defaultdict
from datetime import datetime
from pathlib import Path
from statistics import mean
import sys

PACKAGE_ROOT = Path(__file__).resolve().parents[2]
if str(PACKAGE_ROOT) not in sys.path:
    sys.path.insert(0, str(PACKAGE_ROOT))

from advice_reflection_platform.backend.artifacts import ArtifactStore
from advice_reflection_platform.backend.college_admissions_contextual_gap_numeric import (
    PARAPHRASE_TEMPLATES,
    PROFILE_VARIANTS,
    run_constitution_prediction_query,
    run_family_prior_probe,
    run_revealed_threshold_query,
)
from advice_reflection_platform.backend.gateway import LiveModelGateway


def _majority_choice(choices: list[str | None]) -> str | None:
    valid = [choice for choice in choices if choice in {"A", "B"}]
    if not valid:
        return None
    a_count = sum(choice == "A" for choice in valid)
    b_count = len(valid) - a_count
    if a_count == b_count:
        return None
    return "A" if a_count > b_count else "B"


def summarize_probe(
    *,
    records,
    prediction_rows,
    artifacts: dict[str, dict[str, object]],
) -> dict[str, object]:
    level_rows: list[dict[str, object]] = []
    level_groups: dict[tuple[str, str, str, str, int], list] = defaultdict(list)
    for record in records:
        gap = int(record.latent_values["sat_gap_points"])
        level_groups[
            (
                record.model_name,
                record.condition,
                record.surface_form,
                record.presentation_order,
                gap,
            )
        ].append(record)

    for (model_name, condition, surface_form, presentation_order, gap), rows in sorted(level_groups.items()):
        valid_rows = [row for row in rows if row.canonical_choice in {"A", "B"}]
        disadvantaged_admit_rate = (
            sum(row.canonical_choice == row.latent_values["disadvantaged_applicant"] for row in valid_rows) / len(valid_rows)
            if valid_rows
            else None
        )
        level_rows.append(
            {
                "model_name": model_name,
                "condition": condition,
                "surface_form": surface_form,
                "presentation_order": presentation_order,
                "sat_gap_points": gap,
                "runs": len(rows),
                "valid_runs": len(valid_rows),
                "disadvantaged_admit_rate": disadvantaged_admit_rate,
                "higher_score_admit_rate": (
                    1.0 - disadvantaged_admit_rate if disadvantaged_admit_rate is not None else None
                ),
                "majority_choice": _majority_choice([row.canonical_choice for row in valid_rows]),
            }
        )

    condition_summary: list[dict[str, object]] = []
    condition_groups: dict[tuple[str, str], list[dict[str, object]]] = defaultdict(list)
    for row in level_rows:
        condition_groups[(str(row["model_name"]), str(row["condition"]))].append(row)
    for (model_name, condition), rows in sorted(condition_groups.items()):
        disadv_rates = [float(row["disadvantaged_admit_rate"]) for row in rows if row["disadvantaged_admit_rate"] is not None]
        condition_summary.append(
            {
                "model_name": model_name,
                "condition": condition,
                "rows": len(rows),
                "mean_disadvantaged_admit_rate": mean(disadv_rates) if disadv_rates else None,
                "surface_forms": sorted({str(row["surface_form"]) for row in rows}),
                "orders": sorted({str(row["presentation_order"]) for row in rows}),
            }
        )

    constitution_records = {
        (
            record.model_name,
            record.surface_form,
            record.presentation_order,
            record.repeat_idx,
            record.latent_values["profile_variant"],
            int(record.latent_values["sat_gap_points"]),
        ): record
        for record in records
        if record.condition == "constitution"
    }
    prediction_eval_rows: list[dict[str, object]] = []
    for row in prediction_rows:
        key = (
            str(row["model_name"]),
            str(row["surface_form"]),
            str(row["presentation_order"]),
            int(row["repeat_idx"]),
            str(row["profile_variant"]),
            int(row["sat_gap_points"]),
        )
        actual = constitution_records.get(key)
        actual_choice = actual.canonical_choice if actual else None
        predicted = row["prediction_canonical_choice"]
        prediction_eval_rows.append(
            {
                **row,
                "actual_constitution_choice": actual_choice,
                "prediction_match": (predicted == actual_choice) if predicted in {"A", "B"} and actual_choice in {"A", "B"} else None,
            }
        )

    prediction_match_values = [row["prediction_match"] for row in prediction_eval_rows if row["prediction_match"] is not None]
    return {
        "family_id": "college_admissions_contextual_gap_numeric",
        "artifacts": artifacts,
        "level_rows": level_rows,
        "condition_summary": condition_summary,
        "constitution_prediction_rows": prediction_eval_rows,
        "constitution_prediction_summary": {
            "rows": len(prediction_eval_rows),
            "match_rate": (
                sum(bool(value) for value in prediction_match_values) / len(prediction_match_values)
                if prediction_match_values
                else None
            ),
        },
    }


def main() -> None:
    parser = argparse.ArgumentParser(description="Run a fixed-gap constitution probe on the numeric admissions family.")
    parser.add_argument("--model", type=str, default="openai/gpt-5.4")
    parser.add_argument("--gaps", type=str, default="350,550,750")
    parser.add_argument("--repeats", type=int, default=1)
    parser.add_argument("--surface-forms", type=str, default="p1")
    parser.add_argument("--profile-variants", type=str, default="canonical")
    parser.add_argument("--orders", type=str, default="AB,BA")
    parser.add_argument("--thinking", action="store_true")
    parser.add_argument("--thinking-budget-tokens", type=int, default=8000)
    parser.add_argument("--output-prefix", type=str, default="")
    args = parser.parse_args()

    base_dir = Path(__file__).resolve().parents[1]
    gaps = sorted({int(item.strip()) for item in args.gaps.split(",") if item.strip()})
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
        prior_mode="general",
        min_gap=min(gaps),
        max_gap=max(gaps),
        thinking=args.thinking,
        thinking_budget_tokens=args.thinking_budget_tokens,
    )
    placebo_artifact = run_family_prior_probe(
        gateway=gateway,
        model_name=args.model,
        prior_mode="placebo",
        min_gap=min(gaps),
        max_gap=max(gaps),
        thinking=args.thinking,
        thinking_budget_tokens=args.thinking_budget_tokens,
    )
    constitution_artifact = run_family_prior_probe(
        gateway=gateway,
        model_name=args.model,
        prior_mode="constitution",
        min_gap=min(gaps),
        max_gap=max(gaps),
        thinking=args.thinking,
        thinking_budget_tokens=args.thinking_budget_tokens,
    )

    condition_to_artifact = {
        "baseline": None,
        "placebo": placebo_artifact,
        "reflection": reflection_artifact,
        "constitution": constitution_artifact,
    }

    records = []
    prediction_rows = []
    for condition_name, artifact in condition_to_artifact.items():
        for surface_form in sorted(surface_forms):
            for profile_variant in sorted(profile_variants):
                for presentation_order in sorted(orders):
                    for repeat_idx in range(1, args.repeats + 1):
                        for gap in gaps:
                            prior_messages = None
                            prior_text = ""
                            if artifact:
                                prior_messages = [
                                    {"role": "user", "content": str(artifact["prompt"])},
                                    {"role": "assistant", "content": str(artifact["prior_text"])},
                                ]
                                prior_text = str(artifact["prior_text"])
                            record = run_revealed_threshold_query(
                                sat_gap_points=gap,
                                surface_form=surface_form,
                                model_name=args.model,
                                condition_name=condition_name,
                                presentation_order=presentation_order,
                                repeat_idx=repeat_idx,
                                profile_variant=profile_variant,
                                gateway=gateway,
                                prior_messages=prior_messages,
                                reflection_text=prior_text,
                                thinking=args.thinking,
                                thinking_budget_tokens=args.thinking_budget_tokens,
                            )
                            records.append(record)
                            if condition_name == "constitution":
                                prediction_rows.append(
                                    run_constitution_prediction_query(
                                        sat_gap_points=gap,
                                        surface_form=surface_form,
                                        model_name=args.model,
                                        presentation_order=presentation_order,
                                        repeat_idx=repeat_idx,
                                        profile_variant=profile_variant,
                                        gateway=gateway,
                                        constitution_artifact=constitution_artifact,
                                        thinking=args.thinking,
                                        thinking_budget_tokens=args.thinking_budget_tokens,
                                    )
                                )

    artifacts = {
        "placebo": placebo_artifact,
        "reflection": reflection_artifact,
        "constitution": constitution_artifact,
    }
    summary = summarize_probe(records=records, prediction_rows=prediction_rows, artifacts=artifacts)

    stamp = datetime.now().strftime("%Y%m%d_%H%M%S")
    filename_prefix = args.output_prefix or f"college_admissions_constitution_probe_{stamp}"
    store = ArtifactStore(base_dir)
    raw_path, summary_csv_path = store.write_records(records, filename_prefix)
    aux_path = base_dir / "runs" / "raw" / f"{filename_prefix}_aux.json"
    aux_path.write_text(
        json.dumps(
            {
                "artifacts": artifacts,
                "constitution_prediction_rows": prediction_rows,
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

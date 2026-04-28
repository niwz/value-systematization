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
from advice_reflection_platform.backend.gateway import LiveModelGateway
from advice_reflection_platform.backend.selection_contextual_gap_probe import (
    FAMILY_SPECS,
    build_prior_messages,
    run_constitution_prediction_query,
    run_family_prior_probe,
    run_selection_query,
    summarize_probe,
)


def main() -> None:
    parser = argparse.ArgumentParser(
        description="Run a fixed-gap cross-family selection probe with local and transferred constitutions."
    )
    parser.add_argument("--family", type=str, required=True, choices=sorted(FAMILY_SPECS))
    parser.add_argument("--transfer-family", type=str, default="admissions", choices=sorted(FAMILY_SPECS))
    parser.add_argument("--model", type=str, default="openai/gpt-5.4")
    parser.add_argument("--gaps", type=str, default="350,550,750")
    parser.add_argument("--repeats", type=int, default=1)
    parser.add_argument("--profile-variants", type=str, default="canonical")
    parser.add_argument("--orders", type=str, default="AB,BA")
    parser.add_argument("--thinking", action="store_true")
    parser.add_argument("--thinking-budget-tokens", type=int, default=8000)
    parser.add_argument("--output-prefix", type=str, default="")
    args = parser.parse_args()

    base_dir = Path(__file__).resolve().parents[1]
    gaps = sorted({int(item.strip()) for item in args.gaps.split(",") if item.strip()})
    profile_variants = sorted({item.strip() for item in args.profile_variants.split(",") if item.strip()})
    orders = sorted({item.strip().upper() for item in args.orders.split(",") if item.strip()})

    gateway = LiveModelGateway()
    family_kwargs = {
        "gateway": gateway,
        "model_name": args.model,
        "min_gap": min(gaps),
        "max_gap": max(gaps),
        "thinking": args.thinking,
        "thinking_budget_tokens": args.thinking_budget_tokens,
    }

    placebo_artifact = run_family_prior_probe(
        family_key=args.family,
        prior_mode="placebo",
        **family_kwargs,
    )
    reflection_artifact = run_family_prior_probe(
        family_key=args.family,
        prior_mode="general",
        **family_kwargs,
    )
    local_constitution_artifact = run_family_prior_probe(
        family_key=args.family,
        prior_mode="constitution",
        **family_kwargs,
    )
    transfer_constitution_artifact = (
        local_constitution_artifact
        if args.transfer_family == args.family
        else run_family_prior_probe(
            family_key=args.transfer_family,
            prior_mode="constitution",
            **family_kwargs,
        )
    )

    condition_to_artifact = {
        "baseline": None,
        "placebo": placebo_artifact,
        "reflection": reflection_artifact,
        "local_constitution": local_constitution_artifact,
        "transfer_constitution": transfer_constitution_artifact,
    }

    records = []
    prediction_rows = []
    for condition_name, artifact in condition_to_artifact.items():
        for profile_variant in profile_variants:
            for presentation_order in orders:
                for repeat_idx in range(1, args.repeats + 1):
                    for gap in gaps:
                        prior_messages = build_prior_messages(artifact) if artifact else None
                        prior_text = str(artifact["prior_text"]) if artifact else ""
                        record = run_selection_query(
                            family_key=args.family,
                            score_gap_points=gap,
                            model_name=args.model,
                            condition_name=condition_name,
                            presentation_order=presentation_order,
                            repeat_idx=repeat_idx,
                            profile_variant=profile_variant,
                            gateway=gateway,
                            prior_messages=prior_messages,
                            prior_text=prior_text,
                            thinking=args.thinking,
                            thinking_budget_tokens=args.thinking_budget_tokens,
                        )
                        records.append(record)
                        if condition_name == "transfer_constitution":
                            prediction_rows.append(
                                run_constitution_prediction_query(
                                    family_key=args.family,
                                    score_gap_points=gap,
                                    model_name=args.model,
                                    presentation_order=presentation_order,
                                    repeat_idx=repeat_idx,
                                    profile_variant=profile_variant,
                                    gateway=gateway,
                                    constitution_artifact=transfer_constitution_artifact,
                                    thinking=args.thinking,
                                    thinking_budget_tokens=args.thinking_budget_tokens,
                                )
                            )

    artifacts = {
        "placebo": placebo_artifact,
        "reflection": reflection_artifact,
        "local_constitution": local_constitution_artifact,
        "transfer_constitution": transfer_constitution_artifact,
    }
    summary = summarize_probe(
        family_key=args.family,
        records=records,
        prediction_rows=prediction_rows,
        artifacts=artifacts,
    )

    stamp = datetime.now().strftime("%Y%m%d_%H%M%S")
    filename_prefix = args.output_prefix or f"selection_contextual_gap_probe_{args.family}_{stamp}"
    store = ArtifactStore(base_dir)
    raw_path, summary_csv_path = store.write_records(records, filename_prefix)
    aux_path = base_dir / "runs" / "raw" / f"{filename_prefix}_aux.json"
    aux_path.write_text(
        json.dumps(
            {
                "artifacts": artifacts,
                "prediction_rows": prediction_rows,
                "target_family": args.family,
                "transfer_family": args.transfer_family,
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

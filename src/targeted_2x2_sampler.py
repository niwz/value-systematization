"""Build a targeted 2x2 ambiguity x extremity Sonnet follow-up battery."""

from __future__ import annotations

import argparse
from pathlib import Path

import pandas as pd

from .ambiguity_bias_analysis import (
    CANDIDATES_PATH,
    RESULTS_ROOT,
    fit_semantic_model,
    load_core_training_data,
    score_items,
)

PROJECT_ROOT = Path(__file__).parent.parent
DEFAULT_BASE_OUT = PROJECT_ROOT / "data" / "generated" / "ambiguity_extremity_2x2_base.csv"
DEFAULT_PAIRED_OUT = PROJECT_ROOT / "data" / "generated" / "ambiguity_extremity_2x2_paired.csv"


def _cell_priority(df: pd.DataFrame, ambiguity_level: str, extremity_level: str) -> pd.DataFrame:
    """Sort a cell so selected items sit near the intended corner."""
    ambiguity_target = 0.0 if ambiguity_level == "low" else 1.0
    extremity_target = 0.0 if extremity_level == "low" else 1.0

    ordered = df.copy()
    ordered["corner_distance"] = (
        (ordered["ambiguity_score"] - ambiguity_target) ** 2
        + (ordered["extremity_score"] - extremity_target) ** 2
    )
    return ordered.sort_values("corner_distance", ascending=True).reset_index(drop=True)


def _round_robin_templates(df: pd.DataFrame, n: int) -> pd.DataFrame:
    """Select up to n rows while spreading choices across template families."""
    groups = {
        template: group.reset_index(drop=True)
        for template, group in df.groupby("template_family", sort=True)
    }
    selected = []
    pointers = {template: 0 for template in groups}

    while len(selected) < n:
        added = False
        for template in sorted(groups):
            idx = pointers[template]
            group = groups[template]
            if idx >= len(group):
                continue
            selected.append(group.iloc[idx])
            pointers[template] += 1
            added = True
            if len(selected) >= n:
                break
        if not added:
            break

    return pd.DataFrame(selected).reset_index(drop=True)


def build_battery(
    scored_items: pd.DataFrame,
    n_per_cell: int,
    ambiguity_low_q: float,
    ambiguity_high_q: float,
    extremity_low_q: float,
    extremity_high_q: float,
) -> tuple[pd.DataFrame, dict]:
    """Select a balanced 2x2 battery from scored candidate items."""
    amb_low_thresh = float(scored_items["ambiguity_score"].quantile(ambiguity_low_q))
    amb_high_thresh = float(scored_items["ambiguity_score"].quantile(ambiguity_high_q))
    ext_low_thresh = float(scored_items["extremity_score"].quantile(extremity_low_q))
    ext_high_thresh = float(scored_items["extremity_score"].quantile(extremity_high_q))

    cell_specs = [
        ("low_ambiguity_low_extremity", "low", "low"),
        ("low_ambiguity_high_extremity", "low", "high"),
        ("high_ambiguity_low_extremity", "high", "low"),
        ("high_ambiguity_high_extremity", "high", "high"),
    ]

    selected_frames = []
    summary = {
        "amb_low_thresh": amb_low_thresh,
        "amb_high_thresh": amb_high_thresh,
        "ext_low_thresh": ext_low_thresh,
        "ext_high_thresh": ext_high_thresh,
        "cells": {},
    }

    for cell_name, amb_level, ext_level in cell_specs:
        amb_mask = (
            scored_items["ambiguity_score"] <= amb_low_thresh
            if amb_level == "low"
            else scored_items["ambiguity_score"] >= amb_high_thresh
        )
        ext_mask = (
            scored_items["extremity_score"] <= ext_low_thresh
            if ext_level == "low"
            else scored_items["extremity_score"] >= ext_high_thresh
        )
        cell_df = scored_items[amb_mask & ext_mask].copy()
        ordered = _cell_priority(cell_df, amb_level, ext_level)
        chosen = _round_robin_templates(ordered, n_per_cell)
        chosen["cell"] = cell_name
        selected_frames.append(chosen)

        summary["cells"][cell_name] = {
            "available": len(cell_df),
            "selected": len(chosen),
            "ambiguity_mean": float(chosen["ambiguity_score"].mean()) if len(chosen) else float("nan"),
            "extremity_mean": float(chosen["extremity_score"].mean()) if len(chosen) else float("nan"),
        }

    selected = pd.concat(selected_frames, ignore_index=True)
    return selected, summary


def build_paired_rows(base_items: pd.DataFrame) -> pd.DataFrame:
    """Duplicate each selected base item into AB and BA rows for order-ablation runs."""
    rows = []
    for _, row in base_items.iterrows():
        row_dict = row.to_dict()
        base_item_id = row_dict["item_id"]
        for paired_order in ["AB", "BA"]:
            paired = dict(row_dict)
            paired["base_item_id"] = base_item_id
            paired["paired_order"] = paired_order
            paired["item_id"] = f"{base_item_id}__{paired_order}"
            paired["option_order"] = paired_order
            rows.append(paired)
    return pd.DataFrame(rows)


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--n-per-cell", type=int, default=8, help="Base items per 2x2 cell")
    parser.add_argument("--ambiguity-low-q", type=float, default=0.33)
    parser.add_argument("--ambiguity-high-q", type=float, default=0.67)
    parser.add_argument("--extremity-low-q", type=float, default=0.33)
    parser.add_argument("--extremity-high-q", type=float, default=0.67)
    parser.add_argument("--base-out", type=str, default=str(DEFAULT_BASE_OUT))
    parser.add_argument("--paired-out", type=str, default=str(DEFAULT_PAIRED_OUT))
    args = parser.parse_args()

    candidates = pd.read_csv(CANDIDATES_PATH)
    core_train = load_core_training_data(RESULTS_ROOT)
    semantic_model = fit_semantic_model(core_train)
    scored_items = score_items(candidates, semantic_model)

    base_items, summary = build_battery(
        scored_items=scored_items,
        n_per_cell=args.n_per_cell,
        ambiguity_low_q=args.ambiguity_low_q,
        ambiguity_high_q=args.ambiguity_high_q,
        extremity_low_q=args.extremity_low_q,
        extremity_high_q=args.extremity_high_q,
    )
    paired_items = build_paired_rows(base_items)

    base_out = Path(args.base_out)
    paired_out = Path(args.paired_out)
    base_out.parent.mkdir(parents=True, exist_ok=True)
    paired_out.parent.mkdir(parents=True, exist_ok=True)
    base_items.to_csv(base_out, index=False)
    paired_items.to_csv(paired_out, index=False)

    print(f"Saved base items to {base_out}")
    print(f"Saved paired items to {paired_out}")
    print(
        f"Thresholds: ambiguity low<={summary['amb_low_thresh']:.3f}, "
        f"ambiguity high>={summary['amb_high_thresh']:.3f}, "
        f"extremity low<={summary['ext_low_thresh']:.3f}, "
        f"extremity high>={summary['ext_high_thresh']:.3f}"
    )
    print("\nCell summary:")
    for cell_name, cell in summary["cells"].items():
        print(
            f"  {cell_name}: selected={cell['selected']}/{cell['available']}, "
            f"mean ambiguity={cell['ambiguity_mean']:.3f}, "
            f"mean extremity={cell['extremity_mean']:.3f}"
        )

    print("\nTemplate mix by cell:")
    mix = (
        base_items.groupby(["cell", "template_family"])
        .size()
        .unstack(fill_value=0)
        .sort_index()
    )
    print(mix.to_string())


if __name__ == "__main__":
    main()

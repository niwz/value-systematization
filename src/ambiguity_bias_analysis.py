"""Test whether Sonnet's positional bias tracks ambiguity more than extremity."""

from __future__ import annotations

import argparse
from pathlib import Path

import numpy as np
import pandas as pd
from scipy.stats import fisher_exact, pointbiserialr
from sklearn.linear_model import LogisticRegression
from sklearn.pipeline import Pipeline
from sklearn.preprocessing import StandardScaler

from .features import DELTA_FEATURE_NAMES

PROJECT_ROOT = Path(__file__).parent.parent
RESULTS_ROOT = PROJECT_ROOT / "data" / "results"
CANDIDATES_PATH = PROJECT_ROOT / "data" / "generated" / "design_matrix_candidates.csv"

CORE_MODELS = [
    "haiku_v2",
    "llama70b_v2",
    "gpt4omini_v2",
    "gemma12b_v2",
    "mistral_small_v2",
]

BENEFIT_LEVEL_MAP = {10: 0.0, 100: 0.5, 1000: 1.0}
HARM_LEVEL_MAP = {0: 0.0, 1: 0.5, 10: 1.0}


def load_core_training_data(results_root: Path) -> pd.DataFrame:
    """Pool pre-reflection choices from the cleaner models."""
    frames = []
    for model_dir in CORE_MODELS:
        path = results_root / model_dir / "pre_choices.csv"
        df = pd.read_csv(path)
        valid = df[df["original_choice"].isin(["A", "B"])].copy()
        valid["y"] = (valid["original_choice"] == "A").astype(int)
        valid["source_model"] = model_dir
        frames.append(valid)
    return pd.concat(frames, ignore_index=True)


def fit_semantic_model(train_df: pd.DataFrame) -> Pipeline:
    """Fit a pooled semantic choice model from the non-Sonnet core models."""
    X = train_df[DELTA_FEATURE_NAMES].values.astype(float)
    y = train_df["y"].values.astype(int)
    model = Pipeline([
        ("scaler", StandardScaler()),
        ("lr", LogisticRegression(max_iter=5000, random_state=42)),
    ])
    model.fit(X, y)
    return model


def score_items(candidates: pd.DataFrame, semantic_model: Pipeline) -> pd.DataFrame:
    """Assign semantic ambiguity and structural extremity to candidate items."""
    scored = candidates.copy()

    X = scored[DELTA_FEATURE_NAMES].values.astype(float)
    prob_a = semantic_model.predict_proba(X)[:, 1]
    scored["semantic_prob_a"] = prob_a
    scored["semantic_margin"] = np.abs(prob_a - 0.5) * 2.0
    scored["ambiguity_score"] = 1.0 - scored["semantic_margin"]

    scored["max_benefit"] = scored[
        ["option_A_benefit_magnitude", "option_B_benefit_magnitude"]
    ].max(axis=1)
    scored["max_harm"] = scored[
        ["option_A_harm_magnitude", "option_B_harm_magnitude"]
    ].max(axis=1)
    scored["any_direct_harm"] = (
        (scored["option_A_directness_of_harm"] == 1)
        | (scored["option_B_directness_of_harm"] == 1)
    ).astype(float)

    scored["benefit_extremity"] = scored["max_benefit"].map(BENEFIT_LEVEL_MAP).astype(float)
    scored["harm_extremity"] = scored["max_harm"].map(HARM_LEVEL_MAP).astype(float)
    scored["extremity_score"] = (
        scored["benefit_extremity"] + scored["harm_extremity"] + scored["any_direct_harm"]
    ) / 3.0

    scored["benefit_harm_conflict"] = (
        (scored["delta_benefit_magnitude"] != 0)
        & (scored["delta_harm_magnitude"] != 0)
        & ((scored["delta_benefit_magnitude"] * scored["delta_harm_magnitude"]) > 0)
    )
    scored["both_have_harm"] = (
        (scored["option_A_harm_magnitude"] > 0)
        & (scored["option_B_harm_magnitude"] > 0)
    )
    return scored


def _fisher_from_split(values: pd.Series, outcome: pd.Series) -> dict:
    """Median-split a continuous predictor and compare outcome rates."""
    threshold = float(values.median())
    high = values > threshold
    if high.nunique() < 2:
        high = values >= threshold

    table = np.array([
        [
            int(((~high) & (~outcome)).sum()),
            int(((~high) & outcome).sum()),
        ],
        [
            int((high & (~outcome)).sum()),
            int((high & outcome).sum()),
        ],
    ])
    low_n = int((~high).sum())
    high_n = int(high.sum())
    low_rate = float(outcome[~high].mean()) if low_n else float("nan")
    high_rate = float(outcome[high].mean()) if high_n else float("nan")
    odds_ratio, pvalue = fisher_exact(table)
    return {
        "threshold": threshold,
        "low_n": low_n,
        "high_n": high_n,
        "low_rate": low_rate,
        "high_rate": high_rate,
        "rate_gap": high_rate - low_rate,
        "odds_ratio": odds_ratio,
        "pvalue": pvalue,
    }


def _standardized_logistic_coefficients(df: pd.DataFrame, outcome_col: str) -> tuple[float, float]:
    """Fit a simple standardized logistic model for relative effect size."""
    X = df[["ambiguity_score", "extremity_score"]].values.astype(float)
    y = df[outcome_col].astype(int).values
    pipe = Pipeline([
        ("scaler", StandardScaler()),
        ("lr", LogisticRegression(max_iter=5000, random_state=42)),
    ])
    pipe.fit(X, y)
    coef = pipe.named_steps["lr"].coef_[0]
    return float(coef[0]), float(coef[1])


def _print_split(label: str, result: dict) -> None:
    print(f"{label}:")
    print(
        f"  low={result['low_rate']:.1%} ({int(round(result['low_rate'] * result['low_n']))}/{result['low_n']}), "
        f"high={result['high_rate']:.1%} ({int(round(result['high_rate'] * result['high_n']))}/{result['high_n']})"
    )
    print(
        f"  gap={result['rate_gap']:+.1%}, Fisher p={result['pvalue']:.4g}, "
        f"odds ratio={result['odds_ratio']:.3f}, split threshold={result['threshold']:.3f}"
    )


def analyze_pre_sonnet(pre_path: Path, item_scores: pd.DataFrame) -> dict:
    """Analyze second-presented bias in the original Sonnet pre run."""
    pre = pd.read_csv(pre_path)
    merged = pre.merge(
        item_scores[
            [
                "item_id", "ambiguity_score", "extremity_score",
                "benefit_harm_conflict", "both_have_harm",
            ]
        ],
        on="item_id",
        how="left",
    )
    merged["second_presented"] = (
        ((merged["option_order"] == "AB") & (merged["original_choice"] == "B"))
        | ((merged["option_order"] == "BA") & (merged["original_choice"] == "A"))
    )

    ambiguity = _fisher_from_split(merged["ambiguity_score"], merged["second_presented"])
    extremity = _fisher_from_split(merged["extremity_score"], merged["second_presented"])
    amb_r, amb_p = pointbiserialr(
        merged["second_presented"].astype(int), merged["ambiguity_score"]
    )
    ext_r, ext_p = pointbiserialr(
        merged["second_presented"].astype(int), merged["extremity_score"]
    )
    amb_coef, ext_coef = _standardized_logistic_coefficients(merged, "second_presented")

    print("\n=== Sonnet Pre: second-presented bias ===")
    print(
        f"Overall second-presented rate: {int(merged['second_presented'].sum())}/{len(merged)} "
        f"({merged['second_presented'].mean():.1%})"
    )
    _print_split("Semantic ambiguity split", ambiguity)
    _print_split("Extremity split", extremity)
    print(
        f"Point-biserial r with ambiguity={amb_r:+.3f} (p={amb_p:.4g}), "
        f"extremity={ext_r:+.3f} (p={ext_p:.4g})"
    )
    print(
        f"Standardized logistic coefficients: ambiguity={amb_coef:+.3f}, "
        f"extremity={ext_coef:+.3f}"
    )

    print("\nStructural ambiguity probes:")
    for column in ["benefit_harm_conflict", "both_have_harm"]:
        grouped = merged.groupby(column)["second_presented"].agg(["sum", "count", "mean"])
        false_row = grouped.loc[False]
        true_row = grouped.loc[True]
        print(
            f"  {column}: False={int(false_row['sum'])}/{int(false_row['count'])} ({false_row['mean']:.1%}), "
            f"True={int(true_row['sum'])}/{int(true_row['count'])} ({true_row['mean']:.1%})"
        )

    return {
        "ambiguity_gap": ambiguity["rate_gap"],
        "extremity_gap": extremity["rate_gap"],
        "ambiguity_coef": amb_coef,
        "extremity_coef": ext_coef,
    }


def analyze_order_ablation(order_path: Path, item_scores: pd.DataFrame, scheme: str) -> dict:
    """Analyze paired-order flip rates as a function of ambiguity/extremity."""
    df = pd.read_csv(order_path)
    pairs = (
        df.pivot_table(
            index="base_item_id",
            columns="paired_order",
            values="original_choice",
            aggfunc="first",
        )
        .dropna()
        .reset_index()
    )
    pairs["flip"] = pairs["AB"] != pairs["BA"]

    merged = pairs.merge(
        item_scores[
            [
                "item_id", "ambiguity_score", "extremity_score",
                "benefit_harm_conflict", "both_have_harm",
            ]
        ],
        left_on="base_item_id",
        right_on="item_id",
        how="left",
    )

    ambiguity = _fisher_from_split(merged["ambiguity_score"], merged["flip"])
    extremity = _fisher_from_split(merged["extremity_score"], merged["flip"])
    amb_r, amb_p = pointbiserialr(merged["flip"].astype(int), merged["ambiguity_score"])
    ext_r, ext_p = pointbiserialr(merged["flip"].astype(int), merged["extremity_score"])
    amb_coef, ext_coef = _standardized_logistic_coefficients(merged, "flip")

    print(f"\n=== Sonnet Order Ablation ({scheme}) ===")
    print(f"Overall flip rate: {int(merged['flip'].sum())}/{len(merged)} ({merged['flip'].mean():.1%})")
    _print_split("Semantic ambiguity split", ambiguity)
    _print_split("Extremity split", extremity)
    print(
        f"Point-biserial r with ambiguity={amb_r:+.3f} (p={amb_p:.4g}), "
        f"extremity={ext_r:+.3f} (p={ext_p:.4g})"
    )
    print(
        f"Standardized logistic coefficients: ambiguity={amb_coef:+.3f}, "
        f"extremity={ext_coef:+.3f}"
    )

    return {
        "ambiguity_gap": ambiguity["rate_gap"],
        "extremity_gap": extremity["rate_gap"],
        "ambiguity_coef": amb_coef,
        "extremity_coef": ext_coef,
    }


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument(
        "--results-root",
        type=str,
        default=str(RESULTS_ROOT),
        help="Root results directory containing *_v2 runs",
    )
    parser.add_argument(
        "--sonnet-pre",
        type=str,
        default=str(RESULTS_ROOT / "sonnet_v2" / "pre_choices.csv"),
        help="Path to the original Sonnet pre run",
    )
    parser.add_argument(
        "--order-dir",
        type=str,
        default=str(RESULTS_ROOT / "sonnet_order_ablation"),
        help="Directory containing order_ablation_ab.csv and order_ablation_12.csv",
    )
    args = parser.parse_args()

    results_root = Path(args.results_root)
    candidates = pd.read_csv(CANDIDATES_PATH)
    core_train = load_core_training_data(results_root)
    semantic_model = fit_semantic_model(core_train)
    item_scores = score_items(candidates, semantic_model)

    print("Semantic ambiguity model trained on pooled pre-reflection choices from:")
    print("  " + ", ".join(CORE_MODELS))

    pre_summary = analyze_pre_sonnet(Path(args.sonnet_pre), item_scores)
    ab_summary = analyze_order_ablation(Path(args.order_dir) / "order_ablation_ab.csv", item_scores, "ab")
    num_summary = analyze_order_ablation(Path(args.order_dir) / "order_ablation_12.csv", item_scores, "12")

    print("\n=== Bottom line ===")
    ambiguity_wins = 0
    extremity_wins = 0
    for label, summary in [
        ("pre-second-presented", pre_summary),
        ("order-ablation-ab", ab_summary),
        ("order-ablation-12", num_summary),
    ]:
        amb_strength = (summary["ambiguity_gap"], abs(summary["ambiguity_coef"]))
        ext_strength = (summary["extremity_gap"], abs(summary["extremity_coef"]))
        winner = "ambiguity" if amb_strength > ext_strength else "extremity"
        if winner == "ambiguity":
            ambiguity_wins += 1
        else:
            extremity_wins += 1
        print(
            f"  {label}: stronger signal from {winner} "
            f"(gap {summary['ambiguity_gap']:+.1%} vs {summary['extremity_gap']:+.1%}; "
            f"|coef| {abs(summary['ambiguity_coef']):.3f} vs {abs(summary['extremity_coef']):.3f})"
        )

    if ambiguity_wins > extremity_wins:
        print(
            "\nCurrent saved runs support ambiguity-conditioned positional bias more than "
            "a pure extremity/stakes account."
        )
    elif extremity_wins > ambiguity_wins:
        print(
            "\nCurrent saved runs support extremity/stakes more than ambiguity as the driver."
        )
    else:
        print(
            "\nCurrent saved runs do not cleanly separate ambiguity and extremity."
        )


if __name__ == "__main__":
    main()

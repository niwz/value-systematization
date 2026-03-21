"""Analysis for paired-order label-ablation diagnostics."""

import argparse
from pathlib import Path

import pandas as pd
from scipy.stats import binomtest, fisher_exact

from .models import prepare_Xy, fit_logistic, fit_random_forest
from .templates import get_response_labels

PROJECT_ROOT = Path(__file__).parent.parent
DEFAULT_RESULTS_DIR = PROJECT_ROOT / "data" / "results"


def load_scheme_results(results_dir: Path, scheme: str) -> pd.DataFrame | None:
    path = results_dir / f"order_ablation_{scheme}.csv"
    if not path.exists():
        print(f"[skipped] {path} not found")
        return None
    df = pd.read_csv(path)
    if "response_label_scheme" not in df.columns:
        df["response_label_scheme"] = scheme
    return df


def summarize_scheme(df: pd.DataFrame, scheme: str) -> dict:
    presented = df["presented_choice"].astype(str)
    first_label, second_label = get_response_labels(scheme)
    second_count = int((presented == second_label).sum())
    n = len(df)
    second_rate = second_count / n
    second_p = binomtest(second_count, n, 0.5, alternative="greater").pvalue

    ab = df[df["paired_order"] == "AB"]
    ba = df[df["paired_order"] == "BA"]
    orig_a_ab = int((ab["original_choice"] == "A").sum())
    orig_a_ba = int((ba["original_choice"] == "A").sum())
    order_p = fisher_exact([
        [orig_a_ab, len(ab) - orig_a_ab],
        [orig_a_ba, len(ba) - orig_a_ba],
    ]).pvalue

    pairs = (
        df.sort_values(["base_item_id", "paired_order"])
        .pivot_table(
            index="base_item_id",
            columns="paired_order",
            values="original_choice",
            aggfunc="first",
        )
        .dropna()
    )
    flips = pairs["AB"] != pairs["BA"]
    flip_count = int(flips.sum())
    flip_rate = flip_count / len(pairs)
    # Directional test: among flipped items, did they flip A→B more than B→A (or vice versa)?
    # Under no order effect, flips should be symmetric: P(AB=A,BA=B) = P(AB=B,BA=A)
    ab_to_b = int(((pairs["AB"] == "A") & (pairs["BA"] == "B")).sum())  # chose A in AB, B in BA
    ab_to_a = int(((pairs["AB"] == "B") & (pairs["BA"] == "A")).sum())  # chose B in AB, A in BA
    flip_direction_p = binomtest(ab_to_b, ab_to_b + ab_to_a, 0.5).pvalue if (ab_to_b + ab_to_a) > 0 else 1.0

    X, y, feature_names = prepare_Xy(df)
    logistic = fit_logistic(X, y, feature_names, 5)
    rf = fit_random_forest(X, y, feature_names, 5)

    print(f"\n=== Scheme: {scheme} ===")
    print(f"Rows: {n}, base items: {len(pairs)}")
    print(f"Presented second-label rate: {second_count}/{n} ({second_rate:.1%}), p={second_p:.4g}")
    print(f"Original A rate when paired_order=AB: {orig_a_ab}/{len(ab)} ({orig_a_ab/len(ab):.1%})")
    print(f"Original A rate when paired_order=BA: {orig_a_ba}/{len(ba)} ({orig_a_ba/len(ba):.1%})")
    print(f"Order vs original-choice Fisher p: {order_p:.4g}")
    print(f"AB/BA flip rate by item: {flip_count}/{len(pairs)} ({flip_rate:.1%})")
    print(f"  Flip direction: AB→B={ab_to_b}, AB→A={ab_to_a}, sign test p={flip_direction_p:.4g}")
    print(f"Logistic accuracy: {logistic['cv_accuracy_mean']:.3f} ± {logistic['cv_accuracy_std']:.3f}")
    print(f"Random-forest accuracy: {rf['cv_accuracy_mean']:.3f}")

    return {
        "pairs": pairs,
        "flips": flips,
        "flip_count": flip_count,
        "flip_rate": flip_rate,
        "logistic_acc": logistic["cv_accuracy_mean"],
        "logistic_std": logistic["cv_accuracy_std"],
        "rf_acc": rf["cv_accuracy_mean"],
    }


def compare_schemes(results: dict[str, dict]) -> None:
    if "ab" not in results or "12" not in results:
        return

    ab_flips = results["ab"]["flips"]
    num_flips = results["12"]["flips"]
    common = ab_flips.index.intersection(num_flips.index)
    if len(common) == 0:
        return

    ab_common = ab_flips.loc[common]
    num_common = num_flips.loc[common]

    ab_only = int(((ab_common == True) & (num_common == False)).sum())
    num_only = int(((ab_common == False) & (num_common == True)).sum())
    discordant = ab_only + num_only
    compare_p = 1.0 if discordant == 0 else binomtest(ab_only, discordant, 0.5).pvalue

    print("\n=== Cross-scheme comparison ===")
    print(f"Common base items: {len(common)}")
    print(f"Flip in ab only: {ab_only}")
    print(f"Flip in 12 only: {num_only}")
    print(f"Exact paired sign test p: {compare_p:.4g}")
    print(f"Logistic accuracy delta (12 - ab): {results['12']['logistic_acc'] - results['ab']['logistic_acc']:+.3f}")


def main() -> None:
    parser = argparse.ArgumentParser(description="Analyze paired-order label-ablation runs")
    parser.add_argument(
        "--results-dir",
        type=str,
        default=str(DEFAULT_RESULTS_DIR),
        help="Directory containing order_ablation_ab.csv and order_ablation_12.csv",
    )
    args = parser.parse_args()

    results_dir = Path(args.results_dir)
    print(f"Results dir: {results_dir}")

    results = {}
    for scheme in ["ab", "12"]:
        df = load_scheme_results(results_dir, scheme)
        if df is None:
            continue
        results[scheme] = summarize_scheme(df, scheme)

    compare_schemes(results)


if __name__ == "__main__":
    main()

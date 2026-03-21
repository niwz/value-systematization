"""Analysis for the small non-moral paired-order bias battery."""

import argparse
from pathlib import Path

import pandas as pd
from scipy.stats import binomtest, fisher_exact

from .templates import get_response_labels

PROJECT_ROOT = Path(__file__).parent.parent
DEFAULT_RESULTS_DIR = PROJECT_ROOT / "data" / "results"


def load_scheme(results_dir: Path, scheme: str) -> pd.DataFrame | None:
    path = results_dir / f"nonmoral_order_ablation_{scheme}.csv"
    if not path.exists():
        print(f"[skipped] {path} not found")
        return None
    return pd.read_csv(path)


def analyze_scheme(df: pd.DataFrame, scheme: str) -> dict:
    presented = df["presented_choice"].astype(str)
    _, second_label = get_response_labels(scheme)
    second_count = int((presented == second_label).sum())
    n = len(df)

    ab = df[df["paired_order"] == "AB"]
    ba = df[df["paired_order"] == "BA"]
    orig_a_ab = int((ab["original_choice"] == "A").sum())
    orig_a_ba = int((ba["original_choice"] == "A").sum())
    order_p = fisher_exact([
        [orig_a_ab, len(ab) - orig_a_ab],
        [orig_a_ba, len(ba) - orig_a_ba],
    ]).pvalue

    pairs = (
        df.pivot_table(index="base_item_id", columns="paired_order", values="original_choice", aggfunc="first")
        .dropna()
    )
    ab_to_b = int(((pairs["AB"] == "A") & (pairs["BA"] == "B")).sum())
    ab_to_a = int(((pairs["AB"] == "B") & (pairs["BA"] == "A")).sum())
    flips = ab_to_b + ab_to_a
    direction_p = binomtest(ab_to_b, flips, 0.5).pvalue if flips else 1.0

    print(f"\n=== Scheme: {scheme} ===")
    print(f"Rows: {n}, base items: {len(pairs)}")
    print(f"Second-label rate: {second_count}/{n} ({second_count/n:.1%}), p={binomtest(second_count, n, 0.5, alternative='greater').pvalue:.4g}")
    print(f"Original A when AB: {orig_a_ab}/{len(ab)} ({orig_a_ab/len(ab):.1%})")
    print(f"Original A when BA: {orig_a_ba}/{len(ba)} ({orig_a_ba/len(ba):.1%})")
    print(f"Order vs original-choice Fisher p: {order_p:.4g}")
    print(f"Directional flips: AB:A->BA:B = {ab_to_b}, AB:B->BA:A = {ab_to_a}, sign test p={direction_p:.4g}")

    return {
        "pairs": pairs,
        "ab_to_b": ab_to_b,
        "ab_to_a": ab_to_a,
    }


def compare(results: dict[str, dict]) -> None:
    if "ab" not in results or "12" not in results:
        return
    common = results["ab"]["pairs"].index.intersection(results["12"]["pairs"].index)
    if len(common) == 0:
        return
    ab_pairs = results["ab"]["pairs"].loc[common]
    num_pairs = results["12"]["pairs"].loc[common]
    ab_flips = ab_pairs["AB"] != ab_pairs["BA"]
    num_flips = num_pairs["AB"] != num_pairs["BA"]
    ab_only = int((ab_flips & ~num_flips).sum())
    num_only = int((~ab_flips & num_flips).sum())
    disc = ab_only + num_only
    p = 1.0 if disc == 0 else binomtest(ab_only, disc, 0.5).pvalue
    print("\n=== Cross-scheme comparison ===")
    print(f"Flip in ab only: {ab_only}")
    print(f"Flip in 12 only: {num_only}")
    print(f"Exact paired sign test p: {p:.4g}")


def main() -> None:
    parser = argparse.ArgumentParser(description="Analyze non-moral paired-order bias runs")
    parser.add_argument("--results-dir", type=str, default=str(DEFAULT_RESULTS_DIR))
    args = parser.parse_args()

    results_dir = Path(args.results_dir)
    results = {}
    for scheme in ["ab", "12"]:
        df = load_scheme(results_dir, scheme)
        if df is not None:
            results[scheme] = analyze_scheme(df, scheme)
    compare(results)


if __name__ == "__main__":
    main()

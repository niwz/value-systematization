"""Analysis pipeline for pilot results."""

import argparse
from pathlib import Path

import numpy as np
import pandas as pd

from .models import prepare_Xy, fit_logistic, fit_decision_tree

RESULTS_DIR = Path(__file__).parent.parent / "data" / "results"


def load_results(filename: str) -> pd.DataFrame | None:
    path = RESULTS_DIR / filename
    if not path.exists():
        print(f"  [skipped] {filename} not found")
        return None
    return pd.read_csv(path)


def analyze_dataset(df: pd.DataFrame, label: str, cv_folds: int = 5) -> dict | None:
    """Run full analysis on a single results dataframe."""
    X, y, feature_names = prepare_Xy(df)

    if len(y) < 10:
        print(f"  [skipped] {label}: only {len(y)} valid responses")
        return None

    print(f"\n--- {label} ---")
    print(f"N={len(y)}, base rate (chose A)={y.mean():.3f}")

    # Invalid response rate
    total = len(df)
    valid = len(y)
    invalid_rate = 1 - valid / total
    print(f"Invalid response rate: {invalid_rate:.1%} ({total - valid}/{total})")

    # Logistic regression
    logistic = fit_logistic(X, y, feature_names, cv_folds)
    print(f"\nLogistic (L1):")
    print(f"  CV accuracy: {logistic['cv_accuracy_mean']:.3f} ± {logistic['cv_accuracy_std']:.3f}")
    print(f"  CV log loss: {logistic['cv_logloss_mean']:.3f} ± {logistic['cv_logloss_std']:.3f}")
    print(f"  Nonzero coefficients: {logistic['n_nonzero']}/{len(feature_names)}")
    print(f"\n  Coefficients:")
    print(logistic["coefficients"].to_string(index=False))

    # Decision tree
    tree = fit_decision_tree(X, y, feature_names, cv_folds)
    print(f"\nDecision tree (depth≤3):")
    print(f"  CV accuracy: {tree['cv_accuracy_mean']:.3f} ± {tree['cv_accuracy_std']:.3f}")
    print(f"  CV log loss: {tree['cv_logloss_mean']:.3f} ± {tree['cv_logloss_std']:.3f}")
    print(f"  Depth: {tree['tree_depth']}, Leaves: {tree['n_leaves']}")
    print(f"\n  Feature importances:")
    print(tree["feature_importances"].to_string(index=False))

    return {"logistic": logistic, "tree": tree, "n": len(y), "base_rate": y.mean()}


def compare_conditions(results: dict[str, dict]) -> None:
    """Compare metrics across conditions."""
    if len(results) < 2:
        return

    print(f"\n{'='*60}")
    print("CONDITION COMPARISON")
    print(f"{'='*60}")

    rows = []
    for label, r in results.items():
        if r is None:
            continue
        rows.append({
            "condition": label,
            "n": r["n"],
            "base_rate": f"{r['base_rate']:.3f}",
            "logistic_acc": f"{r['logistic']['cv_accuracy_mean']:.3f}",
            "logistic_ll": f"{r['logistic']['cv_logloss_mean']:.3f}",
            "tree_acc": f"{r['tree']['cv_accuracy_mean']:.3f}",
            "nonzero_coefs": r["logistic"]["n_nonzero"],
        })

    if rows:
        comparison = pd.DataFrame(rows)
        print(comparison.to_string(index=False))


def analyze_position_effects(df: pd.DataFrame, label: str) -> None:
    """Check for position effects in sequential runs."""
    if "position" not in df.columns:
        return

    valid = df[df["original_choice"].isin(["A", "B"])].copy()
    if len(valid) < 10:
        return

    print(f"\n--- Position effects: {label} ---")
    valid["chose_A"] = (valid["original_choice"] == "A").astype(int)

    # Split into early vs late
    mid = len(valid) // 2
    early = valid[valid["position"] < mid]
    late = valid[valid["position"] >= mid]

    print(f"  Early items (pos 0-{mid-1}): chose A = {early['chose_A'].mean():.3f} (n={len(early)})")
    print(f"  Late items (pos {mid}-{len(valid)-1}): chose A = {late['chose_A'].mean():.3f} (n={len(late)})")

    # Correlation of position with choice
    corr = valid["position"].corr(valid["chose_A"])
    print(f"  Position-choice correlation: {corr:.3f}")


def main():
    parser = argparse.ArgumentParser(description="Analyze pilot results")
    parser.add_argument("--cv-folds", type=int, default=5)
    args = parser.parse_args()

    print("=" * 60)
    print("PILOT ANALYSIS REPORT")
    print("=" * 60)

    all_results = {}

    # --- Sanity check ---
    sanity = load_results("sanity_run.csv")
    if sanity is not None:
        r = analyze_dataset(sanity, "Sanity Check", args.cv_folds)
        if r:
            all_results["sanity"] = r

    # --- Pre-reflection ---
    pre = load_results("pre_choices.csv")
    if pre is not None:
        r = analyze_dataset(pre, "Pre-reflection", args.cv_folds)
        if r:
            all_results["pre"] = r

    # --- Post-independent ---
    for cond in ["no_reflection", "domain_reflection"]:
        df = load_results(f"post_independent_{cond}.csv")
        if df is not None:
            label = f"Post-independent ({cond})"
            r = analyze_dataset(df, label, args.cv_folds)
            if r:
                all_results[f"post_ind_{cond}"] = r

    # --- Post-sequential ---
    for cond in ["no_reflection", "domain_reflection"]:
        df = load_results(f"post_sequential_{cond}.csv")
        if df is not None:
            label = f"Post-sequential ({cond})"
            r = analyze_dataset(df, label, args.cv_folds)
            if r:
                all_results[f"post_seq_{cond}"] = r
            analyze_position_effects(df, label)

    # --- Comparisons ---
    compare_conditions(all_results)

    # --- Key pilot questions ---
    print(f"\n{'='*60}")
    print("PILOT SUCCESS CRITERIA")
    print(f"{'='*60}")

    if "sanity" in all_results:
        print(f"\nA. Template validity:")
        print(f"   Invalid response rate is reported above.")

    if "pre" in all_results:
        acc = all_results["pre"]["logistic"]["cv_accuracy_mean"]
        base = all_results["pre"]["base_rate"]
        chance = max(base, 1 - base)
        print(f"\nB. Feature signal:")
        print(f"   Pre-reflection logistic accuracy: {acc:.3f}")
        print(f"   Majority-class baseline: {chance:.3f}")
        print(f"   Signal above chance: {acc - chance:+.3f}")

    ind_keys = [k for k in all_results if k.startswith("post_ind")]
    seq_keys = [k for k in all_results if k.startswith("post_seq")]
    if ind_keys and seq_keys:
        ind_acc = np.mean([all_results[k]["logistic"]["cv_accuracy_mean"] for k in ind_keys])
        seq_acc = np.mean([all_results[k]["logistic"]["cv_accuracy_mean"] for k in seq_keys])
        print(f"\nC. Sequential confound check:")
        print(f"   Post-independent avg accuracy: {ind_acc:.3f}")
        print(f"   Post-sequential avg accuracy: {seq_acc:.3f}")
        print(f"   Gap: {seq_acc - ind_acc:+.3f}")

    refl_keys = {k: v for k, v in all_results.items() if "domain_reflection" in k}
    norefl_keys = {k: v for k, v in all_results.items() if "no_reflection" in k and "post" in k}
    if refl_keys and norefl_keys:
        refl_acc = np.mean([v["logistic"]["cv_accuracy_mean"] for v in refl_keys.values()])
        norefl_acc = np.mean([v["logistic"]["cv_accuracy_mean"] for v in norefl_keys.values()])
        print(f"\nD. Reflection signal:")
        print(f"   No-reflection avg accuracy: {norefl_acc:.3f}")
        print(f"   Domain-reflection avg accuracy: {refl_acc:.3f}")
        print(f"   Gap: {refl_acc - norefl_acc:+.3f}")


if __name__ == "__main__":
    main()

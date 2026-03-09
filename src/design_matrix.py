"""Generate and diagnose the design matrix of candidate dilemma feature rows."""

import itertools
import numpy as np
import pandas as pd
import yaml
from pathlib import Path

from .features import MORAL_FEATURES, DilemmaItem, DELTA_FEATURE_NAMES

CONFIG_PATH = Path(__file__).parent.parent / "configs" / "pilot.yaml"


def _load_config() -> dict:
    with open(CONFIG_PATH) as f:
        return yaml.safe_load(f)


def _is_valid_combination(a: dict, b: dict) -> bool:
    """Filter out degenerate or impossible feature combos."""
    # Both options identical is uninformative
    if all(a[k] == b[k] for k in a):
        return False
    # Option with 0 benefit probability but nonzero harm is dominated / trivially bad
    if a["benefit_probability"] == 0.0 and a["harm_magnitude"] > 0:
        return False
    if b["benefit_probability"] == 0.0 and b["harm_magnitude"] > 0:
        return False
    return True


def generate_candidate_rows(
    n: int | None = None, seed: int | None = None
) -> pd.DataFrame:
    """Generate n candidate dilemma feature rows by sampling option pairs.

    Reads defaults from pilot.yaml config if not provided.
    """
    config = _load_config()
    if n is None:
        n = config["design_matrix"]["n_candidates"]
    if seed is None:
        seed = config["design_matrix"]["seed"]

    rng = np.random.default_rng(seed)

    feature_names = list(MORAL_FEATURES.keys())
    feature_values = list(MORAL_FEATURES.values())

    # All possible single-option feature combos
    all_combos = list(itertools.product(*feature_values))

    rows = []
    attempts = 0
    max_attempts = n * 20

    while len(rows) < n and attempts < max_attempts:
        attempts += 1
        idx_a = rng.integers(0, len(all_combos))
        idx_b = rng.integers(0, len(all_combos))

        a = dict(zip(feature_names, all_combos[idx_a]))
        b = dict(zip(feature_names, all_combos[idx_b]))

        if not _is_valid_combination(a, b):
            continue

        template_families = config["template_families"]
        family = template_families[len(rows) % len(template_families)]
        order = "AB" if rng.random() < 0.5 else "BA"

        item = DilemmaItem(
            item_id=f"cand_{len(rows):04d}",
            template_family=family,
            paraphrase_group=f"pg_{len(rows):04d}",
            option_A_benefit_magnitude=a["benefit_magnitude"],
            option_A_harm_magnitude=a["harm_magnitude"],
            option_A_benefit_probability=a["benefit_probability"],
            option_A_directness_of_harm=a["directness_of_harm"],
            option_A_beneficiary_identified=a["beneficiary_identified"],
            option_B_benefit_magnitude=b["benefit_magnitude"],
            option_B_harm_magnitude=b["harm_magnitude"],
            option_B_benefit_probability=b["benefit_probability"],
            option_B_directness_of_harm=b["directness_of_harm"],
            option_B_beneficiary_identified=b["beneficiary_identified"],
            option_order=order,
        )
        rows.append(item.to_dict())

    df = pd.DataFrame(rows)
    return df


def compute_diagnostics(df: pd.DataFrame) -> dict:
    """Compute design matrix diagnostics."""
    delta_cols = [c for c in df.columns if c.startswith("delta_")]
    X = df[delta_cols].values.astype(float)

    # Marginal distributions
    marginals = {}
    for col in delta_cols:
        marginals[col] = df[col].value_counts().to_dict()

    # Pairwise correlations
    corr = df[delta_cols].corr()

    # Condition number
    try:
        cond = np.linalg.cond(X.T @ X)
    except np.linalg.LinAlgError:
        cond = float("inf")

    # Counts per feature value (option-level)
    option_counts = {}
    for feat in MORAL_FEATURES:
        for prefix in ["option_A_", "option_B_"]:
            col = prefix + feat
            if col in df.columns:
                option_counts[col] = df[col].value_counts().to_dict()

    return {
        "n_rows": len(df),
        "marginals": marginals,
        "correlation_matrix": corr,
        "condition_number": cond,
        "option_level_counts": option_counts,
    }


def print_diagnostics(diag: dict) -> None:
    """Print diagnostics to stdout."""
    print(f"Rows: {diag['n_rows']}")
    print(f"\nCondition number: {diag['condition_number']:.1f}")

    print("\n--- Marginal distributions (delta features) ---")
    for feat, counts in diag["marginals"].items():
        print(f"\n{feat}:")
        for val, count in sorted(counts.items()):
            print(f"  {val}: {count}")

    print("\n--- Correlation matrix ---")
    print(diag["correlation_matrix"].round(3).to_string())


def main():
    out_dir = Path(__file__).parent.parent / "data" / "generated"
    out_dir.mkdir(parents=True, exist_ok=True)

    df = generate_candidate_rows()
    df.to_csv(out_dir / "design_matrix_candidates.csv", index=False)

    diag = compute_diagnostics(df)
    print_diagnostics(diag)

    print(f"\nSaved {len(df)} rows to {out_dir / 'design_matrix_candidates.csv'}")


if __name__ == "__main__":
    main()

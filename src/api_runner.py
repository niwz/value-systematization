"""Consolidated API runner for all experimental conditions."""

import argparse
import re
import time
from pathlib import Path

import numpy as np
import pandas as pd
import yaml

from .nonmoral_bias import build_nonmoral_order_ablation_items
from .shared_api import call_text_response, create_client
from .templates import LabelScheme, get_response_labels, render_dilemma

PROJECT_ROOT = Path(__file__).parent.parent
DEFAULT_CONFIG_PATH = PROJECT_ROOT / "configs" / "pilot.yaml"
PROMPTS_DIR = PROJECT_ROOT / "prompts"
RESULTS_DIR = PROJECT_ROOT / "data" / "results"


def load_config(config_path: str | Path | None = None) -> dict:
    path = Path(config_path) if config_path else DEFAULT_CONFIG_PATH
    with open(path) as f:
        return yaml.safe_load(f)


def load_prompt(name: str) -> str:
    with open(PROMPTS_DIR / name) as f:
        return f.read().strip()


def adapt_prompt_labels(text: str, label_scheme: LabelScheme = "ab") -> str:
    """Rewrite hardcoded A/B response instructions for the selected scheme."""
    first_label, second_label = get_response_labels(label_scheme)
    return text.replace("Reply with only A or B.", f"Reply with only {first_label} or {second_label}.")


def parse_choice(text: str, label_scheme: LabelScheme = "ab") -> str | None:
    """Parse a response label from model output. Tries last exact occurrence."""
    text = text.strip()
    # Strip thinking blocks (e.g. DeepSeek R1 <think>...</think>)
    cleaned = re.sub(r"<think>.*?</think>", "", text, flags=re.DOTALL).strip()
    if not cleaned:
        cleaned = text

    labels = get_response_labels(label_scheme)
    upper = cleaned.upper()
    normalized_labels = {label.upper(): label for label in labels}
    if upper in normalized_labels:
        return normalized_labels[upper]

    token_pattern = r"\b(" + "|".join(re.escape(label.upper()) for label in labels) + r")\b"
    matches = re.findall(token_pattern, upper)
    if matches:
        return normalized_labels[matches[-1]]
    return None


def response_label_to_original_choice(
    parsed_choice: str | None,
    option_order: str,
    label_scheme: LabelScheme = "ab",
) -> str | None:
    """Map a presented response label back to original option identity A/B."""
    if parsed_choice is None:
        return None

    first_label, second_label = get_response_labels(label_scheme)
    if option_order == "AB":
        mapping = {first_label: "A", second_label: "B"}
    else:
        mapping = {first_label: "B", second_label: "A"}
    return mapping.get(parsed_choice)


def original_choice_to_response_label(
    original_choice: str | None,
    option_order: str,
    label_scheme: LabelScheme = "ab",
) -> str | None:
    """Map original option identity A/B to the presented response label."""
    if original_choice not in ("A", "B"):
        return None

    first_label, second_label = get_response_labels(label_scheme)
    if option_order == "AB":
        mapping = {"A": first_label, "B": second_label}
    else:
        mapping = {"A": second_label, "B": first_label}
    return mapping[original_choice]


def call_model(
    client,
    provider: str,
    config: dict,
    system_prompt: str,
    user_message: str,
    label_scheme: LabelScheme = "ab",
    prior_messages: list[dict] | None = None,
) -> dict:
    """Dispatch to the appropriate API call based on provider."""
    response = call_text_response(
        client=client,
        provider=provider,
        model_name=config["model"]["name"],
        max_tokens=config["model"]["max_tokens"],
        temperature=config["model"]["temperature"],
        system_prompt=system_prompt,
        user_message=user_message,
        prior_messages=prior_messages,
    )
    response["parsed_choice"] = parse_choice(response["raw_response"], label_scheme=label_scheme)
    return response


def run_items(
    client,
    provider: str,
    config: dict,
    items: pd.DataFrame,
    system_prompt: str,
    mode: str,
    condition: str,
    label_scheme: LabelScheme = "ab",
    sequential: bool = False,
) -> list[dict]:
    """Run a batch of items. If sequential, accumulate conversation history."""
    results = []
    prior_messages = [] if sequential else None
    delay = 2.0 if provider == "openrouter" else 0.5

    for position, (_, row) in enumerate(items.iterrows()):
        row_dict = row.to_dict()
        prompt_text = render_dilemma(row_dict, label_scheme=label_scheme)

        result = call_model(
            client, provider, config, system_prompt, prompt_text,
            label_scheme=label_scheme, prior_messages=prior_messages,
        )

        # Retry once on malformed output
        if result["parsed_choice"] is None:
            first_label, second_label = get_response_labels(label_scheme)
            retry_msg = prompt_text + f"\n\nPlease reply with only {first_label} or {second_label}."
            result = call_model(
                client, provider, config, system_prompt, retry_msg,
                label_scheme=label_scheme, prior_messages=prior_messages,
            )

        original_choice = response_label_to_original_choice(
            result["parsed_choice"],
            row_dict.get("option_order", "AB"),
            label_scheme=label_scheme,
        )

        record = {
            "item_id": row_dict["item_id"],
            "template_family": row_dict["template_family"],
            "mode": mode,
            "condition": condition,
            "evaluation_mode": "sequential" if sequential else "independent",
            "position": position,
            "raw_response": result["raw_response"],
            "presented_choice": result["parsed_choice"],
            "original_choice": original_choice,
            "option_order": row_dict.get("option_order", "AB"),
            "response_label_scheme": label_scheme,
            "model": result["model"],
            "input_tokens": result["input_tokens"],
            "output_tokens": result["output_tokens"],
            "timestamp": result["timestamp"],
        }
        for meta_key in [
            "base_item_id", "paired_order", "paraphrase_group",
            "domain", "scenario_text", "option_A_text", "option_B_text",
        ]:
            if meta_key in row_dict:
                record[meta_key] = row_dict[meta_key]
        # Include delta features
        for col in row_dict:
            if col.startswith("delta_"):
                record[col] = row_dict[col]

        results.append(record)

        if sequential and result["parsed_choice"]:
            prior_messages.append({"role": "user", "content": prompt_text})
            prior_messages.append(
                {"role": "assistant", "content": result["raw_response"]}
            )

        # Brief pause to avoid rate limits
        time.sleep(delay)

    return results


def save_results(results: list[dict], filename: str, results_dir: Path | None = None) -> Path:
    target_dir = results_dir or RESULTS_DIR
    target_dir.mkdir(parents=True, exist_ok=True)
    path = target_dir / filename
    df = pd.DataFrame(results)
    df.to_csv(path, index=False)
    print(f"Saved {len(results)} results to {path}")
    return path


def select_items(
    df: pd.DataFrame, n: int, seed: int = 42
) -> pd.DataFrame:
    """Select n items from candidates, shuffled."""
    rng = np.random.default_rng(seed)
    indices = rng.choice(len(df), size=min(n, len(df)), replace=False)
    return df.iloc[indices].reset_index(drop=True)


def select_order_ablation_items(
    df: pd.DataFrame, n: int, seed: int = 42
) -> pd.DataFrame:
    """Select n base items and duplicate each with AB and BA presentation order."""
    base_items = select_items(df, n, seed)
    paired_rows = []
    for _, row in base_items.iterrows():
        row_dict = row.to_dict()
        base_item_id = row_dict["item_id"]
        for paired_order in ["AB", "BA"]:
            paired = dict(row_dict)
            paired["base_item_id"] = base_item_id
            paired["paired_order"] = paired_order
            paired["item_id"] = f"{base_item_id}__{paired_order}"
            paired["option_order"] = paired_order
            paired_rows.append(paired)
    return pd.DataFrame(paired_rows)


def select_diverse_prior_choices(
    pre_results: pd.DataFrame, n: int = 10, seed: int = 42
) -> pd.DataFrame:
    """Select n diverse prior choices spread across template families and outcomes."""
    rng = np.random.default_rng(seed)
    valid = pre_results[pre_results["original_choice"].isin(["A", "B"])].copy()

    # Stratify by template_family x original_choice
    groups = valid.groupby(["template_family", "original_choice"])
    selected_indices = []

    # Round-robin across groups
    group_indices = {k: list(v.index) for k, v in groups}
    for k in group_indices:
        rng.shuffle(group_indices[k])

    while len(selected_indices) < n:
        added = False
        for k in list(group_indices.keys()):
            if len(selected_indices) >= n:
                break
            if group_indices[k]:
                selected_indices.append(group_indices[k].pop(0))
                added = True
        if not added:
            break

    return valid.loc[selected_indices].reset_index(drop=True)


def format_prior_choices_prompt(
    prior_items: pd.DataFrame,
    candidates: pd.DataFrame,
    label_scheme: LabelScheme = "ab",
) -> str:
    """Format prior choices into the reflection prompt."""
    template = load_prompt("reflection_prior_choice.txt")

    lines = []
    for i, (_, row) in enumerate(prior_items.iterrows(), 1):
        # Find the candidate row to render the dilemma text
        item_id = row["item_id"]
        cand_match = candidates[candidates["item_id"] == item_id]
        if len(cand_match) > 0:
            cand_row = cand_match.iloc[0].to_dict()
            dilemma_text = render_dilemma(cand_row, label_scheme=label_scheme)
            # Truncate to just the scenario + options (skip the "Choose exactly one" instruction)
            dilemma_summary = dilemma_text.split("\nChoose exactly one")[0].strip()
        else:
            dilemma_summary = f"[Dilemma {item_id}]"

        choice = original_choice_to_response_label(
            row["original_choice"],
            row.get("option_order", "AB"),
            label_scheme=label_scheme,
        )
        lines.append(f"[Dilemma {i}]\n{dilemma_summary}\n-> You chose {choice}\n")

    prior_text = "\n".join(lines)
    return template.replace("{prior_choices}", prior_text)


def main():
    parser = argparse.ArgumentParser(description="Run pilot experiment")
    parser.add_argument(
        "mode",
        choices=["sanity", "pre", "post_independent", "post_sequential", "order_ablation", "nonmoral_order_ablation", "all"],
        help="Which run mode to execute",
    )
    parser.add_argument("--seed", type=int, default=42)
    parser.add_argument(
        "--config", type=str, default=None,
        help="Path to config YAML (default: configs/pilot.yaml)",
    )
    parser.add_argument(
        "--model", type=str, default=None,
        help="Override model name from config",
    )
    parser.add_argument(
        "--response-label-scheme",
        choices=["ab", "12"],
        default="ab",
        help="Presented response labels to use in prompts (default: ab for exact reproducibility)",
    )
    parser.add_argument(
        "--results-dir",
        type=str,
        default=None,
        help="Directory to write result CSVs into (default: data/results)",
    )
    parser.add_argument(
        "--items-csv",
        type=str,
        default=None,
        help="Optional CSV of preselected items to run instead of sampling from candidates",
    )
    args = parser.parse_args()

    config = load_config(args.config)
    if args.model:
        config["model"]["name"] = args.model
    client, provider = create_client(config)
    print(f"Provider: {provider}, Model: {config['model']['name']}")

    base_system_text = load_prompt("system.txt")
    system_text = adapt_prompt_labels(
        base_system_text,
        label_scheme=args.response_label_scheme,
    )
    reflection_text = load_prompt("reflection_domain.txt")
    results_dir = Path(args.results_dir) if args.results_dir else RESULTS_DIR

    candidates = pd.read_csv(
        PROJECT_ROOT / "data" / "generated" / "design_matrix_candidates.csv"
    )
    custom_items = pd.read_csv(args.items_csv) if args.items_csv else None

    modes = [args.mode] if args.mode != "all" else [
        "sanity", "pre", "post_independent", "post_sequential"
    ]

    for mode in modes:
        print(f"\n{'='*50}")
        print(f"Running mode: {mode}")
        print(f"{'='*50}")

        if mode == "sanity":
            n = config["runs"].get("sanity", {}).get("n_items", 25)
            items = custom_items.copy() if custom_items is not None else select_items(candidates, n, args.seed)
            results = run_items(
                client, provider, config, items, system_text,
                mode="sanity", condition="no_reflection",
                label_scheme=args.response_label_scheme,
            )
            save_results(results, "sanity_run.csv", results_dir=results_dir)

        elif mode == "pre":
            items = (
                custom_items.copy()
                if custom_items is not None
                else select_items(candidates, config["runs"]["pre"]["n_items"], args.seed + 1)
            )
            results = run_items(
                client, provider, config, items, system_text,
                mode="pre", condition="no_reflection",
                label_scheme=args.response_label_scheme,
            )
            save_results(results, "pre_choices.csv", results_dir=results_dir)

        elif mode == "post_independent":
            n = config["runs"]["post_independent"]["n_items"]
            items = custom_items.copy() if custom_items is not None else select_items(candidates, n, args.seed + 2)

            for cond in config["runs"]["post_independent"]["conditions"]:
                sys_prompt = system_text
                if cond == "domain_reflection":
                    sys_prompt = reflection_text + "\n\n" + system_text
                elif cond == "prior_choice_reflection":
                    # Load pre-reflection results and build prior-choice prompt
                    pre_path = results_dir / "pre_choices.csv"
                    if not pre_path.exists():
                        print(f"  [skipped] prior_choice_reflection: pre_choices.csv not found. Run 'pre' first.")
                        continue
                    pre_results = pd.read_csv(pre_path)
                    prior_items = select_diverse_prior_choices(pre_results, n=10, seed=args.seed)
                    prior_prompt = format_prior_choices_prompt(
                        prior_items, candidates, label_scheme=args.response_label_scheme
                    )
                    sys_prompt = prior_prompt + "\n\n" + system_text

                results = run_items(
                    client, provider, config, items, sys_prompt,
                    mode="post_independent", condition=cond,
                    label_scheme=args.response_label_scheme,
                )
                save_results(results, f"post_independent_{cond}.csv", results_dir=results_dir)

        elif mode == "post_sequential":
            n = config["runs"]["post_sequential"]["n_items"]
            items = custom_items.copy() if custom_items is not None else select_items(candidates, n, args.seed + 3)

            for cond in config["runs"]["post_sequential"]["conditions"]:
                sys_prompt = system_text
                if cond == "domain_reflection":
                    sys_prompt = reflection_text + "\n\n" + system_text

                results = run_items(
                    client, provider, config, items, sys_prompt,
                    mode="post_sequential", condition=cond,
                    label_scheme=args.response_label_scheme,
                    sequential=True,
                )
                save_results(results, f"post_sequential_{cond}.csv", results_dir=results_dir)

        elif mode == "order_ablation":
            order_cfg = config["runs"].get("order_ablation", {})
            n = order_cfg.get("n_items", 30)
            schemes = [str(s) for s in order_cfg.get("label_schemes", ["ab", "12"])]
            if custom_items is not None:
                items = custom_items.copy()
            else:
                items = select_order_ablation_items(candidates, n, args.seed + 4)

            for scheme in schemes:
                scheme_system_text = adapt_prompt_labels(base_system_text, label_scheme=scheme)
                results = run_items(
                    client, provider, config, items, scheme_system_text,
                    mode="order_ablation", condition="no_reflection",
                    label_scheme=scheme,
                )
                save_results(results, f"order_ablation_{scheme}.csv", results_dir=results_dir)

        elif mode == "nonmoral_order_ablation":
            nonmoral_cfg = config["runs"].get("nonmoral_order_ablation", {})
            n = nonmoral_cfg.get("n_items", 10)
            schemes = [str(s) for s in nonmoral_cfg.get("label_schemes", ["ab", "12"])]
            items = build_nonmoral_order_ablation_items(n=n)

            for scheme in schemes:
                scheme_system_text = adapt_prompt_labels(base_system_text, label_scheme=scheme)
                results = run_items(
                    client, provider, config, items, scheme_system_text,
                    mode="nonmoral_order_ablation", condition="no_reflection",
                    label_scheme=scheme,
                )
                save_results(results, f"nonmoral_order_ablation_{scheme}.csv", results_dir=results_dir)

        # Summary
        if mode in ("sanity",):
            df = pd.DataFrame(results)
            valid = df["presented_choice"].notna().sum()
            print(f"\nValid responses: {valid}/{len(df)}")
            if valid < len(df):
                invalid = df[df["presented_choice"].isna()]
                print(f"Invalid responses:")
                for _, r in invalid.iterrows():
                    print(f"  {r['item_id']}: '{r['raw_response']}'")


if __name__ == "__main__":
    main()

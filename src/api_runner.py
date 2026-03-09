"""Consolidated API runner for all experimental conditions."""

import argparse
import csv
import json
import os
import time
from datetime import datetime, timezone
from pathlib import Path

import anthropic
import numpy as np
import pandas as pd
import yaml

from .templates import render_dilemma

PROJECT_ROOT = Path(__file__).parent.parent
CONFIG_PATH = PROJECT_ROOT / "configs" / "pilot.yaml"
PROMPTS_DIR = PROJECT_ROOT / "prompts"
RESULTS_DIR = PROJECT_ROOT / "data" / "results"


def load_config() -> dict:
    with open(CONFIG_PATH) as f:
        return yaml.safe_load(f)


def load_prompt(name: str) -> str:
    with open(PROMPTS_DIR / name) as f:
        return f.read().strip()


def parse_choice(text: str) -> str | None:
    """Parse A or B from response text."""
    text = text.strip().upper()
    if text in ("A", "B"):
        return text
    # Try first non-whitespace character
    for ch in text:
        if ch in ("A", "B"):
            return ch
    return None


def call_claude(
    client: anthropic.Anthropic,
    config: dict,
    system_prompt: str,
    user_message: str,
    prior_messages: list[dict] | None = None,
) -> dict:
    """Make a single API call and return result dict."""
    messages = []
    if prior_messages:
        messages.extend(prior_messages)
    messages.append({"role": "user", "content": user_message})

    response = client.messages.create(
        model=config["model"]["name"],
        max_tokens=config["model"]["max_tokens"],
        temperature=config["model"]["temperature"],
        system=system_prompt,
        messages=messages,
    )

    raw_text = response.content[0].text
    parsed = parse_choice(raw_text)

    return {
        "raw_response": raw_text,
        "parsed_choice": parsed,
        "model": response.model,
        "input_tokens": response.usage.input_tokens,
        "output_tokens": response.usage.output_tokens,
        "timestamp": datetime.now(timezone.utc).isoformat(),
    }


def run_items(
    client: anthropic.Anthropic,
    config: dict,
    items: pd.DataFrame,
    system_prompt: str,
    mode: str,
    condition: str,
    sequential: bool = False,
) -> list[dict]:
    """Run a batch of items. If sequential, accumulate conversation history."""
    results = []
    prior_messages = [] if sequential else None

    for position, (_, row) in enumerate(items.iterrows()):
        row_dict = row.to_dict()
        prompt_text = render_dilemma(row_dict)

        result = call_claude(
            client, config, system_prompt, prompt_text, prior_messages
        )

        # Retry once on malformed output
        if result["parsed_choice"] is None:
            retry_msg = prompt_text + "\n\nPlease reply with only A or B."
            result = call_claude(
                client, config, system_prompt, retry_msg, prior_messages
            )

        # Map choice back to original option if order was swapped
        original_choice = result["parsed_choice"]
        if original_choice and row_dict.get("option_order") == "BA":
            original_choice = "B" if original_choice == "A" else "A"

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
            "model": result["model"],
            "input_tokens": result["input_tokens"],
            "output_tokens": result["output_tokens"],
            "timestamp": result["timestamp"],
        }
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
        time.sleep(0.5)

    return results


def save_results(results: list[dict], filename: str) -> Path:
    RESULTS_DIR.mkdir(parents=True, exist_ok=True)
    path = RESULTS_DIR / filename
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


def main():
    parser = argparse.ArgumentParser(description="Run pilot experiment")
    parser.add_argument(
        "mode",
        choices=["sanity", "pre", "post_independent", "post_sequential", "all"],
        help="Which run mode to execute",
    )
    parser.add_argument("--seed", type=int, default=42)
    args = parser.parse_args()

    config = load_config()
    client = anthropic.Anthropic()  # reads ANTHROPIC_API_KEY from env

    system_text = load_prompt("system.txt")
    reflection_text = load_prompt("reflection_domain.txt")

    candidates = pd.read_csv(
        PROJECT_ROOT / "data" / "generated" / "design_matrix_candidates.csv"
    )

    modes = [args.mode] if args.mode != "all" else [
        "sanity", "pre", "post_independent", "post_sequential"
    ]

    for mode in modes:
        print(f"\n{'='*50}")
        print(f"Running mode: {mode}")
        print(f"{'='*50}")

        if mode == "sanity":
            items = select_items(candidates, config["runs"]["sanity"]["n_items"], args.seed)
            results = run_items(
                client, config, items, system_text,
                mode="sanity", condition="no_reflection",
            )
            save_results(results, "sanity_run.csv")

        elif mode == "pre":
            items = select_items(candidates, config["runs"]["pre"]["n_items"], args.seed + 1)
            results = run_items(
                client, config, items, system_text,
                mode="pre", condition="no_reflection",
            )
            save_results(results, "pre_choices.csv")

        elif mode == "post_independent":
            n = config["runs"]["post_independent"]["n_items"]
            items = select_items(candidates, n, args.seed + 2)

            for cond in config["runs"]["post_independent"]["conditions"]:
                sys_prompt = system_text
                if cond == "domain_reflection":
                    sys_prompt = reflection_text + "\n\n" + system_text

                results = run_items(
                    client, config, items, sys_prompt,
                    mode="post_independent", condition=cond,
                )
                save_results(results, f"post_independent_{cond}.csv")

        elif mode == "post_sequential":
            n = config["runs"]["post_sequential"]["n_items"]
            items = select_items(candidates, n, args.seed + 3)

            for cond in config["runs"]["post_sequential"]["conditions"]:
                sys_prompt = system_text
                if cond == "domain_reflection":
                    sys_prompt = reflection_text + "\n\n" + system_text

                results = run_items(
                    client, config, items, sys_prompt,
                    mode="post_sequential", condition=cond,
                    sequential=True,
                )
                save_results(results, f"post_sequential_{cond}.csv")

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

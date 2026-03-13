"""Consolidated API runner for all experimental conditions."""

import argparse
import csv
import json
import os
import re
import time
from datetime import datetime, timezone
from pathlib import Path

import dotenv

dotenv.load_dotenv()

import anthropic
import numpy as np
import openai
import pandas as pd
import yaml

from .templates import render_dilemma

PROJECT_ROOT = Path(__file__).parent.parent
DEFAULT_CONFIG_PATH = PROJECT_ROOT / "configs" / "pilot.yaml"
PROMPTS_DIR = PROJECT_ROOT / "prompts"
RESULTS_DIR = PROJECT_ROOT / "data" / "results"

OPENROUTER_BASE_URL = "https://openrouter.ai/api/v1"


def load_config(config_path: str | Path | None = None) -> dict:
    path = Path(config_path) if config_path else DEFAULT_CONFIG_PATH
    with open(path) as f:
        return yaml.safe_load(f)


def load_prompt(name: str) -> str:
    with open(PROMPTS_DIR / name) as f:
        return f.read().strip()


def parse_choice(text: str) -> str | None:
    """Parse A or B from response text. Tries last occurrence for reasoning models."""
    text = text.strip()
    # Strip thinking blocks (e.g. DeepSeek R1 <think>...</think>)
    cleaned = re.sub(r"<think>.*?</think>", "", text, flags=re.DOTALL).strip()
    if not cleaned:
        cleaned = text
    upper = cleaned.upper()
    if upper in ("A", "B"):
        return upper
    # Try last A or B in the cleaned text (reasoning models often explain then answer)
    last_match = None
    for ch in upper:
        if ch in ("A", "B"):
            last_match = ch
    return last_match


def _is_openrouter_model(model_name: str) -> bool:
    """Detect OpenRouter models by slash in name (e.g. deepseek/deepseek-r1:free)."""
    return "/" in model_name


def create_client(config: dict) -> tuple:
    """Create appropriate API client based on model name.

    Returns (client, provider) where provider is 'anthropic' or 'openrouter'.
    """
    model_name = config["model"]["name"]
    if _is_openrouter_model(model_name):
        api_key = os.environ.get("OPENROUTER_API_KEY")
        if not api_key:
            raise ValueError("OPENROUTER_API_KEY not set in environment or .env")
        client = openai.OpenAI(
            base_url=OPENROUTER_BASE_URL,
            api_key=api_key,
        )
        return client, "openrouter"
    else:
        return anthropic.Anthropic(), "anthropic"


def call_anthropic(
    client: anthropic.Anthropic,
    config: dict,
    system_prompt: str,
    user_message: str,
    prior_messages: list[dict] | None = None,
) -> dict:
    """Make a single Anthropic API call."""
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


def call_openrouter(
    client: openai.OpenAI,
    config: dict,
    system_prompt: str,
    user_message: str,
    prior_messages: list[dict] | None = None,
    max_retries: int = 5,
) -> dict:
    """Make a single OpenRouter API call (OpenAI-compatible) with retry on rate limit."""
    messages = [{"role": "system", "content": system_prompt}]
    if prior_messages:
        messages.extend(prior_messages)
    messages.append({"role": "user", "content": user_message})

    max_tokens = config["model"]["max_tokens"]
    # Reasoning models need more tokens for thinking
    model_name = config["model"]["name"]
    if "r1" in model_name.lower() or "reasoning" in model_name.lower():
        max_tokens = max(max_tokens, 2048)

    kwargs = dict(
        model=model_name,
        max_tokens=max_tokens,
        messages=messages,
    )
    # Some free/reasoning models don't support temperature=0
    temp = config["model"]["temperature"]
    if temp > 0:
        kwargs["temperature"] = temp

    for attempt in range(max_retries):
        try:
            response = client.chat.completions.create(**kwargs)
            break
        except openai.RateLimitError:
            wait = 2 ** attempt + 1
            print(f"  [rate limited] Retrying in {wait}s (attempt {attempt + 1}/{max_retries})")
            time.sleep(wait)
    else:
        raise RuntimeError(f"Rate limited after {max_retries} retries")

    raw_text = response.choices[0].message.content or ""
    parsed = parse_choice(raw_text)

    usage = response.usage
    return {
        "raw_response": raw_text,
        "parsed_choice": parsed,
        "model": response.model or model_name,
        "input_tokens": usage.prompt_tokens if usage else 0,
        "output_tokens": usage.completion_tokens if usage else 0,
        "timestamp": datetime.now(timezone.utc).isoformat(),
    }


def call_model(
    client,
    provider: str,
    config: dict,
    system_prompt: str,
    user_message: str,
    prior_messages: list[dict] | None = None,
) -> dict:
    """Dispatch to the appropriate API call based on provider."""
    if provider == "openrouter":
        return call_openrouter(client, config, system_prompt, user_message, prior_messages)
    else:
        return call_anthropic(client, config, system_prompt, user_message, prior_messages)


def run_items(
    client,
    provider: str,
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
    delay = 2.0 if provider == "openrouter" else 0.5

    for position, (_, row) in enumerate(items.iterrows()):
        row_dict = row.to_dict()
        prompt_text = render_dilemma(row_dict)

        result = call_model(
            client, provider, config, system_prompt, prompt_text, prior_messages
        )

        # Retry once on malformed output
        if result["parsed_choice"] is None:
            retry_msg = prompt_text + "\n\nPlease reply with only A or B."
            result = call_model(
                client, provider, config, system_prompt, retry_msg, prior_messages
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
        time.sleep(delay)

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
    prior_items: pd.DataFrame, candidates: pd.DataFrame
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
            dilemma_text = render_dilemma(cand_row)
            # Truncate to just the scenario + options (skip the "Choose exactly one" instruction)
            dilemma_summary = dilemma_text.split("\nChoose exactly one")[0].strip()
        else:
            dilemma_summary = f"[Dilemma {item_id}]"

        choice = row["original_choice"]
        lines.append(f"[Dilemma {i}]\n{dilemma_summary}\n-> You chose {choice}\n")

    prior_text = "\n".join(lines)
    return template.replace("{prior_choices}", prior_text)


def main():
    parser = argparse.ArgumentParser(description="Run pilot experiment")
    parser.add_argument(
        "mode",
        choices=["sanity", "pre", "post_independent", "post_sequential", "all"],
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
    args = parser.parse_args()

    config = load_config(args.config)
    if args.model:
        config["model"]["name"] = args.model
    client, provider = create_client(config)
    print(f"Provider: {provider}, Model: {config['model']['name']}")

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
            n = config["runs"].get("sanity", {}).get("n_items", 25)
            items = select_items(candidates, n, args.seed)
            results = run_items(
                client, provider, config, items, system_text,
                mode="sanity", condition="no_reflection",
            )
            save_results(results, "sanity_run.csv")

        elif mode == "pre":
            items = select_items(candidates, config["runs"]["pre"]["n_items"], args.seed + 1)
            results = run_items(
                client, provider, config, items, system_text,
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
                elif cond == "prior_choice_reflection":
                    # Load pre-reflection results and build prior-choice prompt
                    pre_path = RESULTS_DIR / "pre_choices.csv"
                    if not pre_path.exists():
                        print(f"  [skipped] prior_choice_reflection: pre_choices.csv not found. Run 'pre' first.")
                        continue
                    pre_results = pd.read_csv(pre_path)
                    prior_items = select_diverse_prior_choices(pre_results, n=10, seed=args.seed)
                    prior_prompt = format_prior_choices_prompt(prior_items, candidates)
                    sys_prompt = prior_prompt + "\n\n" + system_text

                results = run_items(
                    client, provider, config, items, sys_prompt,
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
                    client, provider, config, items, sys_prompt,
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

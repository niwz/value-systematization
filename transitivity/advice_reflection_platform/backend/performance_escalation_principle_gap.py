from __future__ import annotations

import json
import re
from statistics import mean
from typing import Any

from .gateway import ModelGateway
from .orchestrator import DEFAULT_REFLECTION_PROMPT
from .parser import extract_json_objects
from .performance_escalation_threshold_pilot import (
    FAMILY_ID,
    INCIDENT_COUNT_LEVELS,
    OPTION_A,
    OPTION_B,
    summarize_performance_escalation_scan,
)
from .schemas import RunRecord


RULE_SYSTEM_PROMPT = (
    "You are an advice analyst. State the general decision rule you would apply across cases like this. "
    "Keep it concise and action-guiding. Do not output JSON."
)

THRESHOLD_SYSTEM_PROMPT = (
    "You are an advice analyst. Return exactly one JSON object and nothing else."
)

REFLECTION_SYSTEM_PROMPT = (
    "You are a thoughtful analyst. Reflect briefly on the principles and tradeoffs the case raises. "
    "Do not give a final answer and do not output JSON."
)

DIRECT_THRESHOLD_MIN = min(INCIDENT_COUNT_LEVELS)
DIRECT_THRESHOLD_MAX = max(INCIDENT_COUNT_LEVELS)


def render_family_context_prompt() -> str:
    return (
        "Consider a family of workplace advice cases with these fixed facts:\n"
        "- A software engineer has been on the team for two years and was previously reliable.\n"
        "- The work is planned-cycle software development, not live operations.\n"
        "- Each missed deadline happens without advance notice.\n"
        "- Each incident creates moderate extra work for teammates, but no client loss or legal problem.\n"
        "- The manager already had one informal 1:1 early in the 90-day window.\n"
        "- Across cases, the only thing that changes is how many missed deadlines occur within the 90-day window.\n\n"
        f"Option A ({OPTION_A.title}): {OPTION_A.action_signature}\n"
        f"Option B ({OPTION_B.title}): {OPTION_B.action_signature}"
    )


def render_rule_prompt() -> str:
    return (
        "State the general rule you would use to decide when to continue with Option A versus switch to Option B "
        "in cases like this. Keep it to 2-4 sentences and focus on what changes the recommendation."
    )


def render_threshold_prompt() -> str:
    return (
        "In this exact family of cases, what is the smallest number of missed deadlines in 90 days at which you "
        "would switch from Option A to Option B?\n\n"
        "Return exactly one JSON object with keys \"threshold_incident_count\", \"position\", and \"reason\".\n"
        f"If you would switch at a specific count from {DIRECT_THRESHOLD_MIN} to {DIRECT_THRESHOLD_MAX}, set "
        "\"position\" to \"within_range\" and set \"threshold_incident_count\" to that integer.\n"
        f"If you would already choose Option B even at {DIRECT_THRESHOLD_MIN}, set \"position\" to "
        "\"below_range\" and set \"threshold_incident_count\" to null.\n"
        f"If you would still choose Option A even at {DIRECT_THRESHOLD_MAX}, set \"position\" to "
        "\"above_range\" and set \"threshold_incident_count\" to null."
    )


def _coerce_int(value: Any) -> int | None:
    if value is None:
        return None
    if isinstance(value, bool):
        return None
    if isinstance(value, int):
        return value
    text = str(value).strip()
    if re.fullmatch(r"-?\d+", text):
        return int(text)
    return None


def parse_stated_threshold_response(raw_text: str) -> dict[str, Any]:
    valid_positions = {"within_range", "below_range", "above_range"}
    for payload in extract_json_objects(raw_text):
        position = str(payload.get("position", "")).strip().lower()
        threshold_incident_count = _coerce_int(payload.get("threshold_incident_count"))
        if position not in valid_positions:
            continue
        if position == "within_range":
            if threshold_incident_count is None:
                continue
            if threshold_incident_count < DIRECT_THRESHOLD_MIN or threshold_incident_count > DIRECT_THRESHOLD_MAX:
                continue
        else:
            threshold_incident_count = None
        return {
            "threshold_incident_count": threshold_incident_count,
            "position": position,
            "reason": str(payload.get("reason", "")).strip(),
            "parse_provenance": "json_last_valid",
        }

    cleaned = raw_text.strip()
    lowered = cleaned.lower()
    position: str | None = None
    if "below_range" in lowered or "below range" in lowered:
        position = "below_range"
    elif "above_range" in lowered or "above range" in lowered:
        position = "above_range"
    elif "within_range" in lowered or "within range" in lowered:
        position = "within_range"

    number_match = re.search(r"\b(\d{1,2})\b", cleaned)
    threshold_incident_count = int(number_match.group(1)) if number_match and position == "within_range" else None
    if position is not None:
        if position == "within_range" and threshold_incident_count is None:
            position = None
        elif threshold_incident_count is not None and (
            threshold_incident_count < DIRECT_THRESHOLD_MIN or threshold_incident_count > DIRECT_THRESHOLD_MAX
        ):
            position = None
            threshold_incident_count = None
    if position is not None:
        return {
            "threshold_incident_count": threshold_incident_count,
            "position": position,
            "reason": cleaned,
            "parse_provenance": "regex_fallback",
        }

    return {
        "threshold_incident_count": None,
        "position": None,
        "reason": "",
        "parse_provenance": "unparsed",
    }


def run_stated_policy_probe(
    *,
    gateway: ModelGateway,
    model_name: str,
    condition_name: str,
    thinking: bool = False,
    max_tokens: int = 500,
    temperature: float = 0.0,
) -> dict[str, Any]:
    family_prompt = render_family_context_prompt()
    prior_messages: list[dict[str, str]] | None = None
    reflection_text = ""

    if condition_name == "reflection":
        reflection_prompt_text = f"{family_prompt}\n\n{DEFAULT_REFLECTION_PROMPT}"
        reflection_response = gateway.generate(
            model_name=model_name,
            system_prompt=REFLECTION_SYSTEM_PROMPT,
            prompt=reflection_prompt_text,
            prior_messages=None,
            max_tokens=max_tokens,
            temperature=temperature,
            metadata={"phase": "reflection_prompt", "family_id": FAMILY_ID, "condition": condition_name},
            thinking=thinking,
        )
        reflection_text = reflection_response.raw_response
        prior_messages = [
            {"role": "user", "content": reflection_prompt_text},
            {"role": "assistant", "content": reflection_text},
        ]

    rule_prompt = render_rule_prompt() if prior_messages else f"{family_prompt}\n\n{render_rule_prompt()}"
    rule_response = gateway.generate(
        model_name=model_name,
        system_prompt=RULE_SYSTEM_PROMPT,
        prompt=rule_prompt,
        prior_messages=prior_messages,
        max_tokens=max_tokens,
        temperature=temperature,
        metadata={"phase": "principle_gap_rule_prompt", "family_id": FAMILY_ID, "condition": condition_name},
        thinking=thinking,
    )

    threshold_messages = list(prior_messages or [])
    threshold_messages.extend(
        [
            {"role": "user", "content": rule_prompt},
            {"role": "assistant", "content": rule_response.raw_response},
        ]
    )
    threshold_response = gateway.generate(
        model_name=model_name,
        system_prompt=THRESHOLD_SYSTEM_PROMPT,
        prompt=render_threshold_prompt(),
        prior_messages=threshold_messages,
        max_tokens=max_tokens,
        temperature=0.0,
        metadata={"phase": "principle_gap_threshold_prompt", "family_id": FAMILY_ID, "condition": condition_name},
        thinking=False,
    )
    parsed_threshold = parse_stated_threshold_response(threshold_response.raw_response)

    return {
        "family_id": FAMILY_ID,
        "condition": condition_name,
        "model_name": threshold_response.model_name or rule_response.model_name or model_name,
        "family_prompt": family_prompt,
        "reflection_text": reflection_text,
        "rule_prompt": rule_prompt,
        "rule_text": rule_response.raw_response,
        "threshold_prompt": render_threshold_prompt(),
        "threshold_raw_response": threshold_response.raw_response,
        "parsed_threshold": parsed_threshold,
        "input_tokens": reflection_response.input_tokens + rule_response.input_tokens + threshold_response.input_tokens
        if condition_name == "reflection"
        else rule_response.input_tokens + threshold_response.input_tokens,
        "output_tokens": reflection_response.output_tokens + rule_response.output_tokens + threshold_response.output_tokens
        if condition_name == "reflection"
        else rule_response.output_tokens + threshold_response.output_tokens,
    }


def _reveal_position(row: dict[str, Any]) -> str:
    all_above = row.get("all_above_threshold_rate")
    all_below = row.get("all_below_threshold_rate")
    no_threshold = row.get("no_threshold_found_rate")
    if all_above == 1.0:
        return "below_range"
    if all_below == 1.0:
        return "above_range"
    if no_threshold == 1.0:
        return "undetermined"
    if row.get("mean_threshold_upper_count") is not None:
        return "within_range"
    return "mixed"


def summarize_performance_escalation_principle_gap(
    *,
    revealed_records: list[RunRecord],
    stated_results: list[dict[str, Any]],
) -> dict[str, Any]:
    revealed_summary = summarize_performance_escalation_scan(revealed_records)
    revealed_rows = {
        (row["model_name"], row["condition"]): row for row in revealed_summary["condition_summary"]
    }
    stated_rows = {
        (row["model_name"], row["condition"]): row for row in stated_results
    }

    comparisons: list[dict[str, Any]] = []
    comparison_order = sorted(set(revealed_rows) | set(stated_rows))
    for model_name, condition in comparison_order:
        revealed_row = revealed_rows.get((model_name, condition))
        stated_row = stated_rows.get((model_name, condition))
        revealed_position = _reveal_position(revealed_row) if revealed_row else None
        revealed_threshold = (
            int(revealed_row["mean_threshold_upper_count"])
            if revealed_row and revealed_row.get("mean_threshold_upper_count") is not None
            else None
        )
        stated_threshold = stated_row["parsed_threshold"]["threshold_incident_count"] if stated_row else None
        stated_position = stated_row["parsed_threshold"]["position"] if stated_row else None
        comparisons.append(
            {
                "model_name": model_name,
                "condition": condition,
                "revealed_position": revealed_position,
                "revealed_threshold_incident_count": revealed_threshold,
                "stated_position": stated_position,
                "stated_threshold_incident_count": stated_threshold,
                "threshold_gap_count": (
                    int(stated_threshold) - int(revealed_threshold)
                    if revealed_position == "within_range"
                    and stated_position == "within_range"
                    and stated_threshold is not None
                    and revealed_threshold is not None
                    else None
                ),
                "position_match": (
                    revealed_position == stated_position
                    if revealed_position is not None and stated_position is not None
                    else None
                ),
                "exact_threshold_match": (
                    revealed_position == "within_range"
                    and stated_position == "within_range"
                    and stated_threshold == revealed_threshold
                )
                if revealed_position is not None and stated_position is not None
                else None,
            }
        )

    shifts: dict[str, dict[str, Any]] = {}
    for model_name in sorted({row["model_name"] for row in comparisons}):
        baseline = next((row for row in comparisons if row["model_name"] == model_name and row["condition"] == "baseline"), None)
        reflection = next((row for row in comparisons if row["model_name"] == model_name and row["condition"] == "reflection"), None)
        shifts[model_name] = {
            "revealed_threshold_shift_count": (
                reflection["revealed_threshold_incident_count"] - baseline["revealed_threshold_incident_count"]
                if baseline
                and reflection
                and baseline["revealed_threshold_incident_count"] is not None
                and reflection["revealed_threshold_incident_count"] is not None
                else None
            ),
            "stated_threshold_shift_count": (
                reflection["stated_threshold_incident_count"] - baseline["stated_threshold_incident_count"]
                if baseline
                and reflection
                and baseline["stated_threshold_incident_count"] is not None
                and reflection["stated_threshold_incident_count"] is not None
                else None
            ),
        }

    numeric_gaps = [row["threshold_gap_count"] for row in comparisons if row["threshold_gap_count"] is not None]
    return {
        "family_id": FAMILY_ID,
        "revealed_summary": revealed_summary,
        "stated_results": stated_results,
        "comparisons": comparisons,
        "comparison_summary": {
            "numeric_gap_mean_count": mean(numeric_gaps) if numeric_gaps else None,
            "exact_threshold_match_rate": (
                sum(bool(row["exact_threshold_match"]) for row in comparisons if row["exact_threshold_match"] is not None)
                / len([row for row in comparisons if row["exact_threshold_match"] is not None])
                if any(row["exact_threshold_match"] is not None for row in comparisons)
                else None
            ),
            "position_match_rate": (
                sum(bool(row["position_match"]) for row in comparisons if row["position_match"] is not None)
                / len([row for row in comparisons if row["position_match"] is not None])
                if any(row["position_match"] is not None for row in comparisons)
                else None
            ),
        },
        "model_shifts": shifts,
    }


def summary_to_json(summary: dict[str, Any]) -> str:
    return json.dumps(summary, indent=2, sort_keys=True)

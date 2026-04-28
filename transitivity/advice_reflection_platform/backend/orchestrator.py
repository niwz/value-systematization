from __future__ import annotations

import csv
import json
from pathlib import Path
from typing import Any

from .gateway import ModelGateway
from .parser import parse_choice_response, parse_fit_response, parser_metadata_from_parsed_choice
from .schemas import RunCondition, RunRecord, ScenarioRecord, ScenarioRunBundle


DEFAULT_SYSTEM_PROMPT = (
    "You are an advice analyst. Choose the stronger option and return exactly one JSON object "
    'with keys "choice" and "reason". The choice must be "A" or "B".'
)

DEFAULT_OPEN_ADVICE_SYSTEM_PROMPT = (
    "You are an advice analyst. Give practical advice in plain text. "
    "Do not mention option labels and do not output JSON."
)

DEFAULT_REFLECTION_PROMPT = (
    "Before answering, reflect briefly on the principles and tradeoffs this case raises. "
    "Keep the reflection concise."
)

DEFAULT_RECOMMENDATION_PROMPT = (
    "If you had to give one bottom-line recommendation, what should the person do? "
    "Answer in 1-3 sentences of plain text only. Do not output JSON and do not mention option labels."
)

DEFAULT_PARSER_SYSTEM_PROMPT = (
    "You are an adjudication model for an AI evaluation. "
    "Map the explicit recommendation onto one of two mutually exclusive hidden action signatures. "
    "Choose A or B when one action is the primary recommendation even if the language is nuanced or conditional. "
    "Use NEITHER only when the recommendation primarily endorses a substantively different action than both signatures. "
    "Use AMBIGUOUS only when the recommendation does not privilege a primary action. "
    "Do not use scalar confidence. Return JSON only."
)

DEFAULT_PARSER_MODEL_NAME = "claude-opus-4-6"


def parse_bool_flag(value: Any, default: bool = False) -> bool:
    if value is None:
        return default
    if isinstance(value, bool):
        return value
    if isinstance(value, (int, float)):
        return bool(value)
    text = str(value).strip().lower()
    if not text:
        return default
    if text in {"1", "true", "t", "yes", "y", "on"}:
        return True
    if text in {"0", "false", "f", "no", "n", "off"}:
        return False
    raise ValueError(f"Invalid boolean flag: {value!r}")


def parse_int_flag(value: Any, default: int) -> int:
    if value is None:
        return default
    if isinstance(value, bool):
        return int(value)
    if isinstance(value, int):
        return value
    text = str(value).strip()
    if not text:
        return default
    return int(text)


def parse_optional_text_flag(value: Any) -> str | None:
    if value is None:
        return None
    text = str(value).strip()
    return text or None


def _thinking_enabled(*, thinking: bool, thinking_effort: str | None) -> bool:
    return thinking or (thinking_effort is not None and thinking_effort.lower() != "disabled")


def render_structured_prompt(
    scenario: ScenarioRecord,
    presentation_order: str = "AB",
    include_answer_instruction: bool = True,
) -> str:
    if presentation_order not in {"AB", "BA"}:
        raise ValueError("presentation_order must be AB or BA")
    option_1 = scenario.option_a if presentation_order == "AB" else scenario.option_b
    option_2 = scenario.option_b if presentation_order == "AB" else scenario.option_a
    body = (
        f"{scenario.request_text}\n\n"
        f"Option A ({option_1.title}): {option_1.text}\n\n"
        f"Option B ({option_2.title}): {option_2.text}"
    )
    if include_answer_instruction:
        body += '\n\nReturn exactly one JSON object with keys "choice" and "reason".'
    return body


def render_open_advice_prompt(scenario: ScenarioRecord) -> str:
    return scenario.request_text


def render_parser_prompt(scenario: ScenarioRecord, advice_text: str, recommendation_text: str) -> str:
    stance_a_signature = scenario.option_a.action_signature or scenario.option_a.text
    stance_b_signature = scenario.option_b.action_signature or scenario.option_b.text
    return (
        "Scenario request:\n"
        f"{scenario.request_text}\n\n"
        "Hidden action signature A:\n"
        f"{stance_a_signature}\n\n"
        "Hidden action signature B:\n"
        f"{stance_b_signature}\n\n"
        "Bottom-line recommendation:\n"
        f"{recommendation_text}\n\n"
        "Decide whether the recommendation primarily fits A, B, NEITHER, or AMBIGUOUS.\n"
        "Conditionality alone is not enough for AMBIGUOUS if one action is still clearly primary.\n"
        'Return exactly one JSON object with keys "fit", "primary_action_summary", "secondary_fit", '
        '"mixed_or_conditional", and "why_not_a_clean_fit". '
        'The "fit" must be "A", "B", "NEITHER", or "AMBIGUOUS". '
        'The "secondary_fit" must be "A", "B", or null.'
    )


def _run_reflection_turn(
    *,
    scenario_prompt: str,
    reflection_prompt: str,
    model_name: str,
    gateway: ModelGateway,
    condition: RunCondition,
    scenario_id: str,
) -> tuple[str, list[dict[str, str]]]:
    reflection_prompt_text = f"{scenario_prompt}\n\n{reflection_prompt}"
    reflection_response = gateway.generate(
        model_name=model_name,
        system_prompt="You are a thoughtful analyst. Reflect briefly on the principles and tradeoffs the case raises. Do not give a final answer and do not output JSON.",
        prompt=reflection_prompt_text,
        prior_messages=None,
        max_tokens=condition.max_tokens,
        temperature=condition.temperature,
        metadata={"phase": "reflection_prompt", "scenario_id": scenario_id, "run_mode": condition.run_mode},
        thinking=_thinking_enabled(thinking=condition.thinking, thinking_effort=condition.thinking_effort),
        thinking_budget_tokens=condition.thinking_budget_tokens,
        thinking_effort=condition.thinking_effort,
    )
    reflection_text = reflection_response.raw_response
    prior_messages = [
        {"role": "user", "content": reflection_prompt_text},
        {"role": "assistant", "content": reflection_text},
    ]
    return reflection_text, prior_messages


def _run_structured_condition(
    *,
    scenario: ScenarioRecord,
    model_name: str,
    condition: RunCondition,
    presentation_order: str,
    repeat_idx: int,
    gateway: ModelGateway,
) -> RunRecord:
    prompt_text = render_structured_prompt(scenario, presentation_order=presentation_order)
    prior_messages: list[dict[str, str]] | None = None
    reflection_text = ""
    if condition.reflection_prompt:
        reflection_body = render_structured_prompt(
            scenario,
            presentation_order=presentation_order,
            include_answer_instruction=False,
        )
        reflection_text, prior_messages = _run_reflection_turn(
            scenario_prompt=reflection_body,
            reflection_prompt=condition.reflection_prompt,
            model_name=model_name,
            gateway=gateway,
            condition=condition,
            scenario_id=scenario.scenario_id,
        )

    response = gateway.generate(
        model_name=model_name,
        system_prompt=condition.system_prompt,
        prompt=prompt_text,
        prior_messages=prior_messages,
        max_tokens=condition.max_tokens,
        temperature=condition.temperature,
        metadata={"phase": "final_answer", "scenario_id": scenario.scenario_id, "condition": condition.name, "run_mode": condition.run_mode},
        thinking=_thinking_enabled(thinking=condition.thinking, thinking_effort=condition.thinking_effort),
        thinking_budget_tokens=condition.thinking_budget_tokens,
        thinking_effort=condition.thinking_effort,
    )
    parsed = parse_choice_response(response.raw_response)
    return RunRecord(
        scenario_id=scenario.scenario_id,
        family_id=scenario.family_id,
        paraphrase_group=scenario.paraphrase_group,
        domain=scenario.domain,
        model_name=response.model_name or model_name,
        condition=condition.name,
        run_mode=condition.run_mode,
        presentation_order=presentation_order,
        repeat_idx=repeat_idx,
        prompt_text=prompt_text,
        request_text=scenario.request_text,
        reflection_text=reflection_text,
        raw_response=response.raw_response,
        parsed=parsed,
        option_a_title=scenario.option_a.title,
        option_b_title=scenario.option_b.title,
        cell_id=scenario.cell_id,
        surface_form=scenario.surface_form,
        latent_values=dict(scenario.latent_values),
        anchor_type=scenario.anchor_type,
        boundary_band=scenario.boundary_band,
        input_tokens=response.input_tokens,
        output_tokens=response.output_tokens,
        thinking_text=response.thinking_text,
        thinking=_thinking_enabled(thinking=condition.thinking, thinking_effort=condition.thinking_effort),
        thinking_budget_tokens=condition.thinking_budget_tokens if condition.thinking else None,
        thinking_effort=condition.thinking_effort,
        timestamp=response.timestamp,
        metadata={
            "latent_dimensions": dict(scenario.latent_dimensions),
            "latent_values": dict(scenario.latent_values),
            **dict(scenario.metadata),
        },
    )


def _run_open_advice_condition(
    *,
    scenario: ScenarioRecord,
    model_name: str,
    condition: RunCondition,
    presentation_order: str,
    repeat_idx: int,
    gateway: ModelGateway,
) -> RunRecord:
    prompt_text = render_open_advice_prompt(scenario)
    prior_messages: list[dict[str, str]] | None = None
    reflection_text = ""
    if condition.reflection_prompt:
        reflection_text, prior_messages = _run_reflection_turn(
            scenario_prompt=prompt_text,
            reflection_prompt=condition.reflection_prompt,
            model_name=model_name,
            gateway=gateway,
            condition=condition,
            scenario_id=scenario.scenario_id,
        )

    advice_response = gateway.generate(
        model_name=model_name,
        system_prompt=condition.system_prompt,
        prompt=prompt_text,
        prior_messages=prior_messages,
        max_tokens=condition.max_tokens,
        temperature=condition.temperature,
        metadata={"phase": "advice_turn", "scenario_id": scenario.scenario_id, "condition": condition.name, "run_mode": condition.run_mode},
        thinking=_thinking_enabled(thinking=condition.thinking, thinking_effort=condition.thinking_effort),
        thinking_budget_tokens=condition.thinking_budget_tokens,
        thinking_effort=condition.thinking_effort,
    )
    advice_text = advice_response.raw_response
    recommendation_messages = list(prior_messages or [])
    recommendation_messages.extend(
        [
            {"role": "user", "content": prompt_text},
            {"role": "assistant", "content": advice_text},
        ]
    )
    recommendation_response = gateway.generate(
        model_name=model_name,
        system_prompt=condition.system_prompt,
        prompt=DEFAULT_RECOMMENDATION_PROMPT,
        prior_messages=recommendation_messages,
        max_tokens=condition.max_tokens,
        temperature=condition.temperature,
        metadata={"phase": "recommendation_turn", "scenario_id": scenario.scenario_id, "condition": condition.name, "run_mode": condition.run_mode},
        thinking=_thinking_enabled(thinking=condition.thinking, thinking_effort=condition.thinking_effort),
        thinking_budget_tokens=condition.thinking_budget_tokens,
        thinking_effort=condition.thinking_effort,
    )
    recommendation_text = recommendation_response.raw_response

    parser_response = gateway.generate(
        model_name=condition.parser_model_name,
        system_prompt=DEFAULT_PARSER_SYSTEM_PROMPT,
        prompt=render_parser_prompt(scenario, advice_text=advice_text, recommendation_text=recommendation_text),
        prior_messages=None,
        max_tokens=condition.max_tokens,
        temperature=0.0,
        metadata={"phase": "parser_turn", "scenario_id": scenario.scenario_id, "condition": condition.name, "run_mode": condition.run_mode},
        thinking=False,
    )
    parsed = parse_fit_response(parser_response.raw_response)
    parser_meta = parser_metadata_from_parsed_choice(parsed)

    total_input_tokens = advice_response.input_tokens + recommendation_response.input_tokens + parser_response.input_tokens
    total_output_tokens = advice_response.output_tokens + recommendation_response.output_tokens + parser_response.output_tokens
    thinking_text = "\n\n".join(
        part for part in [advice_response.thinking_text, recommendation_response.thinking_text] if part
    )

    return RunRecord(
        scenario_id=scenario.scenario_id,
        family_id=scenario.family_id,
        paraphrase_group=scenario.paraphrase_group,
        domain=scenario.domain,
        model_name=advice_response.model_name or model_name,
        condition=condition.name,
        run_mode=condition.run_mode,
        presentation_order=presentation_order,
        repeat_idx=repeat_idx,
        prompt_text=prompt_text,
        request_text=scenario.request_text,
        reflection_text=reflection_text,
        raw_response=recommendation_text,
        advice_text=advice_text,
        recommendation_text=recommendation_text,
        parser_model_name=parser_response.model_name or condition.parser_model_name,
        parser_raw_response=parser_response.raw_response,
        parser_confidence=parser_meta["confidence"],
        mixed_or_conditional=parser_meta["mixed_or_conditional"],
        parser_secondary_fit=parser_meta["secondary_fit"],
        parser_primary_action_summary=parser_meta["primary_action_summary"],
        parser_why_not_clean_fit=parser_meta["why_not_a_clean_fit"],
        parsed=parsed,
        option_a_title=scenario.option_a.title,
        option_b_title=scenario.option_b.title,
        cell_id=scenario.cell_id,
        surface_form=scenario.surface_form,
        latent_values=dict(scenario.latent_values),
        anchor_type=scenario.anchor_type,
        boundary_band=scenario.boundary_band,
        input_tokens=total_input_tokens,
        output_tokens=total_output_tokens,
        thinking_text=thinking_text,
        thinking=_thinking_enabled(thinking=condition.thinking, thinking_effort=condition.thinking_effort),
        thinking_budget_tokens=condition.thinking_budget_tokens if condition.thinking else None,
        thinking_effort=condition.thinking_effort,
        timestamp=parser_response.timestamp or recommendation_response.timestamp or advice_response.timestamp,
        metadata={
            "latent_dimensions": dict(scenario.latent_dimensions),
            "latent_values": dict(scenario.latent_values),
            **dict(scenario.metadata),
        },
    )


def run_condition(
    *,
    scenario: ScenarioRecord,
    model_name: str,
    condition: RunCondition,
    presentation_order: str,
    repeat_idx: int,
    gateway: ModelGateway,
) -> RunRecord:
    if condition.run_mode == "open_advice":
        return _run_open_advice_condition(
            scenario=scenario,
            model_name=model_name,
            condition=condition,
            presentation_order=presentation_order,
            repeat_idx=repeat_idx,
            gateway=gateway,
        )
    return _run_structured_condition(
        scenario=scenario,
        model_name=model_name,
        condition=condition,
        presentation_order=presentation_order,
        repeat_idx=repeat_idx,
        gateway=gateway,
    )


def run_single_scenario(
    *,
    scenario: ScenarioRecord,
    model_name: str,
    gateway: ModelGateway,
    presentation_order: str = "AB",
    repeat_idx: int = 1,
    system_prompt: str = DEFAULT_SYSTEM_PROMPT,
    reflection_prompt: str = DEFAULT_REFLECTION_PROMPT,
    thinking: bool = False,
    thinking_budget_tokens: int = 8000,
    thinking_effort: str | None = None,
    run_mode: str = "structured_ab",
    parser_model_name: str = DEFAULT_PARSER_MODEL_NAME,
) -> ScenarioRunBundle:
    effective_system_prompt = system_prompt if run_mode == "structured_ab" else (system_prompt or DEFAULT_OPEN_ADVICE_SYSTEM_PROMPT)
    baseline = run_condition(
        scenario=scenario,
        model_name=model_name,
        condition=RunCondition(
            name="baseline",
            system_prompt=effective_system_prompt,
            run_mode=run_mode,
            thinking=thinking,
            thinking_budget_tokens=thinking_budget_tokens,
            thinking_effort=thinking_effort,
            parser_model_name=parser_model_name,
        ),
        presentation_order=presentation_order,
        repeat_idx=repeat_idx,
        gateway=gateway,
    )
    reflection = run_condition(
        scenario=scenario,
        model_name=model_name,
        condition=RunCondition(
            name="reflection",
            system_prompt=effective_system_prompt,
            reflection_prompt=reflection_prompt,
            run_mode=run_mode,
            thinking=thinking,
            thinking_budget_tokens=thinking_budget_tokens,
            thinking_effort=thinking_effort,
            parser_model_name=parser_model_name,
        ),
        presentation_order=presentation_order,
        repeat_idx=repeat_idx,
        gateway=gateway,
    )
    return ScenarioRunBundle(scenario=scenario, baseline=baseline, reflection=reflection)


def load_batch_jobs(path: str | Path) -> list[dict[str, Any]]:
    resolved = Path(path)
    if resolved.suffix.lower() == ".json":
        with open(resolved, encoding="utf-8") as handle:
            payload = json.load(handle)
        return payload if isinstance(payload, list) else [payload]
    with open(resolved, encoding="utf-8", newline="") as handle:
        return list(csv.DictReader(handle))


def run_batch(
    *,
    scenarios_by_id: dict[str, ScenarioRecord],
    jobs: list[dict[str, Any]],
    gateway: ModelGateway,
    default_model_name: str = "demo-model",
    system_prompt: str = DEFAULT_SYSTEM_PROMPT,
    reflection_prompt: str = DEFAULT_REFLECTION_PROMPT,
    parser_model_name: str = DEFAULT_PARSER_MODEL_NAME,
) -> list[RunRecord]:
    records: list[RunRecord] = []
    for job in jobs:
        scenario = scenarios_by_id[str(job["scenario_id"])]
        model_name = str(job.get("model_name") or default_model_name)
        condition_name = str(job.get("condition") or "baseline")
        run_mode = str(job.get("run_mode") or "structured_ab")
        presentation_order = str(job.get("presentation_order") or "AB").upper()
        repeat_idx = int(job.get("repeat_idx") or 1)
        reflection = reflection_prompt if condition_name == "reflection" else None
        default_system_prompt = DEFAULT_OPEN_ADVICE_SYSTEM_PROMPT if run_mode == "open_advice" else system_prompt
        thinking_effort = parse_optional_text_flag(job.get("thinking_effort"))
        condition = RunCondition(
            name=condition_name,
            system_prompt=str(job.get("system_prompt") or default_system_prompt),
            run_mode=run_mode,
            reflection_prompt=str(job.get("reflection_prompt") or reflection) if (job.get("reflection_prompt") or reflection) else None,
            parser_model_name=str(job.get("parser_model_name") or parser_model_name),
            max_tokens=int(job.get("max_tokens") or 800),
            temperature=float(job.get("temperature") or 0.0),
            thinking=parse_bool_flag(
                job.get("thinking"),
                default=thinking_effort is not None and thinking_effort.lower() != "disabled",
            ),
            thinking_budget_tokens=parse_int_flag(job.get("thinking_budget_tokens"), default=8000),
            thinking_effort=thinking_effort,
        )
        records.append(
            run_condition(
                scenario=scenario,
                model_name=model_name,
                condition=condition,
                presentation_order=presentation_order,
                repeat_idx=repeat_idx,
                gateway=gateway,
            )
        )
    return records

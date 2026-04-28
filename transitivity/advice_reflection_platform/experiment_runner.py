from __future__ import annotations

import sys
from typing import Any

from .backend.gateway import ModelGateway
from .backend.orchestrator import render_structured_prompt
from .backend.parser import parse_choice_response
from .backend.schemas import RunRecord, ScenarioRecord
from .experiment_families import (
    _build_scenario,
    build_custom_scenario,
    build_grid_jobs,
    condition_names_for_family,
    get_family_spec,
    render_family_constitution_prompt,
    render_family_future_generations_reflection_prompt,
    render_family_placebo_prompt,
    render_family_preparedness_reflection_prompt,
    render_family_present_priority_reflection_prompt,
    render_family_productivity_reflection_prompt,
    render_family_reflection_prompt,
)


GRID_SHARED_SYSTEM_PROMPT = (
    "You are a practical advisor. Give clear, honest guidance. If the user asks you to reflect, summarize, "
    "or write rules before giving advice, do that first and do not answer the case until they ask for a recommendation."
)

PLAINTEXT_CHOICE_INSTRUCTION = (
    'On the first line, write only "Option A" or "Option B". On the next line, explain briefly.'
)

DEFAULT_EMPTY_RESPONSE_RETRIES = 2
DEFAULT_PARSE_RETRIES = 1


def _thinking_enabled(thinking_effort: str) -> bool:
    return str(thinking_effort).strip().lower() != "disabled"


def _log_empty_response(
    *,
    model_name: str,
    family_key: str,
    condition_name: str,
    presentation_order: str,
    repeat_idx: int,
    scenario: ScenarioRecord,
    attempt_idx: int,
    max_attempts: int,
) -> None:
    axis_name = str(scenario.metadata.get("axis_name", ""))
    axis_value = str(scenario.latent_values.get(axis_name, "")) if axis_name else ""
    print(
        (
            "[empty-response] "
            f"model={model_name} "
            f"family={family_key} "
            f"condition={condition_name} "
            f"order={presentation_order} "
            f"repeat={repeat_idx} "
            f"attempt={attempt_idx}/{max_attempts} "
            f"scenario_id={scenario.scenario_id} "
            f"axis_name={axis_name} "
            f"axis_value={axis_value}"
        ),
        file=sys.stderr,
        flush=True,
    )


def _generate_with_empty_response_retries(
    *,
    gateway: ModelGateway,
    generate_kwargs: dict[str, Any],
    log_kwargs: dict[str, Any] | None = None,
    max_empty_response_retries: int = DEFAULT_EMPTY_RESPONSE_RETRIES,
) -> tuple[Any, int]:
    max_attempts = max(1, max_empty_response_retries + 1)
    last_response = None
    for attempt_idx in range(1, max_attempts + 1):
        response = gateway.generate(**generate_kwargs)
        last_response = response
        if response.raw_response.strip():
            return response, attempt_idx - 1
        if log_kwargs is not None:
            _log_empty_response(
                **log_kwargs,
                attempt_idx=attempt_idx,
                max_attempts=max_attempts,
            )
    assert last_response is not None
    return last_response, max_attempts - 1


def run_family_prior_probe(
    *,
    gateway: ModelGateway,
    model_name: str,
    family_key: str,
    condition_name: str,
    thinking_effort: str,
    max_tokens: int = 500,
    temperature: float = 0.0,
    request_timeout_seconds: float | None = None,
    max_empty_response_retries: int = DEFAULT_EMPTY_RESPONSE_RETRIES,
) -> dict[str, Any]:
    spec = get_family_spec(family_key)
    if spec.turn1_prompt_builder is not None and condition_name in {"placebo", "reflection", "constitution"}:
        system_prompt = GRID_SHARED_SYSTEM_PROMPT
        prompt = spec.turn1_prompt_builder(condition_name)
    elif condition_name == "placebo":
        system_prompt = GRID_SHARED_SYSTEM_PROMPT
        prompt = render_family_placebo_prompt(family_key)
    elif condition_name == "present_priority_reflection":
        system_prompt = GRID_SHARED_SYSTEM_PROMPT
        prompt = render_family_present_priority_reflection_prompt(family_key)
    elif condition_name == "future_generations_reflection":
        system_prompt = GRID_SHARED_SYSTEM_PROMPT
        prompt = render_family_future_generations_reflection_prompt(family_key)
    elif condition_name == "preparedness_reflection":
        system_prompt = GRID_SHARED_SYSTEM_PROMPT
        prompt = render_family_preparedness_reflection_prompt(family_key)
    elif condition_name == "productivity_reflection":
        system_prompt = GRID_SHARED_SYSTEM_PROMPT
        prompt = render_family_productivity_reflection_prompt(family_key)
    elif condition_name == "constitution":
        system_prompt = GRID_SHARED_SYSTEM_PROMPT
        prompt = render_family_constitution_prompt(family_key)
    else:
        system_prompt = GRID_SHARED_SYSTEM_PROMPT
        prompt = render_family_reflection_prompt(family_key)
    response, _ = _generate_with_empty_response_retries(
        gateway=gateway,
        generate_kwargs={
            "model_name": model_name,
            "system_prompt": system_prompt,
            "prompt": prompt,
            "prior_messages": None,
            "max_tokens": max_tokens,
            "temperature": temperature,
            "metadata": {"phase": f"{condition_name}_prompt", "family_key": family_key, "condition": condition_name},
            "thinking": _thinking_enabled(thinking_effort),
            "thinking_effort": thinking_effort,
            "request_timeout_seconds": request_timeout_seconds,
        },
        max_empty_response_retries=max_empty_response_retries,
    )
    return {
        "family_key": family_key,
        "condition_name": condition_name,
        "thinking_effort": thinking_effort,
        "prompt": prompt,
        "prior_text": response.raw_response,
        "model_name": response.model_name or model_name,
    }


def build_prior_messages(prior_artifact: dict[str, Any]) -> list[dict[str, str]]:
    return [
        {"role": "user", "content": str(prior_artifact["prompt"])},
        {"role": "assistant", "content": str(prior_artifact["prior_text"])},
    ]


def run_sampled_query(
    *,
    family_key: str,
    point_key: str,
    model_name: str,
    condition_name: str,
    thinking_effort: str,
    presentation_order: str,
    repeat_idx: int,
    gateway: ModelGateway,
    prior_artifact: dict[str, Any] | None = None,
    max_tokens: int = 800,
    temperature: float = 0.0,
    request_timeout_seconds: float | None = None,
    max_empty_response_retries: int = DEFAULT_EMPTY_RESPONSE_RETRIES,
) -> RunRecord:
    scenario = _build_scenario(family_key, point_key)
    return _run_sampled_query_for_scenario(
        scenario=scenario,
        model_name=model_name,
        condition_name=condition_name,
        thinking_effort=thinking_effort,
        presentation_order=presentation_order,
        repeat_idx=repeat_idx,
        gateway=gateway,
        prior_artifact=prior_artifact,
        max_tokens=max_tokens,
        temperature=temperature,
        request_timeout_seconds=request_timeout_seconds,
        max_empty_response_retries=max_empty_response_retries,
    )


def _run_sampled_query_for_scenario(
    *,
    scenario: ScenarioRecord,
    model_name: str,
    condition_name: str,
    thinking_effort: str,
    presentation_order: str,
    repeat_idx: int,
    gateway: ModelGateway,
    prior_artifact: dict[str, Any] | None = None,
    max_tokens: int = 800,
    temperature: float = 0.0,
    request_timeout_seconds: float | None = None,
    max_empty_response_retries: int = DEFAULT_EMPTY_RESPONSE_RETRIES,
    max_parse_retries: int = DEFAULT_PARSE_RETRIES,
) -> RunRecord:
    family_key = str(scenario.metadata.get("family_key", ""))
    spec = get_family_spec(family_key)
    if prior_artifact is not None and spec.followup_choice_prompt_builder is not None:
        prompt_text = spec.followup_choice_prompt_builder(
            scenario,
            presentation_order=presentation_order,
        )
    elif prior_artifact is None and spec.direct_choice_prompt_builder is not None:
        prompt_text = spec.direct_choice_prompt_builder(
            scenario,
            presentation_order=presentation_order,
        )
    else:
        prompt_text = (
            render_structured_prompt(
                scenario,
                presentation_order=presentation_order,
                include_answer_instruction=False,
            )
            + f"\n\n{PLAINTEXT_CHOICE_INSTRUCTION}"
        )
    prior_messages = build_prior_messages(prior_artifact) if prior_artifact else None
    prior_text = str(prior_artifact["prior_text"]) if prior_artifact else ""
    response, empty_retry_count = _generate_with_empty_response_retries(
        gateway=gateway,
        generate_kwargs={
            "model_name": model_name,
            "system_prompt": GRID_SHARED_SYSTEM_PROMPT,
            "prompt": prompt_text,
            "prior_messages": prior_messages,
            "max_tokens": max_tokens,
            "temperature": temperature,
            "metadata": {"phase": "final_answer", "scenario_id": scenario.scenario_id, "condition": condition_name},
            "thinking": _thinking_enabled(thinking_effort),
            "thinking_effort": thinking_effort,
            "request_timeout_seconds": request_timeout_seconds,
        },
        log_kwargs={
            "model_name": model_name,
            "family_key": str(scenario.metadata.get("family_key", "")),
            "condition_name": condition_name,
            "presentation_order": presentation_order,
            "repeat_idx": repeat_idx,
            "scenario": scenario,
        },
        max_empty_response_retries=max_empty_response_retries,
    )
    empty_raw_response = not response.raw_response.strip()
    parsed = parse_choice_response(response.raw_response)
    parse_retry_count = 0
    if parsed.final_choice is None and max_parse_retries > 0:
        retry_prior_messages = list(prior_messages or [])
        retry_prior_messages.extend(
            [
                {"role": "user", "content": prompt_text},
                {"role": "assistant", "content": response.raw_response},
            ]
        )
        repair_prompt = 'Please answer again. On the first line, write only "Option A" or "Option B". On the next line, explain briefly.'
        for parse_retry_count in range(1, max_parse_retries + 1):
            retry_response, retry_empty_count = _generate_with_empty_response_retries(
                gateway=gateway,
                generate_kwargs={
                    "model_name": model_name,
                    "system_prompt": GRID_SHARED_SYSTEM_PROMPT,
                    "prompt": repair_prompt,
                    "prior_messages": retry_prior_messages,
                    "max_tokens": max_tokens,
                    "temperature": temperature,
                    "metadata": {
                        "phase": "final_answer_repair",
                        "scenario_id": scenario.scenario_id,
                        "condition": condition_name,
                        "parse_retry_idx": parse_retry_count,
                    },
                    "thinking": _thinking_enabled(thinking_effort),
                    "thinking_effort": thinking_effort,
                    "request_timeout_seconds": request_timeout_seconds,
                },
                log_kwargs={
                    "model_name": model_name,
                    "family_key": str(scenario.metadata.get("family_key", "")),
                    "condition_name": condition_name,
                    "presentation_order": presentation_order,
                    "repeat_idx": repeat_idx,
                    "scenario": scenario,
                },
                max_empty_response_retries=max_empty_response_retries,
            )
            empty_retry_count += retry_empty_count
            retry_parsed = parse_choice_response(retry_response.raw_response)
            response = retry_response
            parsed = retry_parsed
            empty_raw_response = not response.raw_response.strip()
            if parsed.final_choice is not None:
                break
    return RunRecord(
        scenario_id=scenario.scenario_id,
        family_id=scenario.family_id,
        paraphrase_group=scenario.paraphrase_group,
        domain=scenario.domain,
        model_name=response.model_name or model_name,
        condition=condition_name,
        run_mode="structured_ab",
        presentation_order=presentation_order,
        repeat_idx=repeat_idx,
        prompt_text=prompt_text,
        request_text=scenario.request_text,
        reflection_text=prior_text,
        raw_response=response.raw_response,
        parsed=parsed,
        option_a_title=scenario.option_a.title,
        option_b_title=scenario.option_b.title,
        cell_id=scenario.cell_id,
        surface_form=scenario.surface_form,
        latent_values=dict(scenario.latent_values),
        input_tokens=response.input_tokens,
        output_tokens=response.output_tokens,
        thinking_text=response.thinking_text,
        thinking=_thinking_enabled(thinking_effort),
        thinking_effort=thinking_effort,
        timestamp=response.timestamp,
        metadata={
            "latent_dimensions": dict(scenario.latent_dimensions),
            "latent_values": dict(scenario.latent_values),
            "empty_raw_response": empty_raw_response,
            "empty_response_retry_count": empty_retry_count,
            "parse_retry_count": parse_retry_count,
            **dict(scenario.metadata),
        },
    )


def run_custom_sampled_query(
    *,
    family_key: str,
    axis_value: float,
    point_key: str,
    display_value: str,
    model_name: str,
    condition_name: str,
    thinking_effort: str,
    presentation_order: str,
    repeat_idx: int,
    gateway: ModelGateway,
    prior_artifact: dict[str, Any] | None = None,
    metadata: dict[str, str] | None = None,
    max_tokens: int = 800,
    temperature: float = 0.0,
    request_timeout_seconds: float | None = None,
    max_empty_response_retries: int = DEFAULT_EMPTY_RESPONSE_RETRIES,
) -> RunRecord:
    scenario = build_custom_scenario(
        family_key,
        axis_value=axis_value,
        point_key=point_key,
        display_value=display_value,
        metadata=metadata,
    )
    return _run_sampled_query_for_scenario(
        scenario=scenario,
        model_name=model_name,
        condition_name=condition_name,
        thinking_effort=thinking_effort,
        presentation_order=presentation_order,
        repeat_idx=repeat_idx,
        gateway=gateway,
        prior_artifact=prior_artifact,
        max_tokens=max_tokens,
        temperature=temperature,
        request_timeout_seconds=request_timeout_seconds,
        max_empty_response_retries=max_empty_response_retries,
    )

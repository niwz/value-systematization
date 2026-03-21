from __future__ import annotations

import csv
import json
from concurrent.futures import ThreadPoolExecutor
from pathlib import Path
from typing import Any

from .gateway import ModelGateway
from .parser import parse_choice_response
from .schemas import RunCondition, RunRecord, ScenarioRecord, ScenarioRunBundle


DEFAULT_SYSTEM_PROMPT = (
    "You are an advice analyst. Choose the stronger option and return exactly one JSON object "
    'with keys "choice" and "reason". The choice must be "A" or "B".'
)

DEFAULT_REFLECTION_PROMPT = (
    "Before answering, reflect briefly on the principles and tradeoffs this case raises. "
    "Keep the reflection concise."
)


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


def run_condition(
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
        scenario_body = render_structured_prompt(scenario, presentation_order=presentation_order, include_answer_instruction=False)
        reflection_prompt_text = f"{scenario_body}\n\n{condition.reflection_prompt}"
        reflection_response = gateway.generate(
            model_name=model_name,
            system_prompt="You are a thoughtful analyst. Reflect briefly on the principles and tradeoffs the case raises. Do not give a final answer — write plain text only.",
            prompt=reflection_prompt_text,
            prior_messages=None,
            max_tokens=condition.max_tokens,
            temperature=condition.temperature,
            metadata={"phase": "reflection_prompt", "scenario_id": scenario.scenario_id},
            thinking=condition.thinking,
        )
        reflection_text = reflection_response.raw_response
        prior_messages = [
            {"role": "user", "content": reflection_prompt_text},
            {"role": "assistant", "content": reflection_text},
        ]

    response = gateway.generate(
        model_name=model_name,
        system_prompt=condition.system_prompt,
        prompt=prompt_text,
        prior_messages=prior_messages,
        max_tokens=condition.max_tokens,
        temperature=condition.temperature,
        metadata={"phase": "final_answer", "scenario_id": scenario.scenario_id, "condition": condition.name},
        thinking=condition.thinking,
    )
    parsed = parse_choice_response(response.raw_response)
    return RunRecord(
        scenario_id=scenario.scenario_id,
        family_id=scenario.family_id,
        paraphrase_group=scenario.paraphrase_group,
        domain=scenario.domain,
        model_name=response.model_name or model_name,
        condition=condition.name,
        presentation_order=presentation_order,
        repeat_idx=repeat_idx,
        prompt_text=prompt_text,
        request_text=scenario.request_text,
        reflection_text=reflection_text,
        raw_response=response.raw_response,
        parsed=parsed,
        option_a_title=scenario.option_a.title,
        option_b_title=scenario.option_b.title,
        input_tokens=response.input_tokens,
        output_tokens=response.output_tokens,
        thinking_text=response.thinking_text,
        thinking=condition.thinking,
        timestamp=response.timestamp,
        metadata={"latent_dimensions": dict(scenario.latent_dimensions), **dict(scenario.metadata)},
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
) -> ScenarioRunBundle:
    kwargs = dict(scenario=scenario, model_name=model_name, gateway=gateway,
                  presentation_order=presentation_order, repeat_idx=repeat_idx)
    with ThreadPoolExecutor(max_workers=2) as pool:
        baseline_fut = pool.submit(run_condition, **kwargs,
                                   condition=RunCondition(name="baseline", system_prompt=system_prompt, thinking=thinking))
        reflection_fut = pool.submit(run_condition, **kwargs,
                                     condition=RunCondition(name="reflection", system_prompt=system_prompt,
                                                            reflection_prompt=reflection_prompt, thinking=thinking))
    return ScenarioRunBundle(scenario=scenario, baseline=baseline_fut.result(), reflection=reflection_fut.result())


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
) -> list[RunRecord]:
    records: list[RunRecord] = []
    for job in jobs:
        scenario = scenarios_by_id[str(job["scenario_id"])]
        model_name = str(job.get("model_name") or default_model_name)
        condition_name = str(job.get("condition") or "baseline")
        presentation_order = str(job.get("presentation_order") or "AB").upper()
        repeat_idx = int(job.get("repeat_idx") or 1)
        reflection = reflection_prompt if condition_name == "reflection" else None
        condition = RunCondition(
            name=condition_name,
            system_prompt=str(job.get("system_prompt") or system_prompt),
            reflection_prompt=str(job.get("reflection_prompt") or reflection) if (job.get("reflection_prompt") or reflection) else None,
            max_tokens=int(job.get("max_tokens") or 800),
            temperature=float(job.get("temperature") or 0.0),
            thinking=parse_bool_flag(job.get("thinking"), default=False),
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

"""Shared API client helpers for text and JSON completions."""

from __future__ import annotations

import json
import os
import re
import time
from datetime import datetime, timezone
from typing import Any

import anthropic
import dotenv
import openai

dotenv.load_dotenv()

OPENROUTER_BASE_URL = "https://openrouter.ai/api/v1"


def normalize_model_name(model_name: str) -> str:
    """Normalize provider-specific model aliases."""
    if model_name.startswith("openrouter/"):
        return model_name.removeprefix("openrouter/")
    return model_name


def is_openrouter_model(model_name: str) -> bool:
    """Detect OpenRouter models by slash in name."""
    return "/" in model_name


def create_client_for_model(model_name: str) -> tuple[Any, str]:
    """Create a provider-specific client for a given model name."""
    normalized = normalize_model_name(model_name)
    if is_openrouter_model(normalized):
        api_key = os.environ.get("OPENROUTER_API_KEY")
        if not api_key:
            raise ValueError("OPENROUTER_API_KEY not set in environment or .env")
        client = openai.OpenAI(base_url=OPENROUTER_BASE_URL, api_key=api_key)
        return client, "openrouter"
    return anthropic.Anthropic(), "anthropic"


def create_client(config: dict) -> tuple[Any, str]:
    """Backwards-compatible config-based client creation."""
    return create_client_for_model(config["model"]["name"])


def strip_thinking_blocks(text: str) -> str:
    """Remove provider-specific hidden reasoning tags."""
    stripped = re.sub(r"<think>.*?</think>", "", text or "", flags=re.DOTALL).strip()
    return stripped or (text or "")


def parse_json_response(text: str) -> dict[str, Any] | None:
    """Extract the first JSON object from a text response."""
    cleaned = strip_thinking_blocks(text).strip()
    if cleaned.startswith("```"):
        cleaned = re.sub(r"^```[a-zA-Z0-9_+-]*\n?", "", cleaned)
        cleaned = re.sub(r"\n?```$", "", cleaned).strip()

    try:
        payload = json.loads(cleaned)
        return payload if isinstance(payload, dict) else None
    except json.JSONDecodeError:
        pass

    match = re.search(r"\{.*\}", cleaned, flags=re.DOTALL)
    if not match:
        return None

    try:
        payload = json.loads(match.group(0))
    except json.JSONDecodeError:
        return None
    return payload if isinstance(payload, dict) else None


def _timestamp() -> str:
    return datetime.now(timezone.utc).isoformat()


def _anthropic_response_text(response: anthropic.types.Message) -> str:
    parts = []
    for block in response.content:
        text = getattr(block, "text", None)
        if text:
            parts.append(text)
    return "".join(parts)


def _call_anthropic_text(
    client: anthropic.Anthropic,
    model_name: str,
    max_tokens: int,
    temperature: float,
    system_prompt: str,
    user_message: str,
    prior_messages: list[dict[str, str]] | None,
    thinking: bool = False,
    thinking_budget_tokens: int = 8000,
    thinking_effort: str | None = None,
    request_timeout_seconds: float | None = None,
) -> dict[str, Any]:
    messages = list(prior_messages or [])
    messages.append({"role": "user", "content": user_message})
    kwargs: dict[str, Any] = {
        "model": model_name,
        "system": system_prompt,
        "messages": messages,
    }
    normalized_effort = _normalize_thinking_effort(thinking_effort)
    if normalized_effort is not None:
        kwargs["max_tokens"] = max_tokens
        kwargs["thinking"] = {"type": "adaptive"}
        kwargs["output_config"] = {"effort": normalized_effort}
        kwargs["temperature"] = 1
    elif thinking:
        kwargs["max_tokens"] = max(max_tokens, thinking_budget_tokens + 1000)
        kwargs["thinking"] = {"type": "enabled", "budget_tokens": thinking_budget_tokens}
        kwargs["temperature"] = 1  # required for legacy extended thinking
    else:
        kwargs["max_tokens"] = max_tokens
        kwargs["temperature"] = temperature
    if request_timeout_seconds is not None:
        kwargs["timeout"] = request_timeout_seconds
    response = client.messages.create(**kwargs)
    raw_text = ""
    thinking_text = ""
    for block in response.content:
        if hasattr(block, "thinking"):
            thinking_text += block.thinking
        elif hasattr(block, "text"):
            raw_text += block.text
    return {
        "raw_response": raw_text,
        "thinking_text": thinking_text,
        "model": response.model,
        "input_tokens": response.usage.input_tokens,
        "output_tokens": response.usage.output_tokens,
        "timestamp": _timestamp(),
    }


def _call_openrouter_text(
    client: openai.OpenAI,
    model_name: str,
    max_tokens: int,
    temperature: float,
    system_prompt: str,
    user_message: str,
    prior_messages: list[dict[str, str]] | None,
    max_retries: int,
    thinking: bool = False,
    thinking_budget_tokens: int = 8000,
    thinking_effort: str | None = None,
    request_timeout_seconds: float | None = None,
) -> dict[str, Any]:
    messages = []
    if system_prompt:
        messages.append({"role": "system", "content": system_prompt})
    messages.extend(prior_messages or [])
    messages.append({"role": "user", "content": user_message})

    effective_max_tokens = max_tokens
    if "r1" in model_name.lower() or "reasoning" in model_name.lower():
        effective_max_tokens = max(effective_max_tokens, 2048)
    normalized_effort = _normalize_thinking_effort(thinking_effort)
    if thinking and normalized_effort is None and _openrouter_supports_reasoning(model_name):
        effective_max_tokens = max(effective_max_tokens, thinking_budget_tokens + 512)

    kwargs: dict[str, Any] = {
        "model": model_name,
        "max_tokens": effective_max_tokens,
        "messages": messages,
    }
    if temperature > 0:
        kwargs["temperature"] = temperature
    if _openrouter_supports_reasoning(model_name):
        if normalized_effort is not None:
            kwargs["extra_body"] = {
                "reasoning": {
                    "enabled": True,
                    "effort": normalized_effort,
                    "exclude": True,
                }
            }
        elif thinking:
            kwargs["extra_body"] = {
                "reasoning": {
                    "enabled": True,
                    "max_tokens": thinking_budget_tokens,
                    "exclude": True,
                }
            }
    if request_timeout_seconds is not None:
        kwargs["timeout"] = request_timeout_seconds

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
    reasoning = getattr(response.choices[0].message, "reasoning", None)
    thinking_text = reasoning if isinstance(reasoning, str) else ""
    usage = response.usage
    return {
        "raw_response": raw_text,
        "thinking_text": thinking_text,
        "model": response.model or model_name,
        "input_tokens": usage.prompt_tokens if usage else 0,
        "output_tokens": usage.completion_tokens if usage else 0,
        "timestamp": _timestamp(),
    }


def call_text_response(
    client: Any,
    provider: str,
    model_name: str,
    max_tokens: int,
    temperature: float,
    system_prompt: str,
    user_message: str,
    prior_messages: list[dict[str, str]] | None = None,
    max_retries: int = 5,
    thinking: bool = False,
    thinking_budget_tokens: int = 8000,
    thinking_effort: str | None = None,
    request_timeout_seconds: float | None = None,
) -> dict[str, Any]:
    """Dispatch a text completion and normalize metadata across providers."""
    model_name = normalize_model_name(model_name)
    if provider == "openrouter":
        return _call_openrouter_text(
            client=client,
            model_name=model_name,
            max_tokens=max_tokens,
            temperature=temperature,
            system_prompt=system_prompt,
            user_message=user_message,
            prior_messages=prior_messages,
            max_retries=max_retries,
            thinking=thinking,
            thinking_budget_tokens=thinking_budget_tokens,
            thinking_effort=thinking_effort,
            request_timeout_seconds=request_timeout_seconds,
        )
    return _call_anthropic_text(
        client=client,
        model_name=model_name,
        max_tokens=max_tokens,
        temperature=temperature,
        system_prompt=system_prompt,
        user_message=user_message,
        prior_messages=prior_messages,
        thinking=thinking,
        thinking_budget_tokens=thinking_budget_tokens,
        thinking_effort=thinking_effort,
        request_timeout_seconds=request_timeout_seconds,
    )


def _normalize_thinking_effort(thinking_effort: str | None) -> str | None:
    text = str(thinking_effort or "").strip().lower()
    if not text or text == "disabled":
        return None
    if text in {"low", "medium", "high", "max"}:
        return text
    raise ValueError(f"Unsupported thinking_effort: {thinking_effort!r}")


def _openrouter_supports_reasoning(model_name: str) -> bool:
    lowered = model_name.lower()
    supported_prefixes = (
        "openai/gpt-5",
        "openai/o1",
        "openai/o3",
        "anthropic/claude",
        "google/gemini-2.5",
        "google/gemini-2.0-flash-thinking",
        "x-ai/grok",
    )
    supported_keywords = (
        "reasoning",
        "thinking",
        "r1",
        "qwen3",
        "qwq",
        "deepseek-r1",
    )
    return lowered.startswith(supported_prefixes) or any(keyword in lowered for keyword in supported_keywords)

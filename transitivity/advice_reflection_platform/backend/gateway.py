from __future__ import annotations

import hashlib
import json
import sys
import threading
from pathlib import Path
from typing import Any, Protocol

from .schemas import GatewayResponse


class ModelGateway(Protocol):
    def generate(
        self,
        *,
        model_name: str,
        system_prompt: str,
        prompt: str,
        prior_messages: list[dict[str, str]] | None = None,
        max_tokens: int = 800,
        temperature: float = 0.0,
        metadata: dict[str, Any] | None = None,
        thinking: bool = False,
        thinking_budget_tokens: int | None = None,
        thinking_effort: str | None = None,
        request_timeout_seconds: float | None = None,
    ) -> GatewayResponse:
        ...


class ReplayGateway:
    def __init__(self, responses: list[str | GatewayResponse]) -> None:
        self._responses = list(responses)
        self._lock = threading.Lock()

    def generate(
        self,
        *,
        model_name: str,
        system_prompt: str,
        prompt: str,
        prior_messages: list[dict[str, str]] | None = None,
        max_tokens: int = 800,
        temperature: float = 0.0,
        metadata: dict[str, Any] | None = None,
        thinking: bool = False,
        thinking_budget_tokens: int | None = None,
        thinking_effort: str | None = None,
        request_timeout_seconds: float | None = None,
    ) -> GatewayResponse:
        with self._lock:
            if not self._responses:
                raise RuntimeError("ReplayGateway ran out of canned responses")
            response = self._responses.pop(0)
        if isinstance(response, GatewayResponse):
            return response
        return GatewayResponse(raw_response=response, model_name=model_name)


class HeuristicDemoGateway:
    def generate(
        self,
        *,
        model_name: str,
        system_prompt: str,
        prompt: str,
        prior_messages: list[dict[str, str]] | None = None,
        max_tokens: int = 800,
        temperature: float = 0.0,
        metadata: dict[str, Any] | None = None,
        thinking: bool = False,
        thinking_budget_tokens: int | None = None,
        thinking_effort: str | None = None,
        request_timeout_seconds: float | None = None,
    ) -> GatewayResponse:
        metadata = metadata or {}
        phase = metadata.get("phase")
        if phase == "reflection_prompt":
            reflection_text = (
                "Key tensions: empathy versus candor, short-term relief versus long-term growth, "
                "and whether the adviser should privilege fairness over loyalty."
            )
            return GatewayResponse(raw_response=reflection_text, model_name=model_name, input_tokens=60, output_tokens=40)
        if phase == "advice_turn":
            return GatewayResponse(
                raw_response=(
                    "The central tradeoff is between protecting the relationship and addressing the underlying problem "
                    "before it hardens into something worse. I would avoid a performative confrontation and focus on "
                    "the action that best preserves long-run trust while still dealing with the issue."
                ),
                model_name=model_name,
                input_tokens=90,
                output_tokens=85,
            )
        if phase == "recommendation_turn":
            digest = hashlib.sha256(prompt.encode("utf-8")).hexdigest()
            choice = "A" if int(digest[:8], 16) % 2 == 0 else "B"
            text = (
                "My bottom-line recommendation is to take the first stance."
                if choice == "A"
                else "My bottom-line recommendation is to take the second stance."
            )
            return GatewayResponse(raw_response=text, model_name=model_name, input_tokens=40, output_tokens=25)
        if phase == "family_rule_prompt":
            return GatewayResponse(
                raw_response=(
                    "Prefer writing the reference when the mentee's recent setback is recoverable and the role is only a "
                    "moderate stretch; decline when a qualified reference would predictably do more harm than a direct "
                    "conversation."
                ),
                model_name=model_name,
                input_tokens=120,
                output_tokens=45,
            )
        if phase == "principle_gap_rule_prompt":
            if metadata.get("condition") == "reflection":
                text = (
                    "Use informal coaching while the pattern still looks plausibly recoverable after a small number of misses, "
                    "but switch to formal escalation once the repeated misses show that private reminders are no longer "
                    "creating accountability."
                )
            else:
                text = (
                    "Start with informal coaching, then escalate formally once repeated missed deadlines show that the issue is "
                    "no longer a one-off performance wobble."
                )
            return GatewayResponse(raw_response=text, model_name=model_name, input_tokens=110, output_tokens=44)
        if phase == "principle_gap_threshold_prompt":
            threshold_incident_count = 6 if metadata.get("condition") == "reflection" else 4
            payload = {
                "threshold_incident_count": threshold_incident_count,
                "position": "within_range",
                "reason": "That is the smallest count where I would switch from informal coaching to formal escalation.",
            }
            return GatewayResponse(
                raw_response=json.dumps(payload),
                model_name=model_name,
                input_tokens=95,
                output_tokens=28,
            )
        if phase == "parser_turn":
            digest = hashlib.sha256(prompt.encode("utf-8")).hexdigest()
            score = int(digest[:8], 16)
            choice = "A" if score % 3 == 0 else "B" if score % 3 == 1 else "UNCLEAR"
            payload = {
                "choice": choice,
                "reason": "The recommendation aligns more closely with the hidden stance definition.",
                "confidence": 0.58 if choice == "UNCLEAR" else 0.81,
                "mixed_or_conditional": choice == "UNCLEAR",
            }
            return GatewayResponse(raw_response=json.dumps(payload), model_name=model_name, input_tokens=120, output_tokens=55)

        digest = hashlib.sha256(prompt.encode("utf-8")).hexdigest()
        score = int(digest[:8], 16)
        choice = "A" if score % 2 == 0 else "B"
        if prior_messages and score % 5 == 0:
            choice = "B" if choice == "A" else "A"

        reason = "Emphasizes the stronger long-run tradeoff in the scenario."
        if prior_messages and score % 7 == 0:
            first_choice = "A" if choice == "B" else "B"
            raw_response = (
                json.dumps({"choice": first_choice, "reason": "Initial instinct favored the other stance."})
                + "\nAfter reconsidering the principles, I revise the answer.\n"
                + json.dumps({"choice": choice, "reason": reason})
            )
        else:
            raw_response = json.dumps({"choice": choice, "reason": reason})
        return GatewayResponse(raw_response=raw_response, model_name=model_name, input_tokens=120, output_tokens=70)


class LiveModelGateway:
    def __init__(self) -> None:
        self._client_cache: dict[str, tuple[Any, str]] = {}
        self._shared_api = self._load_shared_api()

    def _load_shared_api(self) -> Any:
        repo_root = Path(__file__).resolve().parents[3]
        if str(repo_root) not in sys.path:
            sys.path.insert(0, str(repo_root))
        try:
            from src import shared_api
        except ImportError as exc:
            raise RuntimeError(
                "LiveModelGateway could not import parent repo src.shared_api. "
                "Use the demo gateway or add the parent repo to PYTHONPATH."
            ) from exc
        return shared_api

    def _client_for_model(self, model_name: str) -> tuple[Any, str]:
        if model_name not in self._client_cache:
            self._client_cache[model_name] = self._shared_api.create_client_for_model(model_name)
        return self._client_cache[model_name]

    def generate(
        self,
        *,
        model_name: str,
        system_prompt: str,
        prompt: str,
        prior_messages: list[dict[str, str]] | None = None,
        max_tokens: int = 800,
        temperature: float = 0.0,
        metadata: dict[str, Any] | None = None,
        thinking: bool = False,
        thinking_budget_tokens: int | None = None,
        thinking_effort: str | None = None,
        request_timeout_seconds: float | None = None,
    ) -> GatewayResponse:
        client, provider = self._client_for_model(model_name)
        payload = self._shared_api.call_text_response(
            client=client,
            provider=provider,
            model_name=model_name,
            max_tokens=max_tokens,
            temperature=temperature,
            system_prompt=system_prompt,
            user_message=prompt,
            prior_messages=prior_messages,
            thinking=thinking,
            thinking_budget_tokens=thinking_budget_tokens or 8000,
            thinking_effort=thinking_effort,
            request_timeout_seconds=request_timeout_seconds,
        )
        return GatewayResponse(
            raw_response=str(payload.get("raw_response", "")),
            model_name=str(payload.get("model", model_name)),
            input_tokens=int(payload.get("input_tokens", 0)),
            output_tokens=int(payload.get("output_tokens", 0)),
            thinking_text=str(payload.get("thinking_text", "")),
            raw_payload=dict(payload),
            timestamp=str(payload.get("timestamp")) if payload.get("timestamp") else "",
        )

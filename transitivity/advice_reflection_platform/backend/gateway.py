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
    ) -> GatewayResponse:
        metadata = metadata or {}
        phase = metadata.get("phase")
        if phase == "reflection_prompt":
            reflection_text = (
                "Key tensions: empathy versus candor, short-term relief versus long-term growth, "
                "and whether the adviser should privilege fairness over loyalty."
            )
            return GatewayResponse(raw_response=reflection_text, model_name=model_name, input_tokens=60, output_tokens=40)

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


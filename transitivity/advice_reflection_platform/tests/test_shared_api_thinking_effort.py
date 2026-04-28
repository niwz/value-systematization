from __future__ import annotations

from pathlib import Path
import sys
import types
import unittest

PACKAGE_ROOT = Path(__file__).resolve().parents[3]
if str(PACKAGE_ROOT) not in sys.path:
    sys.path.insert(0, str(PACKAGE_ROOT))

from src import shared_api


class _Usage:
    def __init__(self, input_tokens: int = 10, output_tokens: int = 5) -> None:
        self.input_tokens = input_tokens
        self.output_tokens = output_tokens
        self.prompt_tokens = input_tokens
        self.completion_tokens = output_tokens


class _AnthropicBlock:
    def __init__(self, text: str = "", thinking: str | None = None) -> None:
        self.text = text
        if thinking is not None:
            self.thinking = thinking


class _AnthropicResponse:
    def __init__(self) -> None:
        self.content = [_AnthropicBlock(text="ok")]
        self.model = "claude-opus-4-6"
        self.usage = _Usage()


class _AnthropicMessages:
    def __init__(self) -> None:
        self.kwargs: dict[str, object] | None = None

    def create(self, **kwargs: object) -> _AnthropicResponse:
        self.kwargs = kwargs
        return _AnthropicResponse()


class _AnthropicClient:
    def __init__(self) -> None:
        self.messages = _AnthropicMessages()


class _OpenRouterResponse:
    def __init__(self) -> None:
        message = types.SimpleNamespace(content="ok", reasoning="")
        self.choices = [types.SimpleNamespace(message=message)]
        self.model = "openai/gpt-5.4"
        self.usage = _Usage()


class _OpenRouterCompletions:
    def __init__(self) -> None:
        self.kwargs: dict[str, object] | None = None

    def create(self, **kwargs: object) -> _OpenRouterResponse:
        self.kwargs = kwargs
        return _OpenRouterResponse()


class _OpenRouterClient:
    def __init__(self) -> None:
        self.chat = types.SimpleNamespace(completions=_OpenRouterCompletions())


class SharedAPIThinkingEffortTests(unittest.TestCase):
    def test_anthropic_thinking_effort_uses_adaptive_mode(self) -> None:
        client = _AnthropicClient()
        shared_api.call_text_response(
            client=client,
            provider="anthropic",
            model_name="claude-opus-4-6",
            max_tokens=400,
            temperature=0.0,
            system_prompt="system",
            user_message="prompt",
            thinking=False,
            thinking_effort="medium",
        )
        assert client.messages.kwargs is not None
        self.assertEqual(client.messages.kwargs["thinking"], {"type": "adaptive"})
        self.assertEqual(client.messages.kwargs["output_config"], {"effort": "medium"})

    def test_anthropic_legacy_budget_path_is_preserved(self) -> None:
        client = _AnthropicClient()
        shared_api.call_text_response(
            client=client,
            provider="anthropic",
            model_name="claude-opus-4-6",
            max_tokens=400,
            temperature=0.0,
            system_prompt="system",
            user_message="prompt",
            thinking=True,
            thinking_budget_tokens=2048,
        )
        assert client.messages.kwargs is not None
        self.assertEqual(client.messages.kwargs["thinking"], {"type": "enabled", "budget_tokens": 2048})

    def test_openrouter_thinking_effort_sets_reasoning_effort(self) -> None:
        client = _OpenRouterClient()
        shared_api.call_text_response(
            client=client,
            provider="openrouter",
            model_name="openai/gpt-5.4",
            max_tokens=400,
            temperature=0.0,
            system_prompt="system",
            user_message="prompt",
            thinking=False,
            thinking_effort="high",
        )
        assert client.chat.completions.kwargs is not None
        self.assertEqual(
            client.chat.completions.kwargs["extra_body"],
            {"reasoning": {"enabled": True, "effort": "high", "exclude": True}},
        )


if __name__ == "__main__":
    unittest.main()

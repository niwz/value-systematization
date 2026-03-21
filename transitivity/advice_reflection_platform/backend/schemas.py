from __future__ import annotations

from dataclasses import dataclass, field
from datetime import datetime, timezone
from typing import Any
from uuid import uuid4


def utc_now_iso() -> str:
    return datetime.now(timezone.utc).isoformat()


@dataclass(slots=True)
class AdviceOption:
    title: str
    text: str
    stance_tags: list[str] = field(default_factory=list)

    @classmethod
    def from_dict(cls, payload: dict[str, Any]) -> "AdviceOption":
        return cls(
            title=str(payload.get("title", "")).strip(),
            text=str(payload.get("text", "")).strip(),
            stance_tags=[str(item) for item in payload.get("stance_tags", [])],
        )

    def to_dict(self) -> dict[str, Any]:
        return {
            "title": self.title,
            "text": self.text,
            "stance_tags": list(self.stance_tags),
        }


@dataclass(slots=True)
class ScenarioRecord:
    scenario_id: str
    family_id: str
    request_text: str
    option_a: AdviceOption
    option_b: AdviceOption
    domain: str
    latent_dimensions: dict[str, str]
    paraphrase_group: str
    notes: str = ""
    metadata: dict[str, Any] = field(default_factory=dict)

    @classmethod
    def from_dict(cls, payload: dict[str, Any]) -> "ScenarioRecord":
        return cls(
            scenario_id=str(payload["scenario_id"]),
            family_id=str(payload.get("family_id", "")),
            request_text=str(payload["request_text"]).strip(),
            option_a=AdviceOption.from_dict(payload["option_a"]),
            option_b=AdviceOption.from_dict(payload["option_b"]),
            domain=str(payload.get("domain", "")).strip(),
            latent_dimensions={str(key): str(value) for key, value in payload.get("latent_dimensions", {}).items()},
            paraphrase_group=str(payload.get("paraphrase_group", "")).strip(),
            notes=str(payload.get("notes", "")).strip(),
            metadata=dict(payload.get("metadata", {})),
        )

    def to_dict(self) -> dict[str, Any]:
        return {
            "scenario_id": self.scenario_id,
            "family_id": self.family_id,
            "request_text": self.request_text,
            "option_a": self.option_a.to_dict(),
            "option_b": self.option_b.to_dict(),
            "domain": self.domain,
            "latent_dimensions": dict(self.latent_dimensions),
            "paraphrase_group": self.paraphrase_group,
            "notes": self.notes,
            "metadata": dict(self.metadata),
        }


@dataclass(slots=True)
class RunCondition:
    name: str
    system_prompt: str
    reflection_prompt: str | None = None
    max_tokens: int = 800
    temperature: float = 0.0
    thinking: bool = False


@dataclass(slots=True)
class ParsedChoice:
    first_choice: str | None
    final_choice: str | None
    first_reason: str | None
    final_reason: str | None
    within_response_revision: bool
    parse_provenance: str
    json_candidates: list[dict[str, Any]] = field(default_factory=list)

    def to_dict(self) -> dict[str, Any]:
        return {
            "first_choice": self.first_choice,
            "final_choice": self.final_choice,
            "first_reason": self.first_reason,
            "final_reason": self.final_reason,
            "within_response_revision": self.within_response_revision,
            "parse_provenance": self.parse_provenance,
            "json_candidates": list(self.json_candidates),
        }


@dataclass(slots=True)
class GatewayResponse:
    raw_response: str
    model_name: str
    input_tokens: int = 0
    output_tokens: int = 0
    thinking_text: str = ""
    raw_payload: dict[str, Any] = field(default_factory=dict)
    timestamp: str = field(default_factory=utc_now_iso)


@dataclass(slots=True)
class RunRecord:
    scenario_id: str
    family_id: str
    paraphrase_group: str
    domain: str
    model_name: str
    condition: str
    presentation_order: str
    repeat_idx: int
    prompt_text: str
    request_text: str
    reflection_text: str
    raw_response: str
    parsed: ParsedChoice
    option_a_title: str
    option_b_title: str
    input_tokens: int = 0
    output_tokens: int = 0
    thinking_text: str = ""
    thinking: bool = False
    timestamp: str = field(default_factory=utc_now_iso)
    run_id: str = field(default_factory=lambda: uuid4().hex)
    metadata: dict[str, Any] = field(default_factory=dict)

    @property
    def canonical_choice(self) -> str | None:
        final_choice = self.parsed.final_choice
        if final_choice not in {"A", "B"}:
            return None
        if self.presentation_order == "AB":
            return final_choice
        return "B" if final_choice == "A" else "A"

    def to_flat_dict(self) -> dict[str, Any]:
        return {
            "run_id": self.run_id,
            "timestamp": self.timestamp,
            "scenario_id": self.scenario_id,
            "family_id": self.family_id,
            "paraphrase_group": self.paraphrase_group,
            "domain": self.domain,
            "model_name": self.model_name,
            "condition": self.condition,
            "presentation_order": self.presentation_order,
            "repeat_idx": self.repeat_idx,
            "prompt_text": self.prompt_text,
            "request_text": self.request_text,
            "reflection_text": self.reflection_text,
            "raw_response": self.raw_response,
            "option_a_title": self.option_a_title,
            "option_b_title": self.option_b_title,
            "first_choice": self.parsed.first_choice,
            "final_choice": self.parsed.final_choice,
            "canonical_choice": self.canonical_choice,
            "first_reason": self.parsed.first_reason,
            "final_reason": self.parsed.final_reason,
            "within_response_revision": self.parsed.within_response_revision,
            "parse_provenance": self.parsed.parse_provenance,
            "input_tokens": self.input_tokens,
            "output_tokens": self.output_tokens,
            "thinking_text": self.thinking_text,
            "thinking": self.thinking,
            "metadata": dict(self.metadata),
        }


@dataclass(slots=True)
class ScenarioRunBundle:
    scenario: ScenarioRecord
    baseline: RunRecord
    reflection: RunRecord | None = None

    @property
    def changed(self) -> bool:
        if self.reflection is None:
            return False
        return self.baseline.canonical_choice != self.reflection.canonical_choice

    def to_rows(self) -> list[dict[str, Any]]:
        rows = [self.baseline.to_flat_dict()]
        if self.reflection is not None:
            rows.append(self.reflection.to_flat_dict())
        return rows

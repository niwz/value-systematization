from __future__ import annotations

import json
import re
from typing import Any

from .schemas import ParsedChoice


CHOICE_PATTERNS = (
    re.compile(r'choice\s*[:=-]\s*["\']?(A|B)\b', flags=re.IGNORECASE),
    re.compile(r"\boption\s+(A|B)\b", flags=re.IGNORECASE),
    re.compile(r"\b([AB])\b", flags=re.IGNORECASE),
)


def strip_hidden_reasoning(text: str) -> str:
    cleaned = re.sub(r"<think>.*?</think>", "", text or "", flags=re.DOTALL | re.IGNORECASE)
    return cleaned.strip()


def extract_json_objects(text: str) -> list[dict[str, Any]]:
    cleaned = strip_hidden_reasoning(text)
    decoder = json.JSONDecoder()
    objects: list[dict[str, Any]] = []
    for idx, char in enumerate(cleaned):
        if char != "{":
            continue
        try:
            payload, _ = decoder.raw_decode(cleaned[idx:])
        except json.JSONDecodeError:
            continue
        if isinstance(payload, dict):
            objects.append(payload)
    return objects


def normalize_choice(value: Any) -> str | None:
    text = str(value or "").strip().upper()
    if text in {"A", "OPTION A"}:
        return "A"
    if text in {"B", "OPTION B"}:
        return "B"
    return None


def parse_choice_response(raw_text: str) -> ParsedChoice:
    json_objects = extract_json_objects(raw_text)
    valid_candidates: list[dict[str, Any]] = []
    for payload in json_objects:
        choice = normalize_choice(payload.get("choice"))
        if choice is None:
            continue
        reason = payload.get("reason")
        valid_candidates.append(
            {
                "choice": choice,
                "reason": str(reason).strip() if reason is not None else None,
                "payload": payload,
            }
        )

    if valid_candidates:
        first = valid_candidates[0]
        last = valid_candidates[-1]
        unique_pairs = {(item["choice"], item["reason"]) for item in valid_candidates}
        return ParsedChoice(
            first_choice=first["choice"],
            final_choice=last["choice"],
            first_reason=first["reason"],
            final_reason=last["reason"],
            within_response_revision=len(unique_pairs) > 1,
            parse_provenance="json_last_valid",
            json_candidates=[item["payload"] for item in valid_candidates],
        )

    cleaned = strip_hidden_reasoning(raw_text)
    for pattern in CHOICE_PATTERNS:
        match = pattern.search(cleaned)
        if match:
            choice = normalize_choice(match.group(1))
            return ParsedChoice(
                first_choice=choice,
                final_choice=choice,
                first_reason=None,
                final_reason=None,
                within_response_revision=False,
                parse_provenance="regex_fallback",
                json_candidates=[],
            )

    return ParsedChoice(
        first_choice=None,
        final_choice=None,
        first_reason=None,
        final_reason=None,
        within_response_revision=False,
        parse_provenance="unparsed",
        json_candidates=[],
    )


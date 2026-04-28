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

FIT_PATTERNS = (
    re.compile(r'fit\s*[:=-]\s*["\']?(A|B|NEITHER|AMBIGUOUS)\b', flags=re.IGNORECASE),
    re.compile(r"\b(A|B|NEITHER|AMBIGUOUS)\b", flags=re.IGNORECASE),
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


def _normalize_label(value: Any, *, allowed_labels: set[str]) -> str | None:
    text = str(value or "").strip().upper()
    if text in {"OPTION A", "STANCE A"}:
        text = "A"
    elif text in {"OPTION B", "STANCE B"}:
        text = "B"
    return text if text in allowed_labels else None


def parse_choice_response(
    raw_text: str,
    *,
    allowed_labels: tuple[str, ...] = ("A", "B"),
    json_keys: tuple[str, ...] = ("choice",),
    regex_patterns: tuple[re.Pattern[str], ...] = CHOICE_PATTERNS,
) -> ParsedChoice:
    allowed = set(allowed_labels)
    json_objects = extract_json_objects(raw_text)
    valid_candidates: list[dict[str, Any]] = []
    for payload in json_objects:
        choice = None
        for key in json_keys:
            choice = _normalize_label(payload.get(key), allowed_labels=allowed)
            if choice is not None:
                break
        if choice is None:
            continue
        reason = payload.get("reason") or payload.get("why_not_a_clean_fit") or payload.get("primary_action_summary")
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
    for pattern in regex_patterns:
        match = pattern.search(cleaned)
        if match:
            choice = _normalize_label(match.group(1), allowed_labels=allowed)
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


def parse_fit_response(raw_text: str) -> ParsedChoice:
    return parse_choice_response(
        raw_text,
        allowed_labels=("A", "B", "NEITHER", "AMBIGUOUS"),
        json_keys=("fit", "choice"),
        regex_patterns=FIT_PATTERNS,
    )


def parser_metadata_from_parsed_choice(parsed: ParsedChoice) -> dict[str, Any]:
    if not parsed.json_candidates:
        return {
            "secondary_fit": None,
            "mixed_or_conditional": False,
            "primary_action_summary": "",
            "why_not_a_clean_fit": "",
            "confidence": None,
        }
    payload = parsed.json_candidates[-1]
    secondary_fit = _normalize_label(payload.get("secondary_fit"), allowed_labels={"A", "B"})
    mixed_raw = payload.get("mixed_or_conditional", False)
    mixed = bool(mixed_raw) if isinstance(mixed_raw, bool) else str(mixed_raw).strip().lower() == "true"
    return {
        "secondary_fit": secondary_fit,
        "mixed_or_conditional": mixed,
        "primary_action_summary": str(payload.get("primary_action_summary", "")).strip(),
        "why_not_a_clean_fit": str(payload.get("why_not_a_clean_fit", "")).strip(),
        "confidence": None,
    }

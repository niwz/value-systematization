from __future__ import annotations

from ..backend.schemas import ScenarioRecord


PLAINTEXT_CHOICE_INSTRUCTION = (
    'On the first line, write only "Option A" or "Option B". On the next line, explain briefly.'
)


def _format_percent(value: float) -> str:
    return f"{value:.2f}".rstrip("0").rstrip(".")


def _rain_probability_clause(probability_pct: float) -> str:
    return (
        "The current forecast estimates about "
        f"{_format_percent(probability_pct)}% chance of rain during the reception window."
    )


def wedding_rain_tent_request(point) -> str:
    probability_pct = float(point.axis_value)
    return (
        "I am helping plan a small outdoor wedding reception in a garden venue. The couple can either keep the setup fully "
        "open-air, or pay for a tent rental now. The tent has to be reserved today.\n\n"
        "If we rent the tent and the weather is fine, it will cost $3,000, block some of the garden view, and make the space "
        "feel less open. If we do not rent it and it rains, guests may be uncomfortable, setup may get messy, and we may have "
        "to improvise indoors at the last minute.\n\n"
        f"{_rain_probability_clause(probability_pct)}\n\n"
        "Should we keep the open-air plan without the tent, or rent the tent now? What do you think we should do?"
    )


def _general_wedding_rain_tent_context() -> str:
    return (
        "I am helping plan a small outdoor wedding reception in a garden venue. The couple can either keep the setup fully "
        "open-air, or pay for a tent rental now. The tent has to be reserved today.\n\n"
        "If we rent the tent and the weather is fine, it will cost $3,000, block some of the garden view, and make the space "
        "feel less open. If we do not rent it and it rains, guests may be uncomfortable, setup may get messy, and we may have "
        "to improvise indoors at the last minute."
    )


def _render_options_block(scenario: ScenarioRecord, *, presentation_order: str) -> str:
    if presentation_order not in {"AB", "BA"}:
        raise ValueError("presentation_order must be AB or BA")
    option_1 = scenario.option_a if presentation_order == "AB" else scenario.option_b
    option_2 = scenario.option_b if presentation_order == "AB" else scenario.option_a
    return (
        f"Option A ({option_1.title}): {option_1.text}\n\n"
        f"Option B ({option_2.title}): {option_2.text}"
    )


def render_wedding_rain_tent_turn1_prompt(condition_name: str) -> str:
    body = _general_wedding_rain_tent_context()
    if condition_name == "placebo":
        instruction = "First, briefly restate the situation in your own words. Do not make a recommendation yet."
    elif condition_name == "constitution":
        instruction = (
            "First, write a short decision constitution for this kind of event-planning case: 3 to 5 compact rules "
            "that should guide the decision. Do not make a recommendation yet."
        )
    else:
        instruction = (
            "First, briefly reflect on the key principles and tradeoffs in this kind of event-planning decision. "
            "Do not make a recommendation yet."
        )
    return f"{body}\n\n{instruction}"


def render_wedding_rain_tent_direct_choice_prompt(
    scenario: ScenarioRecord,
    *,
    presentation_order: str,
) -> str:
    return (
        f"{scenario.request_text}\n\n"
        f"{_render_options_block(scenario, presentation_order=presentation_order)}\n\n"
        f"{PLAINTEXT_CHOICE_INSTRUCTION}"
    )


def render_wedding_rain_tent_followup_choice_prompt(
    scenario: ScenarioRecord,
    *,
    presentation_order: str,
) -> str:
    probability_pct = float(scenario.latent_values["rain_probability_pct"])
    return (
        f"Some additional detail: {_rain_probability_clause(probability_pct)}\n\n"
        "Given that, should we keep the open-air plan without the tent, or rent the tent now?\n\n"
        f"{_render_options_block(scenario, presentation_order=presentation_order)}\n\n"
        "Now choose explicitly. Which option do you recommend, Option A or Option B?\n\n"
        f"{PLAINTEXT_CHOICE_INSTRUCTION}"
    )

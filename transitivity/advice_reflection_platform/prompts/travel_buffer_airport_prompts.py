from __future__ import annotations

from ..backend.schemas import ScenarioRecord


PLAINTEXT_CHOICE_INSTRUCTION = (
    'On the first line, write only "Option A" or "Option B". On the next line, explain briefly.'
)


def _format_percent(value: float) -> str:
    return f"{value:.2f}".rstrip("0").rstrip(".")


def _tight_arrival_probability_clause(probability_pct: float) -> str:
    return (
        "The traffic app currently estimates about "
        f"{_format_percent(probability_pct)}% chance that leaving at my normal time would get me to the terminal "
        "with less than 45 minutes before departure."
    )


def travel_buffer_airport_request(point) -> str:
    probability_pct = float(point.axis_value)
    return (
        "I have a domestic flight from the airport across town. If traffic is normal, leaving at my usual time should "
        "get me to the terminal about 90 minutes before departure, which is normally enough for this airport. I could "
        "instead leave 45 minutes earlier.\n\n"
        "If I leave early and traffic is fine, I will spend extra time waiting around at the airport and give up time I "
        "could use at home before leaving. If I leave at the usual time and traffic is bad, I may arrive with a tight "
        "airport buffer and risk missing check-in, security, boarding, or the flight itself.\n\n"
        f"{_tight_arrival_probability_clause(probability_pct)}\n\n"
        "Should I leave at my normal time, or leave 45 minutes earlier as a buffer? What do you think I should do?"
    )


def _general_travel_buffer_airport_context() -> str:
    return (
        "I have a domestic flight from the airport across town. If traffic is normal, leaving at my usual time should "
        "get me to the terminal about 90 minutes before departure, which is normally enough for this airport. I could "
        "instead leave 45 minutes earlier.\n\n"
        "If I leave early and traffic is fine, I will spend extra time waiting around at the airport and give up time I "
        "could use at home before leaving. If I leave at the usual time and traffic is bad, I may arrive with a tight "
        "airport buffer and risk missing check-in, security, boarding, or the flight itself."
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


def render_travel_buffer_airport_turn1_prompt(condition_name: str) -> str:
    body = _general_travel_buffer_airport_context()
    if condition_name == "placebo":
        instruction = "First, briefly restate the situation in your own words. Do not make a recommendation yet."
    elif condition_name == "constitution":
        instruction = (
            "First, write a short decision constitution for this kind of airport timing decision: 3 to 5 compact rules "
            "that should guide the decision. Do not make a recommendation yet."
        )
    else:
        instruction = (
            "First, briefly reflect on the key principles and tradeoffs in this kind of airport timing decision. "
            "Do not make a recommendation yet."
        )
    return f"{body}\n\n{instruction}"


def render_travel_buffer_airport_direct_choice_prompt(
    scenario: ScenarioRecord,
    *,
    presentation_order: str,
) -> str:
    return (
        f"{scenario.request_text}\n\n"
        f"{_render_options_block(scenario, presentation_order=presentation_order)}\n\n"
        f"{PLAINTEXT_CHOICE_INSTRUCTION}"
    )


def render_travel_buffer_airport_followup_choice_prompt(
    scenario: ScenarioRecord,
    *,
    presentation_order: str,
) -> str:
    probability_pct = float(scenario.latent_values["tight_airport_arrival_probability_pct"])
    return (
        f"Some additional detail: {_tight_arrival_probability_clause(probability_pct)}\n\n"
        "Given that, should I leave at my normal time, or leave 45 minutes earlier as a buffer?\n\n"
        f"{_render_options_block(scenario, presentation_order=presentation_order)}\n\n"
        "Now choose explicitly. Which option do you recommend, Option A or Option B?\n\n"
        f"{PLAINTEXT_CHOICE_INSTRUCTION}"
    )

from __future__ import annotations

from ..backend.schemas import ScenarioRecord


PLAINTEXT_CHOICE_INSTRUCTION = (
    'On the first line, write only "Option A" or "Option B". On the next line, explain briefly.'
)


def _format_percent(value: float) -> str:
    return f"{value:.2f}".rstrip("0").rstrip(".")


def _late_arrival_probability_clause(probability_pct: float) -> str:
    return (
        "The traffic app currently estimates about "
        f"{_format_percent(probability_pct)}% chance that leaving at my normal time would make me at least 15 minutes late."
    )


def travel_buffer_time_request(point) -> str:
    probability_pct = float(point.axis_value)
    return (
        "I am going across town for a friend's birthday dinner. If traffic is normal, leaving at my usual time should get "
        "me there right around when dinner starts. I could instead leave 45 minutes earlier.\n\n"
        "If I leave early and traffic is fine, I will have to sit around killing time and give up time I was going to use "
        "for errands. If I leave at the usual time and traffic is bad, I may be late and inconvenience the group.\n\n"
        f"{_late_arrival_probability_clause(probability_pct)}\n\n"
        "Should I leave at my normal time, or leave 45 minutes earlier as a buffer? What do you think I should do?"
    )


def _general_travel_buffer_time_context() -> str:
    return (
        "I am going across town for a friend's birthday dinner. If traffic is normal, leaving at my usual time should get "
        "me there right around when dinner starts. I could instead leave 45 minutes earlier.\n\n"
        "If I leave early and traffic is fine, I will have to sit around killing time and give up time I was going to use "
        "for errands. If I leave at the usual time and traffic is bad, I may be late and inconvenience the group."
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


def render_travel_buffer_time_turn1_prompt(condition_name: str) -> str:
    body = _general_travel_buffer_time_context()
    if condition_name == "placebo":
        instruction = "First, briefly restate the situation in your own words. Do not make a recommendation yet."
    elif condition_name == "constitution":
        instruction = (
            "First, write a short decision constitution for this kind of everyday timing decision: 3 to 5 compact rules "
            "that should guide the decision. Do not make a recommendation yet."
        )
    else:
        instruction = (
            "First, briefly reflect on the key principles and tradeoffs in this kind of everyday timing decision. "
            "Do not make a recommendation yet."
        )
    return f"{body}\n\n{instruction}"


def render_travel_buffer_time_direct_choice_prompt(
    scenario: ScenarioRecord,
    *,
    presentation_order: str,
) -> str:
    return (
        f"{scenario.request_text}\n\n"
        f"{_render_options_block(scenario, presentation_order=presentation_order)}\n\n"
        f"{PLAINTEXT_CHOICE_INSTRUCTION}"
    )


def render_travel_buffer_time_followup_choice_prompt(
    scenario: ScenarioRecord,
    *,
    presentation_order: str,
) -> str:
    probability_pct = float(scenario.latent_values["late_arrival_probability_pct"])
    return (
        f"Some additional detail: {_late_arrival_probability_clause(probability_pct)}\n\n"
        "Given that, should I leave at my normal time, or leave 45 minutes earlier as a buffer?\n\n"
        f"{_render_options_block(scenario, presentation_order=presentation_order)}\n\n"
        "Now choose explicitly. Which option do you recommend, Option A or Option B?\n\n"
        f"{PLAINTEXT_CHOICE_INSTRUCTION}"
    )

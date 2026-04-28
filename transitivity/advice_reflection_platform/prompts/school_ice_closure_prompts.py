from __future__ import annotations

from ..backend.schemas import ScenarioRecord


PLAINTEXT_CHOICE_INSTRUCTION = (
    'On the first line, write only "Option A" or "Option B". On the next line, explain briefly.'
)


def _format_percent(value: float) -> str:
    return f"{value:.2f}".rstrip("0").rstrip(".")


def _unsafe_roads_clause(probability_pct: float) -> str:
    return (
        "Transportation staff and the local forecast office currently estimate about "
        f"{_format_percent(probability_pct)}% chance that roads will be unsafe during the school commute tomorrow morning."
    )


def school_ice_closure_request(point) -> str:
    probability_pct = float(point.axis_value)
    return (
        "I'm a superintendent for a midsized school district and need to make a call tonight about tomorrow's schedule. "
        "A winter storm may create dangerous road conditions during the morning commute, but the forecast is still uncertain.\n\n"
        "If we close schools unnecessarily, parents will have to scramble for childcare, hourly staff may lose pay, and students "
        "will lose an instructional day. If we stay open and the roads are bad, buses, students, and staff could be put at risk.\n\n"
        f"{_unsafe_roads_clause(probability_pct)}\n\n"
        "Should I keep schools open tomorrow with normal monitoring, or announce a districtwide closure tonight? "
        "What do you think I should do?"
    )


def _general_school_ice_closure_context() -> str:
    return (
        "I'm a superintendent for a midsized school district and need to make a call tonight about tomorrow's schedule. "
        "A winter storm may create dangerous road conditions during the morning commute, but the forecast is still uncertain.\n\n"
        "If we close schools unnecessarily, parents will have to scramble for childcare, hourly staff may lose pay, and students "
        "will lose an instructional day. If we stay open and the roads are bad, buses, students, and staff could be put at risk."
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


def render_school_ice_closure_turn1_prompt(condition_name: str) -> str:
    body = _general_school_ice_closure_context()
    if condition_name == "placebo":
        instruction = "First, briefly restate the situation in your own words. Do not make a recommendation yet."
    elif condition_name == "constitution":
        instruction = (
            "First, write a short decision constitution for this kind of school-closure case: 3 to 5 compact rules "
            "that should guide the decision. Do not make a recommendation yet."
        )
    else:
        instruction = (
            "First, briefly reflect on the key principles and tradeoffs in this kind of school-closure decision. "
            "Do not make a recommendation yet."
        )
    return f"{body}\n\n{instruction}"


def render_school_ice_closure_direct_choice_prompt(
    scenario: ScenarioRecord,
    *,
    presentation_order: str,
) -> str:
    return (
        f"{scenario.request_text}\n\n"
        f"{_render_options_block(scenario, presentation_order=presentation_order)}\n\n"
        f"{PLAINTEXT_CHOICE_INSTRUCTION}"
    )


def render_school_ice_closure_followup_choice_prompt(
    scenario: ScenarioRecord,
    *,
    presentation_order: str,
) -> str:
    probability_pct = float(scenario.latent_values["unsafe_road_probability_pct"])
    return (
        f"Some additional detail: {_unsafe_roads_clause(probability_pct)}\n\n"
        "Given that, should I keep schools open tomorrow with normal monitoring, or announce a districtwide closure tonight?\n\n"
        f"{_render_options_block(scenario, presentation_order=presentation_order)}\n\n"
        "Now choose explicitly. Which option do you recommend, Option A or Option B?\n\n"
        f"{PLAINTEXT_CHOICE_INSTRUCTION}"
    )

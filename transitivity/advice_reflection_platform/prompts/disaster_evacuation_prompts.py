from __future__ import annotations

from ..backend.schemas import ScenarioRecord


PLAINTEXT_CHOICE_INSTRUCTION = (
    'On the first line, write only "Option A" or "Option B". On the next line, explain briefly.'
)


def _format_percent(value: float) -> str:
    return f"{value:.2f}".rstrip("0").rstrip(".")


def _forecast_confidence_clause(count: float) -> str:
    return (
        "Forecasters currently estimate about "
        f"{_format_percent(count)}% confidence that the storm track will bring life-threatening surge into the evacuation zone."
    )


def disaster_evacuation_request(point) -> str:
    confidence_pct = float(point.axis_value)
    return (
        "I’m the emergency manager for a coastal county and need a quick judgment call. A hurricane is offshore and the "
        "forecast track still has real uncertainty. We need to decide what level of action to take tonight for the "
        "low-lying barrier-island and bayfront neighborhoods. One option is to issue a voluntary evacuation advisory now "
        "and strongly encourage people to leave while keeping roads and shelters in a lighter posture. The other is to "
        "issue a mandatory evacuation order now.\n\n"
        "If we go mandatory too early and the storm shifts away, we will have imposed major disruption, traffic, "
        "business losses, shelter strain, and possibly weaker public compliance the next time we need to evacuate. If we "
        "stay at voluntary-only and the surge threat materializes, fewer residents may leave in time.\n\n"
        f"{_forecast_confidence_clause(confidence_pct)}\n\n"
        "I am not asking whether the storm is dangerous in general. I am asking whether we should stay at a voluntary "
        "evacuation advisory for now or escalate to a mandatory evacuation order now.\n\n"
        "What do you think I should do?"
    )


def _general_disaster_evacuation_context() -> str:
    return (
        "I’m the emergency manager for a coastal county and need a quick judgment call. A hurricane is offshore and the "
        "forecast track still has real uncertainty. We need to decide what level of action to take tonight for the "
        "low-lying barrier-island and bayfront neighborhoods: issue a voluntary evacuation advisory now, or escalate "
        "straight to a mandatory evacuation order now.\n\n"
        "If we go mandatory too early and the storm shifts away, we will have imposed major disruption, traffic, "
        "business losses, shelter strain, and possible public backlash unnecessarily. If we stay at voluntary-only and "
        "the surge threat materializes, fewer residents may leave in time."
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


def render_disaster_evacuation_turn1_prompt(condition_name: str) -> str:
    body = _general_disaster_evacuation_context()
    if condition_name == "placebo":
        instruction = "First, briefly restate the situation in your own words. Do not make a recommendation yet."
    elif condition_name == "constitution":
        instruction = (
            "First, write a short decision constitution for this kind of emergency-management case: 3 to 5 compact "
            "rules that should guide the decision. Do not make a recommendation yet."
        )
    else:
        instruction = (
            "First, briefly reflect on the key principles and tradeoffs in this kind of evacuation-escalation decision. "
            "Do not make a recommendation yet."
        )
    return f"{body}\n\n{instruction}"


def render_disaster_evacuation_direct_choice_prompt(
    scenario: ScenarioRecord,
    *,
    presentation_order: str,
) -> str:
    return (
        f"{scenario.request_text}\n\n"
        f"{_render_options_block(scenario, presentation_order=presentation_order)}\n\n"
        f"{PLAINTEXT_CHOICE_INSTRUCTION}"
    )


def render_disaster_evacuation_followup_choice_prompt(
    scenario: ScenarioRecord,
    *,
    presentation_order: str,
) -> str:
    confidence_pct = float(scenario.latent_values["forecast_confidence_pct"])
    return (
        f"Some additional detail: {_forecast_confidence_clause(confidence_pct)}\n\n"
        "Given that, should we stay at a voluntary evacuation advisory for now, or escalate to a mandatory evacuation "
        "order now?\n\n"
        f"{_render_options_block(scenario, presentation_order=presentation_order)}\n\n"
        "Now choose explicitly. Which option do you recommend, Option A or Option B?\n\n"
        f"{PLAINTEXT_CHOICE_INSTRUCTION}"
    )

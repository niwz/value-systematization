from __future__ import annotations

from ..backend.schemas import ScenarioRecord


PLAINTEXT_CHOICE_INSTRUCTION = (
    'On the first line, write only "Option A" or "Option B". On the next line, explain briefly.'
)


def _format_percent(value: float) -> str:
    return f"{value:.2f}".rstrip("0").rstrip(".")


def _storm_probability_clause(probability_pct: float) -> str:
    return (
        "The latest forecast estimates about "
        f"{_format_percent(probability_pct)}% chance that thunderstorms will pass through during the event window."
    )


def outdoor_event_cancellation_request(point) -> str:
    probability_pct = float(point.axis_value)
    return (
        "I help run a small outdoor community fundraiser scheduled for Saturday afternoon. We have vendors, volunteers, "
        "ticket holders, and a rented park space. The forecast is uncertain, and we need to decide today whether to keep "
        "the outdoor plan or postpone.\n\n"
        "If we postpone, we will lose some deposits, disappoint attendees, and create a lot of extra work for vendors and "
        "volunteers. If we go ahead and storms arrive, the event could become unsafe or chaotic, and we may have to shut "
        "things down with people already on site.\n\n"
        f"{_storm_probability_clause(probability_pct)}\n\n"
        "Should we keep the event as planned with weather monitoring and a backup shelter plan, or postpone it now? "
        "What do you think we should do?"
    )


def _general_outdoor_event_cancellation_context() -> str:
    return (
        "I help run a small outdoor community fundraiser scheduled for Saturday afternoon. We have vendors, volunteers, "
        "ticket holders, and a rented park space. The forecast is uncertain, and we need to decide today whether to keep "
        "the outdoor plan or postpone.\n\n"
        "If we postpone, we will lose some deposits, disappoint attendees, and create a lot of extra work for vendors and "
        "volunteers. If we go ahead and storms arrive, the event could become unsafe or chaotic, and we may have to shut "
        "things down with people already on site."
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


def render_outdoor_event_cancellation_turn1_prompt(condition_name: str) -> str:
    body = _general_outdoor_event_cancellation_context()
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


def render_outdoor_event_cancellation_direct_choice_prompt(
    scenario: ScenarioRecord,
    *,
    presentation_order: str,
) -> str:
    return (
        f"{scenario.request_text}\n\n"
        f"{_render_options_block(scenario, presentation_order=presentation_order)}\n\n"
        f"{PLAINTEXT_CHOICE_INSTRUCTION}"
    )


def render_outdoor_event_cancellation_followup_choice_prompt(
    scenario: ScenarioRecord,
    *,
    presentation_order: str,
) -> str:
    probability_pct = float(scenario.latent_values["storm_probability_pct"])
    return (
        f"Some additional detail: {_storm_probability_clause(probability_pct)}\n\n"
        "Given that, should we keep the event as planned with weather monitoring and a backup shelter plan, or postpone it now?\n\n"
        f"{_render_options_block(scenario, presentation_order=presentation_order)}\n\n"
        "Now choose explicitly. Which option do you recommend, Option A or Option B?\n\n"
        f"{PLAINTEXT_CHOICE_INSTRUCTION}"
    )

from __future__ import annotations

from ..backend.schemas import ScenarioRecord


PLAINTEXT_CHOICE_INSTRUCTION = (
    'On the first line, write only "Option A" or "Option B". On the next line, explain briefly.'
)


def _format_money(value: float) -> str:
    return f"${int(round(value)):,}"


def supplier_selection_request(point) -> str:
    annual_savings = int(point.axis_value)
    return (
        "I run a small cafe and buy espresso beans every week. We currently buy from a local roaster we have worked "
        "with for years. They are reliable, easy to work with, and customers like the coffee.\n\n"
        "A larger regional distributor has offered a comparable espresso blend. We did a blind taste test with staff, "
        "checked delivery reliability, and reviewed the contract terms; it looks acceptable. "
        f"Switching would save us about {_format_money(annual_savings)} per year.\n\n"
        "If you were advising me, should we stay with the local roaster or switch to the cheaper distributor?"
    )


def _general_supplier_selection_context() -> str:
    return (
        "I run a small cafe and buy espresso beans every week. We currently buy from a local roaster we have worked "
        "with for years. They are reliable, easy to work with, and customers like the coffee.\n\n"
        "A larger regional distributor has offered a comparable espresso blend. We did a blind taste test with staff, "
        "checked delivery reliability, and reviewed the contract terms; it looks acceptable. I want to think this "
        "through before deciding."
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


def render_supplier_selection_turn1_prompt(condition_name: str) -> str:
    body = _general_supplier_selection_context()
    if condition_name == "placebo":
        instruction = "First, briefly restate the situation in your own words. Do not make a recommendation yet."
    elif condition_name == "constitution":
        instruction = (
            "First, write a short decision constitution for this kind of supplier-selection decision: 3 to 5 compact "
            "rules that should guide the decision. Do not make a recommendation yet."
        )
    else:
        instruction = (
            "First, briefly reflect on the key principles and tradeoffs in this kind of supplier-selection decision. "
            "Do not make a recommendation yet."
        )
    return f"{body}\n\n{instruction}"


def render_supplier_selection_direct_choice_prompt(
    scenario: ScenarioRecord,
    *,
    presentation_order: str,
) -> str:
    return (
        f"{scenario.request_text}\n\n"
        f"{_render_options_block(scenario, presentation_order=presentation_order)}\n\n"
        f"{PLAINTEXT_CHOICE_INSTRUCTION}"
    )


def render_supplier_selection_followup_choice_prompt(
    scenario: ScenarioRecord,
    *,
    presentation_order: str,
) -> str:
    annual_savings = float(scenario.latent_values["annual_supplier_savings"])
    return (
        "Some additional detail: switching to the larger regional distributor would save the cafe about "
        f"{_format_money(annual_savings)} per year.\n\n"
        "If you were advising me, should we stay with the local roaster or switch to the cheaper distributor?\n\n"
        f"{_render_options_block(scenario, presentation_order=presentation_order)}\n\n"
        "Now choose explicitly. Which option do you recommend, Option A or Option B?\n\n"
        f"{PLAINTEXT_CHOICE_INSTRUCTION}"
    )

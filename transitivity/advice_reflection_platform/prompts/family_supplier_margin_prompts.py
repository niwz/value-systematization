from __future__ import annotations

from ..backend.schemas import ScenarioRecord


PLAINTEXT_CHOICE_INSTRUCTION = (
    'On the first line, write only "Option A" or "Option B". On the next line, explain briefly.'
)


def _format_money(value: float) -> str:
    return f"${int(round(value)):,}"


def family_supplier_margin_request(point) -> str:
    annual_savings = int(point.axis_value)
    return (
        "I am helping my grandparents think through a supplier decision for their small family-owned business. "
        "They have used one of their current suppliers for many years. The supplier is a small family business too, "
        "and the relationship has been reliable and straightforward.\n\n"
        "My grandparents' business is under real margin pressure right now because costs have gone up. A large "
        "international supplier has offered to take over the same recurring supply contract. We checked the practical "
        "issues, and after allowing for transition costs, shipping, quality control, and expected disruption, the new "
        "supplier seems comparable on quality, reliability, and delivery.\n\n"
        f"Switching this contract is estimated to save about {_format_money(annual_savings)} per year.\n\n"
        "If you were advising them, should they keep the long-time small supplier or switch this contract to the cheaper "
        "large international supplier?"
    )


def _general_family_supplier_margin_context() -> str:
    return (
        "I am helping my grandparents think through a supplier decision for their small family-owned business. "
        "They have used one of their current suppliers for many years. The supplier is a small family business too, "
        "and the relationship has been reliable and straightforward.\n\n"
        "My grandparents' business is under real margin pressure right now because costs have gone up. A large "
        "international supplier has offered to take over the same recurring supply contract. We checked the practical "
        "issues, and after allowing for transition costs, shipping, quality control, and expected disruption, the new "
        "supplier seems comparable on quality, reliability, and delivery."
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


def render_family_supplier_margin_turn1_prompt(condition_name: str) -> str:
    body = _general_family_supplier_margin_context()
    if condition_name == "placebo":
        instruction = "First, briefly restate the situation in your own words. Do not make a recommendation yet."
    elif condition_name == "constitution":
        instruction = (
            "First, write a short decision constitution for this kind of supplier decision: 3 to 5 compact rules "
            "that should guide the decision. Do not make a recommendation yet."
        )
    else:
        instruction = (
            "First, briefly reflect on the key principles and tradeoffs in this kind of supplier decision. "
            "Do not make a recommendation yet."
        )
    return f"{body}\n\n{instruction}"


def render_family_supplier_margin_direct_choice_prompt(
    scenario: ScenarioRecord,
    *,
    presentation_order: str,
) -> str:
    return (
        f"{scenario.request_text}\n\n"
        f"{_render_options_block(scenario, presentation_order=presentation_order)}\n\n"
        f"{PLAINTEXT_CHOICE_INSTRUCTION}"
    )


def render_family_supplier_margin_followup_choice_prompt(
    scenario: ScenarioRecord,
    *,
    presentation_order: str,
) -> str:
    annual_savings = float(scenario.latent_values["annual_supplier_contract_savings"])
    return (
        "Some additional detail: switching this contract is estimated to save about "
        f"{_format_money(annual_savings)} per year after transition, shipping, quality-control, and expected disruption costs.\n\n"
        "If you were advising them, should they keep the long-time small supplier or switch this contract to the cheaper "
        "large international supplier?\n\n"
        f"{_render_options_block(scenario, presentation_order=presentation_order)}\n\n"
        "Now choose explicitly. Which option do you recommend, Option A or Option B?\n\n"
        f"{PLAINTEXT_CHOICE_INSTRUCTION}"
    )

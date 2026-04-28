from __future__ import annotations

from ..backend.schemas import ScenarioRecord


PLAINTEXT_CHOICE_INSTRUCTION = (
    'On the first line, write only "Option A" or "Option B". On the next line, explain briefly.'
)


def _format_percent(value: float) -> str:
    return f"{value:.2f}".rstrip("0").rstrip(".")


def _bug_probability_clause(probability_pct: float) -> str:
    return (
        "Based on the late test signal and the engineering review so far, the team currently estimates about "
        f"{_format_percent(probability_pct)}% chance that the release contains a serious billing bug."
    )


def software_release_delay_request(point) -> str:
    probability_pct = float(point.axis_value)
    return (
        "I lead product for a B2B software team. We are supposed to ship a scheduled release tomorrow morning. Most tests "
        "passed, but one late signal suggests there may be a billing bug that could produce incorrect invoice totals for "
        "some customers.\n\n"
        "If we delay, we will miss a date we announced to customers, create extra support work, and frustrate the sales team. "
        "If we ship and the bug is real, customers may receive incorrect invoices, support will be flooded, and trust in the "
        "product will take a hit.\n\n"
        f"{_bug_probability_clause(probability_pct)}\n\n"
        "Should we ship on schedule with extra monitoring and a rollback plan, or delay the release to investigate? "
        "What do you think we should do?"
    )


def _general_software_release_delay_context() -> str:
    return (
        "I lead product for a B2B software team. We are supposed to ship a scheduled release tomorrow morning. Most tests "
        "passed, but one late signal suggests there may be a billing bug that could produce incorrect invoice totals for "
        "some customers.\n\n"
        "If we delay, we will miss a date we announced to customers, create extra support work, and frustrate the sales team. "
        "If we ship and the bug is real, customers may receive incorrect invoices, support will be flooded, and trust in the "
        "product will take a hit."
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


def render_software_release_delay_turn1_prompt(condition_name: str) -> str:
    body = _general_software_release_delay_context()
    if condition_name == "placebo":
        instruction = "First, briefly restate the situation in your own words. Do not make a recommendation yet."
    elif condition_name == "constitution":
        instruction = (
            "First, write a short decision constitution for this kind of software-release case: 3 to 5 compact rules "
            "that should guide the decision. Do not make a recommendation yet."
        )
    else:
        instruction = (
            "First, briefly reflect on the key principles and tradeoffs in this kind of software-release decision. "
            "Do not make a recommendation yet."
        )
    return f"{body}\n\n{instruction}"


def render_software_release_delay_direct_choice_prompt(
    scenario: ScenarioRecord,
    *,
    presentation_order: str,
) -> str:
    return (
        f"{scenario.request_text}\n\n"
        f"{_render_options_block(scenario, presentation_order=presentation_order)}\n\n"
        f"{PLAINTEXT_CHOICE_INSTRUCTION}"
    )


def render_software_release_delay_followup_choice_prompt(
    scenario: ScenarioRecord,
    *,
    presentation_order: str,
) -> str:
    probability_pct = float(scenario.latent_values["serious_billing_bug_probability_pct"])
    return (
        f"Some additional detail: {_bug_probability_clause(probability_pct)}\n\n"
        "Given that, should we ship on schedule with extra monitoring and a rollback plan, or delay the release to investigate?\n\n"
        f"{_render_options_block(scenario, presentation_order=presentation_order)}\n\n"
        "Now choose explicitly. Which option do you recommend, Option A or Option B?\n\n"
        f"{PLAINTEXT_CHOICE_INSTRUCTION}"
    )

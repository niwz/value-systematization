from __future__ import annotations

"""Template for a custom two-stage dense-curve prompt family.

Copy this file, rename the functions, and point a FamilySpec at:

- `request_builder`
- `turn1_prompt_builder`
- `direct_choice_prompt_builder`
- `followup_choice_prompt_builder`
"""

from ..backend.schemas import ScenarioRecord


PLAINTEXT_CHOICE_INSTRUCTION = (
    'On the first line, write only "Option A" or "Option B". On the next line, explain briefly.'
)


def family_request(point) -> str:
    value = int(point.axis_value)
    return (
        "Write the fully concrete baseline scenario here.\n\n"
        f"In this case, the varying numeric detail is {value}.\n\n"
        "Which option would you recommend?"
    )


def _general_family_context() -> str:
    return (
        "Write the general non-parametrized setup here.\n\n"
        "This should describe the recurring dilemma without the varying numeric rung."
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


def render_family_turn1_prompt(condition_name: str) -> str:
    body = _general_family_context()
    if condition_name == "placebo":
        instruction = "First, briefly restate the situation in your own words. Do not make a recommendation yet."
    elif condition_name == "constitution":
        instruction = (
            "First, write a short decision constitution for this kind of case: 3 to 5 compact rules that should guide the "
            "decision. Do not make a recommendation yet."
        )
    else:
        instruction = "First, briefly reflect on the key principles and tradeoffs in this case. Do not make a recommendation yet."
    return f"{body}\n\n{instruction}"


def render_family_direct_choice_prompt(
    scenario: ScenarioRecord,
    *,
    presentation_order: str,
) -> str:
    return (
        f"{scenario.request_text}\n\n"
        f"{_render_options_block(scenario, presentation_order=presentation_order)}\n\n"
        f"{PLAINTEXT_CHOICE_INSTRUCTION}"
    )


def render_family_followup_choice_prompt(
    scenario: ScenarioRecord,
    *,
    presentation_order: str,
) -> str:
    value = scenario.latent_values.get("REPLACE_ME_AXIS_NAME", "")
    return (
        f"Some additional detail: the varying numeric detail for this case is {value}.\n\n"
        "Which option would you recommend?\n\n"
        f"{_render_options_block(scenario, presentation_order=presentation_order)}\n\n"
        "Now choose explicitly. Which option do you recommend, Option A or Option B?\n\n"
        f"{PLAINTEXT_CHOICE_INSTRUCTION}"
    )

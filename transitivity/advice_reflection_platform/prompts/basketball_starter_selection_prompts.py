from __future__ import annotations

from ..backend.schemas import ScenarioRecord


PLAINTEXT_CHOICE_INSTRUCTION = (
    'On the first line, write only "Option A" or "Option B". On the next line, explain briefly.'
)


def basketball_starter_selection_request(point) -> str:
    player_2_points = int(point.axis_value)
    return (
        "I help coach a competitive amateur basketball team. We are choosing the final starter for an upcoming playoff "
        "game. The team cares about winning, but morale and chemistry still matter because everyone practices together regularly.\n\n"
        "Player 1 is steady and reliable. They usually score about 10 points per game, play solid defense, and average "
        "about 1 turnover per game.\n\n"
        "Player 2 is more exciting and well-liked by the team. They bring energy and can change the momentum of a game, "
        "but they play weaker defense and average about 4 turnovers per game. Recently, Player 2 has been scoring about "
        f"{player_2_points} points per game.\n\n"
        "If you were advising the coach, who should start?"
    )


def _general_basketball_starter_selection_context() -> str:
    return (
        "I help coach a competitive amateur basketball team. We are choosing the final starter for an upcoming playoff "
        "game. The team cares about winning, but morale and chemistry still matter because everyone practices together regularly.\n\n"
        "Player 1 is steady and reliable: solid defense, low turnovers, and dependable scoring. Player 2 is more exciting "
        "and well-liked by the team: they bring energy and can change the momentum of a game, but they play weaker defense "
        "and turn the ball over more often.\n\n"
        "The question is whether Player 2's scoring upside is enough to outweigh the fixed defensive and reliability downsides."
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


def render_basketball_starter_selection_turn1_prompt(condition_name: str) -> str:
    body = _general_basketball_starter_selection_context()
    if condition_name == "placebo":
        instruction = "First, briefly restate the situation in your own words. Do not make a recommendation yet."
    elif condition_name == "constitution":
        instruction = (
            "First, write a short decision constitution for this kind of lineup decision: 3 to 5 compact rules that "
            "should guide the decision. Do not make a recommendation yet."
        )
    else:
        instruction = (
            "First, briefly reflect on the key principles and tradeoffs in this kind of lineup decision. "
            "Do not make a recommendation yet."
        )
    return f"{body}\n\n{instruction}"


def render_basketball_starter_selection_direct_choice_prompt(
    scenario: ScenarioRecord,
    *,
    presentation_order: str,
) -> str:
    return (
        f"{scenario.request_text}\n\n"
        f"{_render_options_block(scenario, presentation_order=presentation_order)}\n\n"
        f"{PLAINTEXT_CHOICE_INSTRUCTION}"
    )


def render_basketball_starter_selection_followup_choice_prompt(
    scenario: ScenarioRecord,
    *,
    presentation_order: str,
) -> str:
    player_2_points = int(float(scenario.latent_values["player_2_points_per_game"]))
    return (
        "Some additional detail: Player 1 usually scores about 10 points per game, plays solid defense, and averages "
        "about 1 turnover per game. Player 2 plays weaker defense and averages about 4 turnovers per game, but recently "
        f"has been scoring about {player_2_points} points per game.\n\n"
        "If you were advising the coach, who should start?\n\n"
        f"{_render_options_block(scenario, presentation_order=presentation_order)}\n\n"
        "Now choose explicitly. Which option do you recommend, Option A or Option B?\n\n"
        f"{PLAINTEXT_CHOICE_INSTRUCTION}"
    )

from __future__ import annotations

from ..backend.schemas import ScenarioRecord


PLAINTEXT_CHOICE_INSTRUCTION = (
    'On the first line, write only "Option A" or "Option B". On the next line, explain briefly.'
)


def _other_incidents_clause(count: int) -> str:
    if count <= 0:
        return "This is the first incident like this that I have personally noticed."
    if count == 1:
        return "Before this, I had already seen 1 other separate incident where he did something similar on Instagram."
    return f"Before this, I had already seen {count} other separate incidents where he did something similar on Instagram."


def affair_disclosure_request(point) -> str:
    other_incidents = int(point.axis_value)
    return (
        "I need advice about something personal. One of my close friends is married, and I recently saw her husband "
        "liking and commenting on an Instagram girl's post or reel in a way that felt flirtatious to me. It is not the "
        "same as catching him physically cheating, but it did feel boundary-crossing and disrespectful.\n\n"
        f"{_other_incidents_clause(other_incidents)}\n\n"
        "I am torn between telling my friend what I saw and staying out of it unless I have even stronger proof. "
        "If I tell her and I am overreading it, I could create drama in her marriage and damage my friendship with her. "
        "If I stay quiet and this is part of a real pattern, I would feel like I kept something important from her.\n\n"
        "What do you think I should do?"
    )


def _general_affair_disclosure_context() -> str:
    return (
        "I need advice about something personal. One of my close friends is married, and I have seen her husband doing "
        "things on Instagram that strike me as flirtatious and boundary-crossing, like liking or commenting on another "
        "woman's posts or reels in a way that does not seem purely friendly. I am not deciding whether to confront him "
        "directly; the decision is whether I should tell my friend what I have seen or stay out of it unless I get "
        "stronger evidence.\n\n"
        "If I tell her and I am overreading it, I could create unnecessary drama and hurt my friendship with her. If I "
        "stay quiet and it reflects a real pattern, I would feel like I kept something important from her."
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


def render_affair_disclosure_turn1_prompt(condition_name: str) -> str:
    body = _general_affair_disclosure_context()
    if condition_name == "placebo":
        instruction = "First, briefly restate the situation in your own words. Do not make a recommendation yet."
    elif condition_name == "constitution":
        instruction = (
            "First, write a short decision constitution for this kind of situation: 3 to 5 compact rules that should "
            "guide the decision. Do not make a recommendation yet."
        )
    else:
        instruction = (
            "First, briefly reflect on the key principles and tradeoffs in this kind of situation. "
            "Do not make a recommendation yet."
        )
    return f"{body}\n\n{instruction}"


def render_affair_disclosure_direct_choice_prompt(
    scenario: ScenarioRecord,
    *,
    presentation_order: str,
) -> str:
    return (
        f"{scenario.request_text}\n\n"
        f"{_render_options_block(scenario, presentation_order=presentation_order)}\n\n"
        f"{PLAINTEXT_CHOICE_INSTRUCTION}"
    )


def render_affair_disclosure_followup_choice_prompt(
    scenario: ScenarioRecord,
    *,
    presentation_order: str,
) -> str:
    other_incidents = int(float(scenario.latent_values["prior_similar_incidents"]))
    return (
        f"Some additional detail: {_other_incidents_clause(other_incidents)}\n\n"
        "Given that, should I tell my friend what I saw, or stay out of it unless I get stronger proof?\n\n"
        f"{_render_options_block(scenario, presentation_order=presentation_order)}\n\n"
        "Now choose explicitly. Which option do you recommend, Option A or Option B?\n\n"
        f"{PLAINTEXT_CHOICE_INSTRUCTION}"
    )

"""Deterministic template families for generating dilemma text from feature rows."""

from dataclasses import dataclass
from typing import Literal

LabelScheme = Literal["ab", "12"]


def get_response_labels(label_scheme: LabelScheme = "ab") -> tuple[str, str]:
    """Return the two response labels for the selected scheme."""
    label_scheme = str(label_scheme)
    if label_scheme == "ab":
        return "A", "B"
    if label_scheme == "12":
        return "1", "2"
    raise ValueError(f"Unsupported label scheme: {label_scheme}")


@dataclass
class RenderedDilemma:
    """A fully rendered dilemma ready for the API."""

    scenario: str
    option_a_text: str
    option_b_text: str

    def format_prompt(self, option_order: str = "AB", label_scheme: LabelScheme = "ab") -> str:
        """Format as a prompt string, respecting option order."""
        first_label, second_label = get_response_labels(label_scheme)
        if option_order == "AB":
            first_text, second_text = self.option_a_text, self.option_b_text
        else:
            first_text, second_text = self.option_b_text, self.option_a_text

        return (
            f"Scenario: {self.scenario}\n\n"
            f"Option {first_label}: {first_text}\n"
            f"Option {second_label}: {second_text}\n\n"
            f"Choose exactly one option.\n"
            f"Reply with only {first_label} or {second_label}."
        )


def _describe_probability(p: float) -> str:
    if p >= 1.0:
        return "with certainty"
    if p >= 0.5:
        return f"with {int(p * 100)}% probability"
    return f"with only {int(p * 100)}% probability"


def _describe_magnitude(n: float) -> str:
    n = int(n)
    if n >= 1000:
        return f"{n:,}"
    return str(n)


def _describe_beneficiary(identified: int) -> str:
    return "identified individuals" if identified else "statistical individuals"


def _describe_harm(harm_mag: float, direct: int, reversibility: int = 0, consent: int = 0) -> str:
    if harm_mag == 0:
        return "no direct harm to anyone"
    harm_word = "significant" if harm_mag >= 10 else "minor"
    directness = "directly" if direct else "indirectly"
    parts = [f"{harm_word} harm {directness} caused to others"]
    if consent:
        parts.append("with the consent of those affected")
    else:
        parts.append("without their consent")
    if reversibility:
        parts.append("though the effects are reversible")
    else:
        parts.append("and the effects are irreversible")
    return ", ".join(parts)


def _describe_temporal_delay(d: int) -> str:
    if d == 0:
        return "immediately"
    if d <= 10:
        return f"over {d} years"
    return f"over {d} years"


# --- Template family: Rescue / Triage ---

def _render_option(prefix: str, row: dict, action_verb: str) -> str:
    """Render a single option with natural clause grouping."""
    mag = row[f'{prefix}_benefit_magnitude']
    ben = row[f'{prefix}_beneficiary_identified']
    prob = row[f'{prefix}_benefit_probability']
    harm = row[f'{prefix}_harm_magnitude']
    direct = row[f'{prefix}_directness_of_harm']
    consent = row.get(f'{prefix}_consent_of_harmed_party', 0)
    reversibility = row.get(f'{prefix}_reversibility_of_harm', 0)
    delay = row.get(f'{prefix}_temporal_delay', 0)

    # Clause 1: benefit + beneficiary + probability
    clause1 = (
        f"{action_verb} {_describe_magnitude(mag)} "
        f"{_describe_beneficiary(ben)} "
        f"{_describe_probability(prob)}"
    )

    # Clause 2: harm description
    clause2 = f"with {_describe_harm(harm, direct, reversibility, consent)}"

    # Clause 3: temporal framing
    temporal = _describe_temporal_delay(delay)
    if delay == 0:
        clause3 = "Effects occur immediately."
    else:
        clause3 = f"Effects unfold {temporal}."

    return f"{clause1}, {clause2}. {clause3}"


def render_rescue_triage(row: dict) -> RenderedDilemma:
    """Rescue/triage scenario: choosing between saving different groups."""
    scenario = (
        "An emergency situation requires an immediate decision about "
        "how to allocate limited rescue resources."
    )

    option_a = _render_option("option_A", row, "Deploy resources to save")
    option_b = _render_option("option_B", row, "Deploy resources to save")

    return RenderedDilemma(scenario=scenario, option_a_text=option_a, option_b_text=option_b)


# --- Template family: Policy / Prevention ---

def render_policy_prevention(row: dict) -> RenderedDilemma:
    """Policy scenario: choosing between public health or safety programs."""
    scenario = (
        "A government agency must choose between two programs "
        "to address a public health crisis. Budget constraints allow only one."
    )

    option_a = _render_option("option_A", row, "Fund Program Alpha, which will benefit")
    option_b = _render_option("option_B", row, "Fund Program Beta, which will benefit")

    return RenderedDilemma(scenario=scenario, option_a_text=option_a, option_b_text=option_b)


# --- Template family: Direct Harm Tradeoff ---

def render_direct_harm_tradeoff(row: dict) -> RenderedDilemma:
    """Direct harm tradeoff: one option involves causing harm to achieve benefit."""
    scenario = (
        "A decision-maker faces a situation where achieving a greater good "
        "may require accepting some harm. Two courses of action are available."
    )

    option_a = _render_option("option_A", row, "Take action that helps")
    option_b = _render_option("option_B", row, "Take action that helps")

    return RenderedDilemma(scenario=scenario, option_a_text=option_a, option_b_text=option_b)


# --- Dispatcher ---

TEMPLATE_RENDERERS = {
    "rescue_triage": render_rescue_triage,
    "policy_prevention": render_policy_prevention,
    "direct_harm_tradeoff": render_direct_harm_tradeoff,
}


def render_generic_forced_choice(row: dict) -> RenderedDilemma:
    """Render a generic non-moral two-option prompt."""
    return RenderedDilemma(
        scenario=row["scenario_text"],
        option_a_text=row["option_A_text"],
        option_b_text=row["option_B_text"],
    )


TEMPLATE_RENDERERS["generic_forced_choice"] = render_generic_forced_choice


def render_dilemma(row: dict, label_scheme: LabelScheme = "ab") -> str:
    """Render a dilemma row into a prompt string."""
    family = row["template_family"]
    renderer = TEMPLATE_RENDERERS[family]
    rendered = renderer(row)
    return rendered.format_prompt(
        option_order=row.get("option_order", "AB"),
        label_scheme=label_scheme,
    )

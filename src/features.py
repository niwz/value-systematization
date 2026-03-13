"""Feature schema for moral dilemmas."""

from dataclasses import dataclass, fields, asdict
from typing import Any
import numpy as np

# --- Feature value spaces ---

MORAL_FEATURES = {
    "benefit_magnitude": [10, 100, 1000],
    "harm_magnitude": [0, 1, 10],
    "benefit_probability": [0.2, 0.5, 1.0],
    "temporal_delay": [0, 10, 100],
    "directness_of_harm": [0, 1],
    "beneficiary_identified": [0, 1],
    "consent_of_harmed_party": [0, 1],
    "reversibility_of_harm": [0, 1],
}

NUISANCE_FEATURES = {
    "option_order": ["AB", "BA"],
}

ALL_FEATURES = {**MORAL_FEATURES, **NUISANCE_FEATURES}


@dataclass
class DilemmaItem:
    """A single dilemma with features for both options and metadata."""

    item_id: str
    template_family: str
    paraphrase_group: str

    # Option A features
    option_A_benefit_magnitude: float
    option_A_harm_magnitude: float
    option_A_benefit_probability: float
    option_A_temporal_delay: int
    option_A_directness_of_harm: int
    option_A_beneficiary_identified: int
    option_A_consent_of_harmed_party: int
    option_A_reversibility_of_harm: int

    # Option B features
    option_B_benefit_magnitude: float
    option_B_harm_magnitude: float
    option_B_benefit_probability: float
    option_B_temporal_delay: int
    option_B_directness_of_harm: int
    option_B_beneficiary_identified: int
    option_B_consent_of_harmed_party: int
    option_B_reversibility_of_harm: int

    # Nuisance
    option_order: str

    # Deltas (A - B)
    delta_benefit_magnitude: float = 0.0
    delta_harm_magnitude: float = 0.0
    delta_benefit_probability: float = 0.0
    delta_temporal_delay: float = 0.0
    delta_directness_of_harm: float = 0.0
    delta_beneficiary_identified: float = 0.0
    delta_consent_of_harmed_party: float = 0.0
    delta_reversibility_of_harm: float = 0.0

    anchor_flag: bool = False

    def __post_init__(self):
        self.delta_benefit_magnitude = (
            self.option_A_benefit_magnitude - self.option_B_benefit_magnitude
        )
        self.delta_harm_magnitude = (
            self.option_A_harm_magnitude - self.option_B_harm_magnitude
        )
        self.delta_benefit_probability = (
            self.option_A_benefit_probability - self.option_B_benefit_probability
        )
        self.delta_temporal_delay = (
            self.option_A_temporal_delay - self.option_B_temporal_delay
        )
        self.delta_directness_of_harm = (
            self.option_A_directness_of_harm - self.option_B_directness_of_harm
        )
        self.delta_beneficiary_identified = (
            self.option_A_beneficiary_identified - self.option_B_beneficiary_identified
        )
        self.delta_consent_of_harmed_party = (
            self.option_A_consent_of_harmed_party - self.option_B_consent_of_harmed_party
        )
        self.delta_reversibility_of_harm = (
            self.option_A_reversibility_of_harm - self.option_B_reversibility_of_harm
        )

    def to_dict(self) -> dict[str, Any]:
        return asdict(self)

    @property
    def delta_features(self) -> dict[str, float]:
        return {
            "delta_benefit_magnitude": self.delta_benefit_magnitude,
            "delta_harm_magnitude": self.delta_harm_magnitude,
            "delta_benefit_probability": self.delta_benefit_probability,
            "delta_temporal_delay": self.delta_temporal_delay,
            "delta_directness_of_harm": self.delta_directness_of_harm,
            "delta_beneficiary_identified": self.delta_beneficiary_identified,
            "delta_consent_of_harmed_party": self.delta_consent_of_harmed_party,
            "delta_reversibility_of_harm": self.delta_reversibility_of_harm,
        }

    @property
    def presented_options(self) -> tuple[str, str]:
        """Return (first_option, second_option) accounting for order."""
        if self.option_order == "AB":
            return ("A", "B")
        return ("B", "A")


DELTA_FEATURE_NAMES = [
    "delta_benefit_magnitude",
    "delta_harm_magnitude",
    "delta_benefit_probability",
    "delta_temporal_delay",
    "delta_directness_of_harm",
    "delta_beneficiary_identified",
    "delta_consent_of_harmed_party",
    "delta_reversibility_of_harm",
]

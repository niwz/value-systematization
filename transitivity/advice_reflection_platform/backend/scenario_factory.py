from __future__ import annotations

import json
from pathlib import Path
from typing import Any

from .schemas import AdviceOption, ScenarioRecord


def _render_template(text: str, values: dict[str, Any]) -> str:
    return text.format(**values).strip()


def load_family_templates(path: str | Path) -> list[dict[str, Any]]:
    with open(path, encoding="utf-8") as handle:
        payload = json.load(handle)
    if not isinstance(payload, list):
        raise ValueError("family template file must contain a list")
    return payload


def generate_scenarios_from_templates(path: str | Path) -> list[ScenarioRecord]:
    templates = load_family_templates(path)
    scenarios: list[ScenarioRecord] = []
    for family in templates:
        family_id = str(family["family_id"])
        domain = str(family["domain"])
        option_a_template = str(family["option_a_template"])
        option_b_template = str(family["option_b_template"])
        option_a_title = str(family["option_a_title"])
        option_b_title = str(family["option_b_title"])
        default_latents = {str(key): str(value) for key, value in family.get("latent_dimensions", {}).items()}
        for case in family.get("cases", []):
            values = dict(case.get("values", {}))
            scenario_id = f"{family_id}__{case['scenario_suffix']}"
            latent_dimensions = dict(default_latents)
            latent_dimensions.update({str(key): str(value) for key, value in case.get("latent_dimensions", {}).items()})
            scenarios.append(
                ScenarioRecord(
                    scenario_id=scenario_id,
                    family_id=family_id,
                    request_text=_render_template(str(case["request_template"]), values),
                    option_a=AdviceOption(
                        title=option_a_title,
                        text=_render_template(option_a_template, values),
                        stance_tags=[str(item) for item in family.get("option_a_tags", [])],
                    ),
                    option_b=AdviceOption(
                        title=option_b_title,
                        text=_render_template(option_b_template, values),
                        stance_tags=[str(item) for item in family.get("option_b_tags", [])],
                    ),
                    domain=domain,
                    latent_dimensions=latent_dimensions,
                    paraphrase_group=str(case.get("paraphrase_group", scenario_id)),
                    notes=str(case.get("notes", "")).strip(),
                    metadata={"source_template_id": family_id, "surface_form": str(case.get("surface_form", "default"))},
                )
            )
    return scenarios

